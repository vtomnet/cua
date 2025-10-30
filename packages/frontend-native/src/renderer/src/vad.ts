import * as ort from 'onnxruntime-web';

interface VADOptions {
  modelPath: string;
  threshold?: number;
  minSilenceDurationMs?: number;
  speechPadMs?: number;
  samplingRate?: 16000 | 8000;
  returnSeconds?: boolean;
  onSpeechStart?: () => void;
  onSpeechEnd?: () => void;
  onError?: (error: Error) => void;
}

interface SpeechEvent {
  start?: number;
  end?: number;
}

type EventType = 'speech-start' | 'speech-end';
type EventCallback = (timestamp: number) => void;

export class VAD {
  private session: ort.InferenceSession | null = null;
  private state: ort.Tensor;
  private context: Float32Array; // Context buffer for model input
  private contextSize: number; // Size of context (64 for 16kHz, 32 for 8kHz)
  private windowSizeSamples: number; // Fixed window size (512 for 16kHz, 256 for 8kHz)
  private threshold: number;
  private minSilenceSamples: number;
  private speechPadSamples: number;
  private samplingRate: number;
  private returnSeconds: boolean;
  private triggered: boolean = false;
  private tempEnd: number = 0;
  private currentSample: number = 0;
  private eventListeners: Map<EventType, Set<EventCallback>> = new Map();

  // Queue for sequential processing
  private queue: Float32Array[] = [];
  private isProcessing: boolean = false;

  // Callbacks from options
  private onSpeechStartCallback?: () => void;
  private onSpeechEndCallback?: () => void;
  private onErrorCallback?: (error: Error) => void;

  // Model loading state
  private modelPath: string;
  private isReady: boolean = false;
  private initPromise: Promise<void> | null = null;

  constructor(options: VADOptions) {
    this.threshold = options.threshold ?? 0.5;
    this.samplingRate = options.samplingRate ?? 16000;
    this.returnSeconds = options.returnSeconds ?? false;
    const minSilenceDurationMs = options.minSilenceDurationMs ?? 100;
    const speechPadMs = options.speechPadMs ?? 30;
    this.minSilenceSamples = this.samplingRate * minSilenceDurationMs / 1000;
    this.speechPadSamples = this.samplingRate * speechPadMs / 1000;

    // Store callbacks
    this.onSpeechStartCallback = options.onSpeechStart;
    this.onSpeechEndCallback = options.onSpeechEnd;
    this.onErrorCallback = options.onError;

    this.modelPath = options.modelPath;

    if (![8000, 16000].includes(this.samplingRate)) {
      throw new Error('Unsupported sampling rate; must be 8000 or 16000 Hz');
    }

    // Set window size and context size based on sampling rate
    // Reference: silero-vad Python implementation
    this.windowSizeSamples = this.samplingRate === 16000 ? 512 : 256;
    this.contextSize = this.samplingRate === 16000 ? 64 : 32;

    // Initialize recurrent state (zeros)
    this.state = new ort.Tensor('float32', new Float32Array(2 * 1 * 128), [2, 1, 128]);

    // Initialize context buffer (zeros)
    this.context = new Float32Array(this.contextSize);
  }

  /**
   * Initialize the VAD model. Must be called before processing audio.
   * @returns Promise that resolves when the model is loaded
   */
  async init(): Promise<void> {
    // Return existing promise if already initializing
    if (this.initPromise) {
      return this.initPromise;
    }

    // Return immediately if already ready
    if (this.isReady) {
      return Promise.resolve();
    }

    this.initPromise = this.loadModel();
    return this.initPromise;
  }

  private async loadModel(): Promise<void> {
    try {
      // Configure WASM paths for Electron environment
      // The WASM files are in the assets directory relative to index.html
      ort.env.wasm.wasmPaths = './';
      ort.env.wasm.numThreads = 1; // Start with single thread for stability

      this.session = await ort.InferenceSession.create(this.modelPath, {
        executionProviders: ['wasm'],
      });
      this.isReady = true;
      console.log('VAD model loaded successfully');
    } catch (error) {
      console.error('Failed to load ONNX model:', error);
      const err = error instanceof Error ? error : new Error(String(error));
      this.onErrorCallback?.(err);
      throw err;
    }
  }

  resetStates() {
    this.state = new ort.Tensor('float32', new Float32Array(2 * 1 * 128), [2, 1, 128]);
    this.context = new Float32Array(this.contextSize); // Reset context to zeros
    this.triggered = false;
    this.tempEnd = 0;
    this.currentSample = 0;
  }

  // Synchronous process - just queues the chunk
  process(chunk: Float32Array): void {
    this.queue.push(chunk);
    this.processQueue();
  }

  // Asynchronously process queued chunks sequentially
  private async processQueue(): Promise<void> {
    if (this.isProcessing) return;

    this.isProcessing = true;

    while (this.queue.length > 0) {
      const chunk = this.queue.shift();
      if (!chunk) continue;

      try {
        await this.processChunk(chunk);
      } catch (error) {
        console.error('Error processing audio chunk:', error);
        const err = error instanceof Error ? error : new Error(String(error));
        this.onErrorCallback?.(err);
      }
    }

    this.isProcessing = false;
  }

  private async processChunk(chunk: Float32Array): Promise<void> {
    if (!this.session) {
      throw new Error('Model not loaded yet');
    }

    // Validate chunk size - must match the expected window size
    if (chunk.length !== this.windowSizeSamples) {
      throw new Error(
        `Invalid chunk size ${chunk.length} for ${this.samplingRate} Hz. Expected ${this.windowSizeSamples} samples.`
      );
    }

    // Create input by concatenating context + chunk
    // This matches the reference implementation
    const inputSize = this.contextSize + this.windowSizeSamples;
    const inputData = new Float32Array(inputSize);
    inputData.set(this.context, 0); // Copy context to beginning
    inputData.set(chunk, this.contextSize); // Copy chunk after context

    // Prepare model inputs
    const input = new ort.Tensor('float32', inputData, [1, inputSize]);
    const sr = new ort.Tensor('int64', new BigInt64Array([BigInt(this.samplingRate)]), [1]);
    const feeds: { [key: string]: ort.Tensor } = {
      input: input,
      state: this.state,
      sr: sr
    };

    // Run inference
    const outputMap = await this.session.run(feeds);
    const speechProb = outputMap.output.data[0] as number;
    this.state = outputMap.stateN; // Update state for next iteration (note: output is 'stateN')

    // Update context: last contextSize samples from the full input
    this.context.set(inputData.slice(-this.contextSize));

    // Update current sample position (only by the new chunk size, not including context)
    this.currentSample += this.windowSizeSamples;

    // Detection logic (mirrors Python VADIterator)
    let event: SpeechEvent | null = null;

    if (speechProb >= this.threshold && this.tempEnd !== 0) {
      this.tempEnd = 0;
    }

    if (speechProb >= this.threshold && !this.triggered) {
      this.triggered = true;
      let speechStart = this.currentSample - this.speechPadSamples - this.windowSizeSamples;
      if (speechStart < 0) speechStart = 0;
      if (this.returnSeconds) {
        speechStart = Math.round(speechStart / this.samplingRate * 10) / 10; // 1 decimal
      }
      event = { start: speechStart };
    }

    if (speechProb < this.threshold - 0.15 && this.triggered) {
      if (this.tempEnd === 0) {
        this.tempEnd = this.currentSample;
      }
      if (this.currentSample - this.tempEnd >= this.minSilenceSamples) {
        let speechEnd = this.tempEnd + this.speechPadSamples - this.windowSizeSamples;
        if (this.returnSeconds) {
          speechEnd = Math.round(speechEnd / this.samplingRate * 10) / 10;
        }
        event = { end: speechEnd };
        this.tempEnd = 0;
        this.triggered = false;
      }
    }

    if (event) {
      // Call callbacks from constructor options
      if (event.start !== undefined) {
        this.onSpeechStartCallback?.();
        this.emit('speech-start', event.start);
      }
      if (event.end !== undefined) {
        this.onSpeechEndCallback?.();
        this.emit('speech-end', event.end);
      }
    }
  }

  private emit(eventType: EventType, timestamp: number) {
    const listeners = this.eventListeners.get(eventType);
    if (listeners) {
      listeners.forEach(callback => {
        try {
          callback(timestamp);
        } catch (error) {
          console.error(`Error in VAD event listener (${eventType}):`, error);
        }
      });
    }
  }

  /**
   * Register a callback for VAD events
   * @param event - The event type ('speech-start' or 'speech-end')
   * @param callback - Function to call when the event occurs
   */
  on(event: EventType, callback: EventCallback): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(callback);
  }

  /**
   * Unregister a callback for VAD events
   * @param event - The event type ('speech-start' or 'speech-end')
   * @param callback - The callback function to remove
   */
  off(event: EventType, callback: EventCallback): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.delete(callback);
    }
  }

  /**
   * Register a one-time callback for VAD events
   * @param event - The event type ('speech-start' or 'speech-end')
   * @param callback - Function to call once when the event occurs
   */
  once(event: EventType, callback: EventCallback): void {
    const wrappedCallback: EventCallback = (timestamp) => {
      callback(timestamp);
      this.off(event, wrappedCallback);
    };
    this.on(event, wrappedCallback);
  }

  /**
   * Remove all callbacks for a specific event or all events
   * @param event - Optional event type to clear. If not provided, clears all events
   */
  removeAllListeners(event?: EventType): void {
    if (event) {
      this.eventListeners.delete(event);
    } else {
      this.eventListeners.clear();
    }
  }

  // To close the stream when done
  close() {
    this.queue = [];
    this.isProcessing = false;
    this.resetStates();
    this.removeAllListeners();
  }
}
