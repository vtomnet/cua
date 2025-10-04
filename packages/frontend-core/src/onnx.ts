import * as ort from "onnxruntime-web";
import llamaTokenizer from 'llama-tokenizer-js';
import { WhisperFeatureExtractor } from 'whisper-feature-extractor';

// Utility: load WAV file (16kHz mono PCM) into Float32Array
export async function loadWavFile(url: string): Promise<Float32Array> {
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  const audioCtx = new AudioContext({ sampleRate: 16000 });
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  return audioBuffer.getChannelData(0); // mono channel
}

// Timestamp class
export class Timestamp {
  start: number;
  end: number;

  constructor(start = -1, end = -1) {
    this.start = start;
    this.end = end;
  }

  toString(): string {
    return `{start:${this.start}, end:${this.end}}`;
  }
}

// Ring buffer for audio data to prevent losing first bit of speech
export class RingBuffer {
  private buffer: Float32Array;
  private writeIndex: number;
  private size: number;
  private isFull: boolean;

  constructor(sizeInSamples: number) {
    this.size = sizeInSamples;
    this.buffer = new Float32Array(sizeInSamples);
    this.writeIndex = 0;
    this.isFull = false;
  }

  // Add audio data to the ring buffer
  write(data: Float32Array): void {
    for (let i = 0; i < data.length; i++) {
      this.buffer[this.writeIndex] = data[i];
      this.writeIndex = (this.writeIndex + 1) % this.size;

      // Mark as full when we've written enough data to fill the buffer once
      if (this.writeIndex === 0) {
        this.isFull = true;
      }
    }
  }

  // Get all buffered data in the correct order
  read(): Float32Array {
    if (!this.isFull) {
      // If buffer isn't full yet, return data from start to writeIndex
      return this.buffer.slice(0, this.writeIndex);
    }

    // If buffer is full, return data in correct chronological order
    const result = new Float32Array(this.size);
    const firstPart = this.buffer.slice(this.writeIndex);
    const secondPart = this.buffer.slice(0, this.writeIndex);

    result.set(firstPart, 0);
    result.set(secondPart, firstPart.length);

    return result;
  }

  // Get the current amount of data stored
  getStoredSamples(): number {
    return this.isFull ? this.size : this.writeIndex;
  }

  // Clear the buffer
  clear(): void {
    this.buffer.fill(0);
    this.writeIndex = 0;
    this.isFull = false;
  }
}

// VadIterator class
export class VadIterator {
  private sampleRate: number;
  private threshold: number;
  private speechPadSamples: number;
  private srPerMs: number;

  private windowSizeSamples: number;
  private contextSamples: number;
  private effectiveWindowSize: number;

  private minSpeechSamples: number;
  private maxSpeechSamples: number;
  private minSilenceSamples: number;
  private minSilenceSamplesAtMaxSpeech: number;

  private stateSize: number;
  private _state: Float32Array;
  private _context: Float32Array;

  private tempEnd: number;
  private currentSample: number;
  private prevEnd: number;
  private nextStart: number;
  private speeches: Timestamp[];
  private currentSpeech: Timestamp;

  private modelPath: string;
  private session: ort.InferenceSession | null;

  public triggered: boolean;

  constructor(
    modelPath: string,
    sampleRate = 16000,
    windowMs = 32,
    threshold = 0.5,
    minSilenceMs = 100,
    speechPadMs = 30,
    minSpeechMs = 250,
    maxSpeechSec = Infinity
  ) {
    this.sampleRate = sampleRate;
    this.threshold = threshold;
    this.speechPadSamples = speechPadMs * (sampleRate / 1000);
    this.srPerMs = sampleRate / 1000;

    this.windowSizeSamples = windowMs * this.srPerMs; // e.g. 32ms * 16 = 512
    this.contextSamples = 64;
    this.effectiveWindowSize = this.windowSizeSamples + this.contextSamples;

    this.minSpeechSamples = this.srPerMs * minSpeechMs;
    this.maxSpeechSamples = sampleRate * maxSpeechSec - this.windowSizeSamples - 2 * this.speechPadSamples;
    this.minSilenceSamples = this.srPerMs * minSilenceMs;
    this.minSilenceSamplesAtMaxSpeech = this.srPerMs * 98;

    this.stateSize = 2 * 1 * 128;
    this._state = new Float32Array(this.stateSize);
    this._context = new Float32Array(this.contextSamples);

    this.triggered = false;
    this.tempEnd = 0;
    this.currentSample = 0;
    this.prevEnd = 0;
    this.nextStart = 0;
    this.speeches = [];
    this.currentSpeech = new Timestamp();

    this.modelPath = modelPath;
    this.session = null;
  }

  async init(): Promise<void> {
    this.session = await ort.InferenceSession.create(this.modelPath, {
      executionProviders: ["wasm"], // or "webgl"
    });
  }

  reset(): void {
    this._state.fill(0);
    this._context.fill(0);
    this.triggered = false;
    this.tempEnd = 0;
    this.currentSample = 0;
    this.prevEnd = 0;
    this.nextStart = 0;
    this.speeches = [];
    this.currentSpeech = new Timestamp();
  }

  async predict(chunk: Float32Array): Promise<void> {
    if (!this.session) throw new Error("VAD session not initialized.");

    // Add context before current chunk
    const newData = new Float32Array(this.effectiveWindowSize);
    newData.set(this._context, 0);
    newData.set(chunk, this.contextSamples);

    // Update context with last samples
    this._context.set(newData.slice(-this.contextSamples));

    const inputTensor = new ort.Tensor("float32", newData, [1, this.effectiveWindowSize]);
    const stateTensor = new ort.Tensor("float32", this._state, [2, 1, 128]);
    const srTensor = new ort.Tensor("int64", BigInt64Array.from([BigInt(this.sampleRate)]), [1]);

    const feeds: Record<string, ort.Tensor> = {
      input: inputTensor,
      state: stateTensor,
      sr: srTensor,
    };

    const results = await this.session.run(feeds);
    const speechProb: number = (results.output.data as Float32Array)[0];
    const newState = results.stateN.data as Float32Array;
    this._state.set(newState);

    this.currentSample += this.windowSizeSamples;

    // ==== speech detection logic ====
    if (speechProb >= this.threshold) {
      if (this.tempEnd !== 0) {
        this.tempEnd = 0;
        if (this.nextStart < this.prevEnd) {
          this.nextStart = this.currentSample - this.windowSizeSamples;
        }
      }
      if (!this.triggered) {
        this.triggered = true;
        this.currentSpeech.start = this.currentSample - this.windowSizeSamples;
      }
      return;
    }

    if (this.triggered && (this.currentSample - this.currentSpeech.start) > this.maxSpeechSamples) {
      this.currentSpeech.end = this.currentSample;
      this.speeches.push(this.currentSpeech);
      this.currentSpeech = new Timestamp();
      this.triggered = false;
      this.prevEnd = this.nextStart = this.tempEnd = 0;
      return;
    }

    if (speechProb < (this.threshold - 0.15)) {
      if (this.triggered) {
        if (this.tempEnd === 0) this.tempEnd = this.currentSample;
        if (this.currentSample - this.tempEnd > this.minSilenceSamplesAtMaxSpeech) {
          this.prevEnd = this.tempEnd;
        }
        if ((this.currentSample - this.tempEnd) >= this.minSilenceSamples) {
          this.currentSpeech.end = this.tempEnd;
          if (this.currentSpeech.end - this.currentSpeech.start > this.minSpeechSamples) {
            this.speeches.push(this.currentSpeech);
          }
          this.currentSpeech = new Timestamp();
          this.prevEnd = this.nextStart = this.tempEnd = 0;
          this.triggered = false;
        }
      }
    }
  }

  async process(inputWav: Float32Array): Promise<void> {
    this.reset();
    for (let i = 0; i + this.windowSizeSamples <= inputWav.length; i += this.windowSizeSamples) {
      const chunk = inputWav.slice(i, i + this.windowSizeSamples);
      await this.predict(chunk);
    }
    if (this.currentSpeech.start >= 0) {
      this.currentSpeech.end = inputWav.length;
      this.speeches.push(this.currentSpeech);
    }
  }

  getSpeechTimestamps(): Timestamp[] {
    return this.speeches;
  }
}

// Smart Turn v3 class for turn detection
export class SmartTurnV3 {
  private modelPath: string;
  private session: ort.InferenceSession | null = null;
  private featureExtractor: WhisperFeatureExtractor;
  private sampleRate: number;
  private nSeconds: number;
  private initialized: boolean = false;

  constructor(
    modelPath = "/models/smart-turn-v3.0.onnx",
    chunkLength = 8,
    sampleRate = 16000,
    nSeconds = 8
  ) {
    this.modelPath = modelPath;
    this.sampleRate = sampleRate;
    this.nSeconds = nSeconds;
    this.featureExtractor = new WhisperFeatureExtractor({ chunkLength: chunkLength });
  }

  async init(): Promise<void> {
    try {
      this.session = await ort.InferenceSession.create(this.modelPath, {
        executionProviders: ["wasm"]
      });
      this.initialized = true;
    } catch (error) {
      console.error("Failed to initialize Smart Turn v3 model:", error);
      throw error;
    }
  }

  private truncateAudioToLastNSeconds(audioArray: Float32Array): Float32Array {
    const targetSamples = this.nSeconds * this.sampleRate;

    if (audioArray.length >= targetSamples) {
      // Keep last N seconds
      return audioArray.slice(-targetSamples);
    } else {
      // Pad at the front with zeros
      const padded = new Float32Array(targetSamples);
      padded.set(audioArray, targetSamples - audioArray.length);
      return padded;
    }
  }

  async predictEndpoint(audioArray: Float32Array): Promise<{ prediction: number; probability: number }> {
    if (!this.session || !this.initialized) {
      throw new Error("Smart Turn v3 model not initialized");
    }

    try {
      // Step 1: Ensure audio length equals 8 seconds
      const processedAudio = this.truncateAudioToLastNSeconds(audioArray);

      // Step 2: Extract features using WhisperFeatureExtractor
      // Convert Float32Array to number[] for compatibility with the library
      const audioAsNumbers = Array.from(processedAudio);
      const features = this.featureExtractor.extractFeatures(audioAsNumbers, {
        samplingRate: this.sampleRate,
        returnTensors: "np",
        padding: "maxLength",
        maxLength: this.nSeconds * this.sampleRate,
        truncation: true,
        doNormalize: true
      });

      // Step 3: Prepare input tensor for ONNX
      // Get features from BatchFeature object
      const inputFeaturesArray = features.get("input_features") as number[][][];

      // Flatten to Float32Array for ONNX (assuming single batch item)
      const inputFeatures = new Float32Array(inputFeaturesArray[0].flat());

      // Add batch dimension to make it [1, feat_dim, seq_len] to match model expectations
      const inputTensor = new ort.Tensor("float32", inputFeatures, [1, 80, inputFeatures.length / 80]); // 80-dim features, transposed

      // Step 4: Run inference
      const outputs = await this.session.run({
        input_features: inputTensor
      });

      // Step 5: Extract probability and make prediction
      const outputKey = Object.keys(outputs)[0];
      const probability = (outputs[outputKey].data as Float32Array)[0];
      const prediction = probability > 0.5 ? 1 : 0;

      return {
        prediction,
        probability
      };
    } catch (error) {
      console.error("Error in Smart Turn v3 prediction:", error);
      throw error;
    }
  }
}

// Moonshine ASR class
export class Moonshine {
  private modelName: string;
  private model: {
    preprocess?: ort.InferenceSession;
    encode?: ort.InferenceSession;
    uncached_decode?: ort.InferenceSession;
    cached_decode?: ort.InferenceSession;
  };
  private initialized: boolean;
  private minSamples: number;

  constructor(modelName = "moonshine-base") {
    this.modelName = modelName;
    this.model = {};
    this.initialized = false;
    this.minSamples = 2048; // guard to avoid extremely short inputs that break conv layers
  }

  private argMax(array: Float32Array | number[]): number {
    return Array.from(array).reduce((maxIndex: number, current: number, index: number, arr: number[]) =>
      current > arr[maxIndex] ? index : maxIndex, 0);
  }

  async loadModel(): Promise<void> {
    try {
      console.log(`Loading ${this.modelName} model...`);

      // Load all four ONNX sessions
      console.log("Loading preprocess...");
      this.model.preprocess = await ort.InferenceSession.create(
        `/models/${this.modelName}/preprocess.onnx`,
        { executionProviders: ["wasm", "cpu"] }
      );
      console.log("preprocess loaded");

      console.log("Loading encode...");
      this.model.encode = await ort.InferenceSession.create(
        `/models/${this.modelName}/encode.onnx`,
        { executionProviders: ["wasm", "cpu"] }
      );
      console.log("encode loaded");

      console.log("Loading uncached_decode...");
      this.model.uncached_decode = await ort.InferenceSession.create(
        `/models/${this.modelName}/uncached_decode.onnx`,
        { executionProviders: ["wasm", "cpu"] }
      );
      console.log("uncached_decode loaded");

      console.log("Loading cached_decode...");
      this.model.cached_decode = await ort.InferenceSession.create(
        `/models/${this.modelName}/cached_decode.onnx`,
        { executionProviders: ["wasm", "cpu"] }
      );
      console.log("cached_decode loaded");

      this.initialized = true;
      console.log(`${this.modelName} model loaded successfully`);
    } catch (error) {
      console.error("Failed to load moonshine model:", error);
      throw error;
    }
  }

  async generate(audio: Float32Array): Promise<string> {
    if (!this.initialized || !this.model.preprocess || !this.model.encode ||
        !this.model.uncached_decode || !this.model.cached_decode) {
      console.warn("Tried to call Moonshine.generate() before the model was loaded.");
      return "";
    }

    // Skip inference if too few samples; accumulate more audio first
    if (audio.length < this.minSamples) {
      return "";
    }

    try {
      // Calculate max decoding length from audio duration
      const maxLen = Math.floor((audio.length / 16000) * 6);

      // Step 1: Preprocess - outputs "sequential" with data and dims
      const preprocessResult = await this.model.preprocess.run({
        args_0: new ort.Tensor("float32", audio, [1, audio.length])
      });

      const seqFeatureData = preprocessResult["sequential"]["data"];
      const seqFeatureDims = preprocessResult["sequential"]["dims"];

      // Step 2: Encode - find layer_norm output
      const encodeResult = await this.model.encode.run({
        args_0: new ort.Tensor("float32", seqFeatureData, seqFeatureDims),
        args_1: new ort.Tensor("int32", new Int32Array([seqFeatureDims[1]]), [1])
      });

      // Find context tensor (key starting with "layer_norm")
      let layerNormKey = "";
      for (const key in encodeResult) {
        if (key.startsWith("layer_norm")) {
          layerNormKey = key;
          break;
        }
      }

      if (!layerNormKey) {
        throw new Error('encode output missing "layer_norm*" key');
      }

      const contextData = encodeResult[layerNormKey]["data"];
      const contextDims = encodeResult[layerNormKey]["dims"];

      // Step 3: Initial uncached decode
      let seqLen = 1;
      let decode = await this.model.uncached_decode.run({
        args_0: new ort.Tensor("int32", new Int32Array([1]), [1, 1]),
        args_1: new ort.Tensor("float32", contextData, contextDims),
        args_2: new ort.Tensor("int32", new Int32Array([seqLen]), [1])
      });

      // Step 4: Autoregressive decoding loop
      const tokens = [1]; // Start token

      for (let i = 0; i < maxLen; i++) {
        // Extract logits from reversible_embedding
        const logits = decode["reversible_embedding"]["data"] as Float32Array;
        const nextToken = this.argMax(logits);

        // Check for end token
        if (nextToken === 2) break;

        tokens.push(nextToken);
        seqLen++;

        // Prepare feed for cached decode
        const feed: Record<string, ort.Tensor> = {
          args_0: new ort.Tensor("int32", new Int32Array([nextToken]), [1, 1]),
          args_1: new ort.Tensor("float32", contextData, contextDims),
          args_2: new ort.Tensor("int32", new Int32Array([seqLen]), [1])
        };

        // Add non-reversible tensors from previous decode
        let argIndex = 3;
        for (const key of Object.keys(decode)) {
          if (!key.startsWith("reversible")) {
            feed[`args_${argIndex}`] = decode[key];
            argIndex++;
          }
        }

        // Run cached decode
        decode = await this.model.cached_decode.run(feed);
      }

      // Step 5: Decode tokens to text
      const transcription = llamaTokenizer.decode(tokens);
      return transcription;

    } catch (error) {
      console.error("Error during moonshine generation:", error);
      return "";
    }
  }
}

// OpenAI Realtime Transcription API integration
export class OpenAIRealtimeTranscription {
  private ws: WebSocket | null = null;
  private apiKey: string;
  private sampleRate: number;
  private initialized: boolean = false;
  private minSamples: number;

  constructor(apiKey: string, sampleRate = 24000) {
    this.apiKey = apiKey;
    this.sampleRate = sampleRate;
    // Minimum 100ms of audio required by OpenAI API
    this.minSamples = Math.ceil((sampleRate * 100) / 1000); // 100ms worth of samples
  }

  async init(): Promise<void> {
    try {
      const uri = "wss://api.openai.com/v1/realtime?intent=transcription";

      this.ws = await new Promise((resolve, reject) => {
        // Retry with subprotocol authentication
        const wsWithAuth = new WebSocket(uri, ["realtime", `openai-insecure-api-key.${this.apiKey}`]);

        wsWithAuth.onopen = () => {
          console.log('Connected to OpenAI realtime transcription API');
          resolve(wsWithAuth);
        };

        wsWithAuth.onerror = (error) => {
          console.error('WebSocket connection error:', error);
          reject(new Error('Failed to connect to OpenAI realtime transcription API'));
        };

        wsWithAuth.onclose = (event) => {
          if (!this.initialized) {
            reject(new Error(`WebSocket closed during initialization: ${event.code} ${event.reason}`));
          }
        };
      });

      // Send session configuration for transcription
      const sessionUpdate = {
        "type": "session.update",
        "session": {
          "type": "transcription",
          "audio": {
            "input": {
              "format": {
                "rate": this.sampleRate,
                "type": "audio/pcm"
              },
              "noise_reduction": {
                "type": "far_field"
              },
              "transcription": {
                "language": "en",
                "model": "gpt-4o-mini-transcribe"
              },
              "turn_detection": null
            }
          },
          "include": [
            "item.input_audio_transcription.logprobs"
          ]
        }
      };

      if (this.ws) {
        this.ws.send(JSON.stringify(sessionUpdate));
        this.initialized = true;
      } else {
        throw new Error('WebSocket connection failed');
      }
    } catch (error) {
      console.error('Failed to initialize OpenAI realtime transcription:', error);
      throw error;
    }
  }

  async transcribe(audioData: Float32Array): Promise<string> {
    if (!this.ws || !this.initialized) {
      throw new Error('OpenAI realtime transcription not initialized');
    }

    // Check if we have enough audio data (minimum 100ms)
    if (audioData.length < this.minSamples) {
      console.warn(`Audio too short for OpenAI transcription: ${audioData.length} samples (${(audioData.length / this.sampleRate * 1000).toFixed(1)}ms), minimum required: ${this.minSamples} samples (100ms)`);
      return ""; // Return empty string for short audio
    }

    return new Promise((resolve, reject) => {
      // Convert Float32Array to PCM format (16-bit)
      const pcmData = new Int16Array(audioData.length);
      for (let i = 0; i < audioData.length; i++) {
        pcmData[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32767));
      }

      // Convert to base64 in chunks to avoid stack overflow
      const bytes = new Uint8Array(pcmData.buffer);
      let binaryString = '';
      const chunkSize = 8192; // Process in 8KB chunks
      for (let i = 0; i < bytes.length; i += chunkSize) {
        const chunk = bytes.slice(i, i + chunkSize);
        binaryString += String.fromCharCode(...chunk);
      }
      const b64Data = btoa(binaryString);

      // Send audio data
      this.ws!.send(JSON.stringify({
        type: "input_audio_buffer.append",
        audio: b64Data,
      }));

      // Commit the audio buffer
      this.ws!.send(JSON.stringify({
        type: "input_audio_buffer.commit",
      }));

      // Listen for transcription response
      const messageHandler = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "conversation.item.input_audio_transcription.completed") {
            this.ws!.removeEventListener('message', messageHandler);

            // Clear the buffer for next transcription
            this.ws!.send(JSON.stringify({
              type: "input_audio_buffer.clear",
            }));

            resolve(data.transcript || "");
          } else if (data.type === "error") {
            this.ws!.removeEventListener('message', messageHandler);
            reject(new Error(`OpenAI transcription error: ${data.error?.message || 'Unknown error'}`));
          }
        } catch (error) {
          this.ws!.removeEventListener('message', messageHandler);
          reject(error);
        }
      };

      if (this.ws) {
        this.ws.addEventListener('message', messageHandler);

        // Set timeout for transcription
        setTimeout(() => {
          if (this.ws) {
            this.ws.removeEventListener('message', messageHandler);
          }
          reject(new Error('Transcription timeout'));
        }, 30000);
      } else {
        reject(new Error('WebSocket not available'));
      }
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
      this.initialized = false;
    }
  }
}

// OpenAI Realtime API integration
export class OpenAIRealtime {
  private ws: WebSocket | null = null;
  private apiKey: string;
  private model: string;
  private tools: Array<{}>;
  private messages: Array<{}> = [];

  constructor(apiKey: string, model = "gpt-4o-mini-realtime-preview-2024-12-17") {
    this.apiKey = apiKey;
    this.model = model;
    this.tools = [{
      // TODO: "new window" bool option, akin to open(1) cli util
      type: "function",
      name: "open",
      description: "Open a given URL, application, or file.",
      parameters: {
        type: "object",
        properties: {
          thing: {
            type: "string"
          }
        },
        required: ["thing"],
      },
    }, {
      type: "function",
      name: "scroll",
      description: "Scroll on the current window. The distance is a percentage of the current view, default 70.",
      parameters: {
        type: "object",
        properties: {
          direction: {
            type: "string",
            enum: ["up", "down", "left", "right"]
          },
          distance: {
            type: "number",
          }
        },
        required: ["direction"]
      },
    }, {
      type: "function",
      name: "keys",
      parameters: {
        description: 'Send a list of keypresses. You can use <ctrl>, <shift>, etc. E.g., "<cmd>+c". You may also pass a string of characters to type it, like "Hello world".',
        type: "object",
        properties: {
          list: {
            type: "array",
            items: {
              type: "string"
            }
          }
        }
      },
    }];
  }

  async connect(): Promise<void> {
    // Use OpenAI's official WebSocket authentication method
    const uri = `wss://api.openai.com/v1/realtime?model=${this.model}`;

    // Create WebSocket with proper authentication subprotocols
    const protocols = [
      "realtime",
      `openai-insecure-api-key.${this.apiKey}`
    ];

    try {
      this.ws = new WebSocket(uri, protocols);

      await new Promise((resolve, reject) => {
        if (!this.ws) return reject(new Error('WebSocket failed to initialize'));

        let isConnected = false;

        this.ws.onopen = async () => {
          console.log('WebSocket connected to OpenAI realtime API');
          isConnected = true;

          try {
            // Send session configuration for text-only mode
            const sessionUpdate = {
              type: "session.update",
              session: {
                type: "realtime",
                audio: {
                  input: {
                    turn_detection: null
                  }
                },
                tools: this.tools,
                output_modalities: ["text"]
              }
            };

            this.ws!.send(JSON.stringify(sessionUpdate));
            resolve(undefined);
          } catch (error) {
            reject(error);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          if (!isConnected) {
            reject(new Error('WebSocket connection failed - falling back to REST API'));
          }
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket closed:', event.code, event.reason);
          if (!isConnected) {
            reject(new Error('WebSocket connection failed - falling back to REST API'));
          }
        };

        // Timeout after 10 seconds for connection
        setTimeout(() => {
          if (!isConnected) {
            this.ws?.close();
            reject(new Error('WebSocket connection timeout - falling back to REST API'));
          }
        }, 10000);
      });
    } catch (error) {
      console.warn('WebSocket connection failed, using REST API fallback:', error);
      this.ws = null;
    }
  }

  async sendTextAndGetResponse(text: string): Promise<string> {
    return new Promise((resolve, reject) => {
      if (!this.ws) return reject(new Error('WebSocket not available'));

      this.messages.push({
        type: "message",
        role: "user",
        content: [{
          type: "input_text",
          text: text
        }]
      });

      console.log("Sending messages:", this.messages);

      this.ws.send(JSON.stringify({
        type: "response.create",
        response: {
          conversation: "none",
          input: this.messages,
        }
      }));

      // Listen for response
      const messageHandler = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          console.log('OpenAI WebSocket message:', data.type, data);

          if (data.type === "response.done") {
            this.ws!.removeEventListener('message', messageHandler);
            this.messages.push(...data.response.output);
            this.executeTools(data.response);
            resolve(data);
          } else if (data.type === "error") {
            this.ws!.removeEventListener('message', messageHandler);
            reject(new Error(`OpenAI API error: ${data.error?.message || 'Unknown error'}`));
          }
        } catch (error) {
          this.ws!.removeEventListener('message', messageHandler);
          reject(error);
        }
      };

      this.ws.addEventListener('message', messageHandler);

      // Set timeout for response
      setTimeout(() => {
        this.ws!.removeEventListener('message', messageHandler);
        reject(new Error('Response timeout'));
      }, 30000);
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  executeTools(response: any) {
    for (const item of response.output) {
      if (item.type === "function_call") {
        const args = JSON.parse(item.arguments);
        let toolOutput = "";
        switch (item.name) {
          case "open":
            console.log(`TOOL: open(${JSON.stringify(args)})`);
            toolOutput = `Opened ${args.thing}`;
            break;
          case "scroll":
            console.log(`TOOL: scroll(${JSON.stringify(args)})`);
            toolOutput = `Scrolled.`; // TODO later, determine whether the page actually scrolled and inform the LLM of that.
            break;
          case "keys":
            console.log(`TOOL: keys(${JSON.stringify(args)})`);
            toolOutput = `Pressed the keys`;
            break;
          default:
            const msg = `Bad tool: ${item.name}`;
            console.warn(msg);
            toolOutput = `ERROR: ${msg}`;
        }
        this.messages.push({
          type: "function_call_output",
          output: toolOutput,
          call_id: item.call_id,
        });
      }
    }
  }
}

// Helper function to combine audio chunks into a single Float32Array
export function combineAudioChunks(chunks: Float32Array[]): Float32Array {
  if (chunks.length === 0) return new Float32Array(0);

  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const combined = new Float32Array(totalLength);
  let offset = 0;

  for (const chunk of chunks) {
    combined.set(chunk, offset);
    offset += chunk.length;
  }

  return combined;
}
