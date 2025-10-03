import * as ort from "onnxruntime-web";
import llamaTokenizer from 'llama-tokenizer-js';

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
  private initialized: boolean;

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
    this.initialized = false;
  }

  async init(): Promise<void> {
    this.session = await ort.InferenceSession.create(this.modelPath, {
      executionProviders: ["wasm"], // or "webgl"
    });
    this.initialized = true;
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

  constructor(modelName = "moonshine-base") {
    this.modelName = modelName;
    this.model = {};
    this.initialized = false;
  }

  private argMax(array: Float32Array | number[]): number {
    return array.reduce((maxIndex: number, current: number, index: number, arr: Float32Array | number[]) =>
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
        args_1: new ort.Tensor("int32", [seqFeatureDims[1]], [1])
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
        args_0: new ort.Tensor("int32", [[1]], [1, 1]),
        args_1: new ort.Tensor("float32", contextData, contextDims),
        args_2: new ort.Tensor("int32", [seqLen], [1])
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
          args_0: new ort.Tensor("int32", [[nextToken]], [1, 1]),
          args_1: new ort.Tensor("float32", contextData, contextDims),
          args_2: new ort.Tensor("int32", [seqLen], [1])
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

      this.ws.send(JSON.stringify({
        type: "response.create",
        response: {
          conversation: null,
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
            return data;
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
            toolOutput = `Scrolled`;
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

// DOM elements
const startBtn = document.getElementById('startBtn') as HTMLButtonElement;
const stopBtn = document.getElementById('stopBtn') as HTMLButtonElement;
const statusDiv = document.getElementById('status') as HTMLDivElement;
const speechSegmentsDiv = document.getElementById('speechSegments') as HTMLDivElement;
const transcriptionsDiv = document.getElementById('transcriptions') as HTMLDivElement;

// Global VAD and ASR instances and recording state
let vad: VadIterator | null = null;
let moonshine: Moonshine | null = null;
let openaiRealtime: OpenAIRealtime | null = null;
let mediaRecorder: MediaRecorder | null = null;
let audioContext: AudioContext | null = null;
let isRecording = false;
let recordedChunks: Float32Array[] = [];
let lastTranscriptionTime = 0;
let currentSpeechBuffer: Float32Array[] = [];
let speechStartTime = 0;
let silenceTimeout: NodeJS.Timeout | null = null;
let currentTranscription = "";
let messages: Array<any> = [];

// Update status display
function updateStatus(msg: string, className = '') {
  if (statusDiv) {
    statusDiv.textContent = 'Status: ' + msg;
    statusDiv.className = className;
  }
}

// Display speech segments
function displayResults(timestamps: Timestamp[]) {
  if (!speechSegmentsDiv) return;

  if (timestamps.length === 0) {
    speechSegmentsDiv.innerHTML = '<p>No speech segments detected.</p>';
    return;
  }

  const segmentsList = timestamps.map((ts, index) => {
    const startSec = Math.round((ts.start / 16000) * 100) / 100;
    const endSec = Math.round((ts.end / 16000) * 100) / 100;
    return `<div class="segment">Segment ${index + 1}: ${startSec}s - ${endSec}s</div>`;
  }).join('');

  speechSegmentsDiv.innerHTML = segmentsList;
}

// Display transcriptions
function displayTranscription(transcription: string, timestamp: number) {
  if (!transcriptionsDiv) return;

  const timeStr = new Date(timestamp).toLocaleTimeString();
  const transcriptionElement = document.createElement('div');
  transcriptionElement.className = 'transcription';
  transcriptionElement.innerHTML = `<strong>[${timeStr}]</strong> ${transcription}`;

  transcriptionsDiv.appendChild(transcriptionElement);
  transcriptionsDiv.scrollTop = transcriptionsDiv.scrollHeight;
}

// Display OpenAI responses
function displayOpenAIResponse(response: any, timestamp: number) {
  if (!transcriptionsDiv) return;

  const responseText = response;
  // const responseText = JSON.parse(response).response.output[0].content[0].text;
  const timeStr = new Date(timestamp).toLocaleTimeString();
  const responseElement = document.createElement('div');
  responseElement.className = 'openai-response';
  responseElement.innerHTML = `<strong>[${timeStr}] GPT:</strong> ${responseText}`;
  responseElement.style.backgroundColor = '#e8f4fd';
  responseElement.style.padding = '8px';
  responseElement.style.marginTop = '4px';
  responseElement.style.borderRadius = '4px';

  transcriptionsDiv.appendChild(responseElement);
  transcriptionsDiv.scrollTop = transcriptionsDiv.scrollHeight;
}

// Send transcription to OpenAI and get response
async function sendToOpenAI(transcription: string) {
  if (!openaiRealtime || !transcription.trim()) return;

  try {
    console.log('Sending to OpenAI:', transcription);
    const response = await openaiRealtime.sendTextAndGetResponse(transcription);
    console.log('OpenAI response:', response);
    displayOpenAIResponse(response, Date.now());
  } catch (error) {
    console.error('Error getting OpenAI response:', error);
    displayOpenAIResponse('Error getting response from OpenAI', Date.now());
  }
}

// Handle transcription for speech segment
async function transcribeSpeech(audioData: Float32Array) {
  if (!moonshine) return;

  try {
    const transcription = await moonshine.generate(audioData);
    if (transcription.trim()) {
      displayTranscription(transcription.trim(), Date.now());

      // Accumulate transcription for OpenAI
      currentTranscription += (currentTranscription ? ' ' : '') + transcription.trim();

      // Clear any existing timeout and set new one for 1.5s
      if (silenceTimeout) {
        clearTimeout(silenceTimeout);
      }

      silenceTimeout = setTimeout(async () => {
        if (currentTranscription.trim()) {
          await sendToOpenAI(currentTranscription.trim());
          currentTranscription = "";
        }
      }, 1500); // 1.5 seconds after speech ends
    }
  } catch (error) {
    console.error('Error transcribing speech:', error);
  }
}

// Process audio chunk for real-time VAD and transcription
let speechDetected = false;
let speechStarted = false;
let silenceCounter = 0;
const SILENCE_THRESHOLD = 10; // chunks of silence to detect speech end

async function processAudioChunk(audioChunk: Float32Array) {
  if (!vad) return;

  // Add chunk to current speech buffer
  currentSpeechBuffer.push(audioChunk);

  // Process chunk with VAD to detect speech activity
  try {
    // Use a sliding window approach for real-time detection
    const windowSize = 512; // Match VAD window size
    for (let i = 0; i < audioChunk.length; i += windowSize) {
      const chunk = audioChunk.slice(i, Math.min(i + windowSize, audioChunk.length));
      if (chunk.length === windowSize) {
        // Simple VAD prediction on chunk (this is a simplified approach)
        await vad.predict(chunk);
      }
    }

    // Check VAD state for speech activity
    const hasCurrentSpeech = vad.triggered;

    if (hasCurrentSpeech && !speechStarted) {
      // Speech started
      speechStarted = true;
      silenceCounter = 0;
      speechStartTime = Date.now();
    } else if (!hasCurrentSpeech && speechStarted) {
      // Potential speech end, count silence
      silenceCounter++;

      if (silenceCounter >= SILENCE_THRESHOLD) {
        // Speech ended, transcribe the current buffer
        const currentAudio = combineAudioChunks(currentSpeechBuffer);
        if (currentAudio.length > 0) {
          await transcribeSpeech(currentAudio);
          currentSpeechBuffer = [];
          lastTranscriptionTime = Date.now();
        }
        speechStarted = false;
        silenceCounter = 0;
      }
    } else if (hasCurrentSpeech) {
      // Reset silence counter if speech continues
      silenceCounter = 0;
    }

    // Check for 5-second interval transcription
    const now = Date.now();
    if (now - lastTranscriptionTime >= 5000 && currentSpeechBuffer.length > 0) {
      // Check if there's ongoing speech to transcribe
      const currentAudio = combineAudioChunks(currentSpeechBuffer);
      if (currentAudio.length > 0) {
        await transcribeSpeech(currentAudio);
        currentSpeechBuffer = []; // Clear buffer after transcription
        lastTranscriptionTime = now;
      }
    }

  } catch (error) {
    console.error('Error processing audio chunk:', error);
  }
}

// Helper function to combine audio chunks into a single Float32Array
function combineAudioChunks(chunks: Float32Array[]): Float32Array {
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

// Initialize VAD and Moonshine ASR
async function initializeModels() {
  try {
    updateStatus('Initializing VAD model...', 'loading');
    vad = new VadIterator("/models/silero_vad.onnx");
    await vad.init();
    updateStatus('VAD model loaded, loading ASR...', 'loading');

    moonshine = new Moonshine("moonshine-base");
    await moonshine.loadModel();
    updateStatus('ASR loaded, connecting to OpenAI...', 'loading');

    const apiKey = import.meta.env.VITE_OPENAI_API_KEY;

    openaiRealtime = new OpenAIRealtime(apiKey);
    await openaiRealtime.connect();
    updateStatus('All models loaded and connected successfully', 'success');
    return true;
  } catch (error) {
    console.error('Failed to initialize models:', error);
    updateStatus('Failed to load models or connect to OpenAI', 'error');
    return false;
  }
}

// Start microphone recording
async function startRecording() {
  try {
    updateStatus('Requesting microphone access...', 'loading');

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true
      }
    });

    audioContext = new AudioContext({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(stream);

    // Initialize models if not already done
    if (!vad || !moonshine) {
      const initialized = await initializeModels();
      if (!initialized) return;
    }

    // Create audio worklet for real-time processing
    await audioContext.audioWorklet.addModule(createAudioProcessorBlob());
    const processorNode = new AudioWorkletNode(audioContext, 'vad-processor');

    processorNode.port.onmessage = async (event) => {
      const { audioData } = event.data;
      const audioChunk = new Float32Array(audioData);
      recordedChunks.push(audioChunk);

      // Real-time processing for speech detection and transcription
      await processAudioChunk(audioChunk);
    };

    source.connect(processorNode);
    processorNode.connect(audioContext.destination);

    isRecording = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    recordedChunks = [];
    currentSpeechBuffer = [];
    lastTranscriptionTime = Date.now();
    speechStarted = false;
    silenceCounter = 0;

    updateStatus('Recording... Speak into your microphone', 'recording');

  } catch (error) {
    console.error('Error starting recording:', error);
    updateStatus('Failed to access microphone', 'error');
  }
}

// Stop recording and process audio
async function stopRecording() {
  if (!isRecording || !audioContext) return;

  isRecording = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;

  // Clear any pending timeout
  if (silenceTimeout) {
    clearTimeout(silenceTimeout);
    silenceTimeout = null;
  }

  // Send any remaining transcription to OpenAI
  if (currentTranscription.trim()) {
    await sendToOpenAI(currentTranscription.trim());
    currentTranscription = "";
  }

  // Stop audio context
  await audioContext.close();
  audioContext = null;

  if (recordedChunks.length === 0) {
    updateStatus('No audio recorded', 'error');
    return;
  }

  try {
    updateStatus('Processing recorded audio...', 'loading');

    // Combine all recorded chunks
    const totalLength = recordedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const combinedAudio = new Float32Array(totalLength);
    let offset = 0;

    for (const chunk of recordedChunks) {
      combinedAudio.set(chunk, offset);
      offset += chunk.length;
    }

    // Process with VAD
    await vad!.process(combinedAudio);
    const timestamps = vad!.getSpeechTimestamps();
    displayResults(timestamps);

    updateStatus(`Processing complete - ${timestamps.length} speech segments found`, 'success');
  } catch (error) {
    console.error('Error processing recorded audio:', error);
    updateStatus('Error processing recorded audio', 'error');
  }
}

// Create audio processor worklet as blob
function createAudioProcessorBlob(): string {
  const processorCode = `
    class VADProcessor extends AudioWorkletProcessor {
      constructor() {
        super();
        this.bufferSize = 512;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
      }

      process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
          const inputChannel = input[0];

          for (let i = 0; i < inputChannel.length; i++) {
            this.buffer[this.bufferIndex] = inputChannel[i];
            this.bufferIndex++;

            if (this.bufferIndex >= this.bufferSize) {
              // Send buffer to main thread
              this.port.postMessage({
                audioData: Array.from(this.buffer)
              });

              this.bufferIndex = 0;
            }
          }
        }

        return true;
      }
    }

    registerProcessor('vad-processor', VADProcessor);
  `;

  return URL.createObjectURL(new Blob([processorCode], { type: 'application/javascript' }));
}

// Event listeners
startBtn?.addEventListener('click', startRecording);
stopBtn?.addEventListener('click', stopRecording);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  updateStatus('Ready - click Start Recording to begin');
});
