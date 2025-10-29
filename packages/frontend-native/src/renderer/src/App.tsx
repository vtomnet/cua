import { useEffect, useRef, useState } from "react";
// import { library } from "@fortawesome/fontawesome-svg-core";
// import { fas } from "@fortawesome/free-solid-svg-icons";
// import { far } from "@fortawesome/free-regular-svg-icons";
// import { fab } from "@fortawesome/free-brands-svg-icons";
import {
  VadIterator,
  SmartTurnV3,
  OpenAIRealtimeTranscription,
  RingBuffer,
} from "frontend-core";
import "./app.css";
import Cursor from "./components/Cursor";

// library.add(fas, far, fab);

type TranscriptionJob = {
  data: Float32Array;
  reason: "speech_end" | "timer";
};

const MIN_TRANSCRIBE_SAMPLES = Math.ceil(16000 * 0.1);
const SILENCE_THRESHOLD = 10;
const MAX_AUDIO_BUFFER_CHUNKS = 500; // Limit buffer growth (~25 seconds at 512 samples/chunk)
const MAX_TURN_DETECTION_BUFFER_SAMPLES = 8 * 16000; // 8 seconds max for turn detection

// Configurable server URLs with fallbacks
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:3000";
const MODELS_BASE_URL = import.meta.env.VITE_MODELS_BASE_URL || `${API_BASE_URL}/models`;

type AppStatus = "idle" | "loading" | "recording" | "error";

const App = (): JSX.Element => {
  const [status, setStatus] = useState<AppStatus>("idle");
  const [currentError, setCurrentError] = useState<string | null>(null);
  const [currentResponse, setCurrentResponse] = useState<string | null>(null);
  const [visualizerAnalyser, setVisualizerAnalyser] = useState<
    AnalyserNode | null
  >(null);

  const vadRef = useRef<VadIterator | null>(null);
  const openaiTranscriptionRef = useRef<OpenAIRealtimeTranscription | null>(null);
  const smartTurnRef = useRef<SmartTurnV3 | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const analyserNodeRef = useRef<AnalyserNode | null>(null);
  const recordedChunksRef = useRef<Float32Array[]>([]);
  const lastTranscriptionTimeRef = useRef(0);
  const currentSpeechBufferRef = useRef<Float32Array[]>([]);
  const ringBufferRef = useRef<RingBuffer | null>(null);
  const turnDetectionBufferRef = useRef<Float32Array[]>([]);
  const pendingTranscriptionRef = useRef("");
  const isProcessingOpenAIRef = useRef(false);
  const lastRequestTimestampRef = useRef(0);
  const lastUserSpeechTimeRef = useRef(0);
  const lastTranscribedAudioLengthRef = useRef(0);
  const isTranscribingRef = useRef(false);
  const transcriptionQueueRef = useRef<TranscriptionJob[]>([]);
  const speechStartedRef = useRef(false);
  const silenceCounterRef = useRef(0);
  const silenceTimeoutRef = useRef<number | null>(null);
  const toggleInProgressRef = useRef(false);
  const isRecordingRef = useRef(false);

  const isRecording = status === "recording";

  const updateStatus = (newStatus: AppStatus, errorMsg?: string) => {
    const prevStatus = status;
    console.log(`updateStatus: ${prevStatus} → ${newStatus}, error: ${errorMsg}`);

    setStatus(newStatus);
    if (newStatus === "error" && errorMsg) {
      setCurrentError(errorMsg);
    } else {
      setCurrentError(null);
    }

    // Update recording ref to always be in sync
    const newIsRecording = newStatus === "recording";
    isRecordingRef.current = newIsRecording;

    console.log(`Sending recording state: ${newIsRecording} (status: ${prevStatus} → ${newStatus})`);
    window.electronAPI?.sendRecordingState?.(newIsRecording);
  };

  const displayTranscription = (transcription: string, timestamp: number) => {
    console.log(`[${formatTimestamp(timestamp)}] User: ${transcription}`);
    // setTranscriptions((prev) => [...prev, { text: transcription, timestamp }]);
  };

  const displayOpenAIResponse = (responseText: string, timestamp: number) => {
    console.log(`[${formatTimestamp(timestamp)}] Assistant: ${responseText}`);
    setCurrentResponse(responseText);
  };

  const formatTimestamp = (timestamp: number): string => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const combineAudioChunksLocal = (chunks: Float32Array[]): Float32Array => {
    if (chunks.length === 0) {
      return new Float32Array(0);
    }

    const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const combined = new Float32Array(totalLength);
    let offset = 0;

    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    return combined;
  };

  const resample16to24 = (input: Float32Array): Float32Array => {
    const inputSampleRate = 16000;
    const outputSampleRate = 24000;
    const ratio = outputSampleRate / inputSampleRate;
    const outputLength = Math.floor(input.length * ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const inputIndex = i / ratio;
      const inputIndexFloor = Math.floor(inputIndex);
      const inputIndexCeil = Math.min(inputIndexFloor + 1, input.length - 1);
      const fraction = inputIndex - inputIndexFloor;
      output[i] =
        input[inputIndexFloor] * (1 - fraction) + input[inputIndexCeil] * fraction;
    }

    return output;
  };

  const clearVisualizerAnalyser = () => {
    if (analyserNodeRef.current) {
      try {
        analyserNodeRef.current.disconnect();
      } catch (error) {
        console.warn("analyser disconnect failed", error);
      }
      analyserNodeRef.current = null;
    }
    setVisualizerAnalyser(null);
  };

  const sendToOpenAI = async (transcription: string) => {
    if (!transcription.trim() || isProcessingOpenAIRef.current) {
      return;
    }

    isProcessingOpenAIRef.current = true;
    lastRequestTimestampRef.current = Date.now();

    try {
      // Take screenshot before sending to API
      let screenshotBase64: string | undefined;
      try {
        const screenshotResult = await window.electronAPI.takeScreenshot();
        if (screenshotResult.success && screenshotResult.image) {
          screenshotBase64 = screenshotResult.image;
          console.log(`Screenshot captured: ${screenshotResult.width}x${screenshotResult.height} (resized from ${screenshotResult.originalWidth}x${screenshotResult.originalHeight})`);
        } else {
          console.warn('Screenshot failed:', screenshotResult.error);
        }
      } catch (screenshotError) {
        console.error('Error taking screenshot:', screenshotError);
        // Continue without screenshot if it fails
      }

      // Build request body with text and optional image
      const requestBody: { text: string; image?: string } = {
        text: transcription,
      };

      if (screenshotBase64) {
        requestBody.image = screenshotBase64;
      }

      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      const responseLines: string[] = [];

      // Handle the new server response format
      const text = data?.text;
      if (typeof text === "string" && text.trim()) {
        responseLines.push(text.trim());
      }

      // Handle tool calls from the new API format
      const toolCalls = Array.isArray(data?.toolCalls) ? data.toolCalls : [];
      for (const call of toolCalls) {
        console.log(call);
        const toolName = call?.toolName ?? "unknown_tool";
        const args = call?.input ?? {};
        const argsStr = JSON.stringify(args);
        responseLines.push(`Called ${toolName}(${argsStr})`);

        // Execute tool calls with proper error handling
        try {
          let result;
          if (toolName === "open") {
            result = await window.electronAPI.openTool(args);
          } else if (toolName === "scroll") {
            result = await window.electronAPI.scrollTool(args);
          } else if (toolName === "click") {
            result = await window.electronAPI.clickTool(args);
          } else if (toolName === "keys") {
            result = await window.electronAPI.keysTool(args);
          } else {
            console.warn(`Unknown tool: ${toolName}`);
            continue;
          }

          console.log(`Result from ${toolName}:`, result);

          // Add result info to response if tool execution failed
          if (result && !result.success) {
            responseLines.push(`${toolName} failed: ${result.output || 'Unknown error'}`);
          }
        } catch (toolError) {
          console.error(`Error executing tool ${toolName}:`, toolError);
          responseLines.push(`${toolName} error: ${toolError instanceof Error ? toolError.message : 'Unknown error'}`);
        }
      }

      // Fallback if no content
      if (!responseLines.length) {
        if (data) {
          responseLines.push(JSON.stringify(data));
        } else {
          responseLines.push("No response received");
        }
      }

      const responseText = responseLines.join("\n");
      if (lastUserSpeechTimeRef.current <= lastRequestTimestampRef.current) {
        displayOpenAIResponse(responseText, Date.now());
      }
    } catch (error) {
      console.error("Error getting generator response:", error);
      if (lastUserSpeechTimeRef.current <= lastRequestTimestampRef.current) {
        const errorMsg = "Error getting response from generator";
        console.log(`[${formatTimestamp(Date.now())}] Assistant: ${errorMsg}`);
        setCurrentResponse(errorMsg);
      }
    } finally {
      isProcessingOpenAIRef.current = false;
    }
  };

  const transcribeSpeechLocal = async (
    audioData: Float32Array,
    reason: "speech_end" | "timer",
  ) => {
    const openaiTranscription = openaiTranscriptionRef.current;
    const smartTurn = smartTurnRef.current;
    if (!openaiTranscription || !smartTurn) {
      return;
    }

    if (isTranscribingRef.current) {
      transcriptionQueueRef.current.push({ data: audioData, reason });
      // Prevent memory leak by limiting transcription queue
      const MAX_TRANSCRIPTION_QUEUE = 20;
      if (transcriptionQueueRef.current.length > MAX_TRANSCRIPTION_QUEUE) {
        console.warn("Transcription queue overflow, dropping oldest request");
        transcriptionQueueRef.current.shift();
      }
      return;
    }

    isTranscribingRef.current = true;

    const processOne = async (
      data: Float32Array,
      why: "speech_end" | "timer",
    ): Promise<void> => {
      try {
        const resampledData = resample16to24(data);
        const transcription = await openaiTranscription.transcribe(resampledData);
        const trimmed = transcription.trim();
        if (trimmed) {
          displayTranscription(trimmed, Date.now());
          pendingTranscriptionRef.current = pendingTranscriptionRef.current
            ? `${pendingTranscriptionRef.current} ${trimmed}`
            : trimmed;

          turnDetectionBufferRef.current.push(data);

          let totalSamples = turnDetectionBufferRef.current.reduce(
            (sum, chunk) => sum + chunk.length,
            0,
          );
          while (
            totalSamples > MAX_TURN_DETECTION_BUFFER_SAMPLES &&
            turnDetectionBufferRef.current.length > 1
          ) {
            const removed = turnDetectionBufferRef.current.shift();
            if (removed) {
              totalSamples -= removed.length;
            }
          }

          if (why === "speech_end") {
            const combinedAudio = combineAudioChunksLocal(
              turnDetectionBufferRef.current,
            );
            const turnResult = await smartTurn.predictEndpoint(combinedAudio);
            console.log(
              `Turn detection (reason=${why}): prediction=${turnResult.prediction}, probability=${turnResult.probability.toFixed(3)}`,
            );
            if (turnResult.prediction === 1) {
              if (
                pendingTranscriptionRef.current.trim() &&
                !isProcessingOpenAIRef.current
              ) {
                const transcriptionToSend = pendingTranscriptionRef.current.trim();
                pendingTranscriptionRef.current = "";
                await sendToOpenAI(transcriptionToSend);
                turnDetectionBufferRef.current = [];
              }
            }
          } else {
            console.log("Timer-based transcription buffered (no endpoint check)");
          }
        }
      } catch (error) {
        console.error("Error transcribing speech or detecting turn:", error);
      }
    };

    try {
      await processOne(audioData, reason);
      while (transcriptionQueueRef.current.length > 0) {
        const next = transcriptionQueueRef.current.shift();
        if (next) {
          await processOne(next.data, next.reason);
        }
      }
    } finally {
      isTranscribingRef.current = false;
    }
  };

  const processAudioChunkLocal = async (audioChunk: Float32Array) => {
    const vad = vadRef.current;
    const ringBuffer = ringBufferRef.current;
    if (!vad || !ringBuffer) {
      return;
    }

    const preChunkBufferedAudio = ringBuffer.read();

    try {
      const windowSize = 512;
      for (let i = 0; i < audioChunk.length; i += windowSize) {
        const chunk = audioChunk.slice(i, Math.min(i + windowSize, audioChunk.length));
        if (chunk.length === windowSize) {
          await vad.predict(chunk);
        }
      }

      const hasCurrentSpeech = vad.triggered;

      if (hasCurrentSpeech && !speechStartedRef.current) {
        speechStartedRef.current = true;
        silenceCounterRef.current = 0;
        lastUserSpeechTimeRef.current = Date.now();
        lastTranscribedAudioLengthRef.current = 0;

        currentSpeechBufferRef.current = [];
        if (preChunkBufferedAudio.length > 0) {
          currentSpeechBufferRef.current.push(preChunkBufferedAudio);
        }
        currentSpeechBufferRef.current.push(audioChunk);
      } else if (speechStartedRef.current) {
        currentSpeechBufferRef.current.push(audioChunk);

        // Prevent memory leak by limiting current speech buffer
        if (currentSpeechBufferRef.current.length > MAX_AUDIO_BUFFER_CHUNKS) {
          currentSpeechBufferRef.current.shift(); // Remove oldest chunk
        }

        if (!hasCurrentSpeech) {
          silenceCounterRef.current += 1;
          if (silenceCounterRef.current >= SILENCE_THRESHOLD) {
            const currentAudio = combineAudioChunksLocal(
              currentSpeechBufferRef.current,
            );
            if (
              currentAudio.length > 0 &&
              currentAudio.length > lastTranscribedAudioLengthRef.current
            ) {
              const previousLength = lastTranscribedAudioLengthRef.current;
              const targetLength = currentAudio.length;
              const newAudio = currentAudio.slice(previousLength);
              if (newAudio.length >= MIN_TRANSCRIBE_SAMPLES) {
                lastTranscribedAudioLengthRef.current = targetLength;
                try {
                  await transcribeSpeechLocal(newAudio, "speech_end");
                } catch (error) {
                  lastTranscribedAudioLengthRef.current = previousLength;
                  throw error;
                }
              }
            }
            currentSpeechBufferRef.current = [];
            lastTranscriptionTimeRef.current = Date.now();
            speechStartedRef.current = false;
            silenceCounterRef.current = 0;
          }
        } else {
          silenceCounterRef.current = 0;
        }
      }

      const now = Date.now();
      if (
        now - lastTranscriptionTimeRef.current >= 5000 &&
        currentSpeechBufferRef.current.length > 0
      ) {
        const currentAudio = combineAudioChunksLocal(currentSpeechBufferRef.current);
        if (currentAudio.length > lastTranscribedAudioLengthRef.current) {
          const previousLength = lastTranscribedAudioLengthRef.current;
          const targetLength = currentAudio.length;
          const newAudio = currentAudio.slice(previousLength);
          if (newAudio.length >= MIN_TRANSCRIBE_SAMPLES) {
            lastTranscribedAudioLengthRef.current = targetLength;
            try {
              await transcribeSpeechLocal(newAudio, "timer");
            } catch (error) {
              lastTranscribedAudioLengthRef.current = previousLength;
              throw error;
            }
          }
          lastTranscriptionTimeRef.current = now;
        }
      }

      ringBuffer.write(audioChunk);
    } catch (error) {
      console.error("Error processing audio chunk:", error);
    }
  };

  const initializeModelsLocal = async () => {
    try {
      updateStatus("loading");
      const vad = new VadIterator(`${MODELS_BASE_URL}/silero_vad.onnx`);
      await vad.init();
      vadRef.current = vad;

      updateStatus("loading");
      const smartTurn = new SmartTurnV3();
      await smartTurn.init();
      smartTurnRef.current = smartTurn;

      updateStatus("loading");

      const apiKey = import.meta.env.VITE_OPENAI_API_KEY;
      if (!apiKey) {
        throw new Error("VITE_OPENAI_API_KEY not found in environment variables");
      }

      const openaiTranscription = new OpenAIRealtimeTranscription(apiKey, 24000);
      await openaiTranscription.init();
      openaiTranscriptionRef.current = openaiTranscription;

      updateStatus("idle");
      return true;
    } catch (error) {
      console.error("Failed to initialize models:", error);
      updateStatus("error", "Failed to load models or connect to OpenAI");
      return false;
    }
  };

  const createAudioProcessorBlobLocal = () => {
    const processorCode = `
      class VADProcessor extends AudioWorkletProcessor {
        constructor() {
          super();
          this.bufferSize = 512;
          this.buffer = new Float32Array(this.bufferSize);
          this.bufferIndex = 0;
        }

        process(inputs) {
          const input = inputs[0];
          if (input.length > 0) {
            const inputChannel = input[0];

            for (let i = 0; i < inputChannel.length; i++) {
              this.buffer[this.bufferIndex] = inputChannel[i];
              this.bufferIndex++;

              if (this.bufferIndex >= this.bufferSize) {
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

    return URL.createObjectURL(
      new Blob([processorCode], { type: "application/javascript" }),
    );
  };

  const startRecording = async () => {
    try {
      updateStatus("loading");

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      mediaStreamRef.current = stream;
      const AudioContextCtor =
        window.AudioContext ||
        (window as typeof window & { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (!AudioContextCtor) {
        throw new Error("Web Audio API not supported in this browser");
      }

      const audioContext = new AudioContextCtor({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);

      if (!vadRef.current || !openaiTranscriptionRef.current) {
        const initialized = await initializeModelsLocal();
        if (!initialized) {
          stream.getTracks().forEach((track) => track.stop());
          mediaStreamRef.current = null;
          clearVisualizerAnalyser();
          return;
        }
      }

      const workletUrl = createAudioProcessorBlobLocal();
      try {
        await audioContext.audioWorklet.addModule(workletUrl);
      } finally {
        URL.revokeObjectURL(workletUrl);
      }

      const processorNode = new AudioWorkletNode(audioContext, "vad-processor");
      processorNode.port.onmessage = async (
        event: MessageEvent<{ audioData: number[] }>,
      ) => {
        const { audioData } = event.data;
        const audioChunk = new Float32Array(audioData);

        // Prevent memory leak by limiting recorded chunks
        recordedChunksRef.current.push(audioChunk);
        if (recordedChunksRef.current.length > MAX_AUDIO_BUFFER_CHUNKS) {
          recordedChunksRef.current.shift(); // Remove oldest chunk
        }

        await processAudioChunkLocal(audioChunk);
      };

      const analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 256;
      analyserNode.smoothingTimeConstant = 0.8;
      source.connect(analyserNode);
      analyserNodeRef.current = analyserNode;
      setVisualizerAnalyser(analyserNode);

      source.connect(processorNode);
      processorNode.connect(audioContext.destination);

      recordedChunksRef.current = [];
      currentSpeechBufferRef.current = [];
      lastTranscriptionTimeRef.current = Date.now();
      speechStartedRef.current = false;
      silenceCounterRef.current = 0;
      turnDetectionBufferRef.current = [];
      pendingTranscriptionRef.current = "";
      isProcessingOpenAIRef.current = false;
      lastRequestTimestampRef.current = 0;
      lastUserSpeechTimeRef.current = 0;
      lastTranscribedAudioLengthRef.current = 0;

      ringBufferRef.current = new RingBuffer(16000);

      updateStatus("recording");
    } catch (error) {
      console.error("Error starting recording:", error);
      clearVisualizerAnalyser();
      updateStatus("error", "Failed to access microphone");
    }
  };

  const toggleRecording = async () => {
    // Use a more robust state check that considers both ref and status
    const currentIsRecording = isRecordingRef.current || status === "recording";
    console.log(`toggleRecording called - status: ${status}, isRecording (state): ${isRecording}, isRecording (ref): ${currentIsRecording}, toggleInProgress: ${toggleInProgressRef.current}`);

    // Prevent multiple simultaneous toggles with atomic operation
    if (toggleInProgressRef.current) {
      console.log('Toggle already in progress, ignoring...');
      return;
    }

    // Set toggle in progress atomically
    toggleInProgressRef.current = true;

    try {
      // Double-check state hasn't changed since we started
      const finalIsRecording = isRecordingRef.current || status === "recording";

      if (finalIsRecording) {
        console.log('Stopping recording...');
        await stopRecording();
      } else {
        console.log('Starting recording...');
        await startRecording();
      }
    } catch (error) {
      console.error('Error during recording toggle:', error);
      // Ensure we're in a consistent state after error
      if (isRecordingRef.current && (status === "error" || status === "idle")) {
        updateStatus("idle");
      }
    } finally {
      // Always clear the toggle flag
      toggleInProgressRef.current = false;
    }
  };

  const stopRecording = async () => {
    if (!isRecordingRef.current) {
      console.log('stopRecording called but not currently recording, ignoring');
      return;
    }

    updateStatus("idle");
    clearVisualizerAnalyser();

    if (silenceTimeoutRef.current !== null) {
      clearTimeout(silenceTimeoutRef.current);
      silenceTimeoutRef.current = null;
    }

    if (pendingTranscriptionRef.current.trim() && !isProcessingOpenAIRef.current) {
      const transcriptionToSend = pendingTranscriptionRef.current.trim();
      pendingTranscriptionRef.current = "";
      await sendToOpenAI(transcriptionToSend);
    }

    const audioContext = audioContextRef.current;
    audioContextRef.current = null;

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (audioContext) {
      await audioContext.close();
    }

    if (ringBufferRef.current) {
      ringBufferRef.current.clear();
      ringBufferRef.current = null;
    }

    // If no audio was recorded, just stay in idle state (not an error)
    if (recordedChunksRef.current.length === 0) {
      return;
    }

    // Process recorded audio silently in the background without changing UI state
    try {
      const totalLength = recordedChunksRef.current.reduce(
        (sum, chunk) => sum + chunk.length,
        0,
      );
      const combinedAudio = new Float32Array(totalLength);
      let offset = 0;

      for (const chunk of recordedChunksRef.current) {
        combinedAudio.set(chunk, offset);
        offset += chunk.length;
      }

      const vad = vadRef.current;
      if (vad) {
        await vad.process(combinedAudio);
        vad.getSpeechTimestamps();
      }
    } catch (error) {
      console.error("Error processing recorded audio:", error);
      // Don't change UI state for post-processing errors
    }
  };


  useEffect(() => {
    // Listen for tray toggle recording events
    const removeToggleListener = window.electronAPI?.onToggleRecording?.(toggleRecording);


    return () => {
      if (silenceTimeoutRef.current !== null) {
        clearTimeout(silenceTimeoutRef.current);
        silenceTimeoutRef.current = null;
      }

      clearVisualizerAnalyser();

      const context = audioContextRef.current;
      if (context) {
        context.close().catch(() => undefined);
        audioContextRef.current = null;
      }

      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }

      // Clean up tray toggle listener
      if (removeToggleListener) {
        removeToggleListener();
      }
    };
  }, []);


  return (
    <main className="relative flex min-h-screen w-full flex-col overflow-hidden font-sans leading-relaxed text-gray-800">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <Cursor analyser={visualizerAnalyser} status={status} currentResponse={currentResponse} />
      </div>


      {/* Error Display - positioned at upper center */}
      {currentError && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10 md:top-8 max-w-md p-4 rounded-md border border-red-300 bg-red-50 flex items-center">
          <svg className="h-5 w-5 text-red-400 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
          </svg>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <div className="mt-1 text-sm text-red-700">{currentError}</div>
          </div>
        </div>
      )}
    </main>
  );
};

export default App;
