import { useEffect, useRef, useState } from "react";
import {
  VadIterator,
  SmartTurnV3,
  OpenAIRealtimeTranscription,
  RingBuffer,
} from "frontend-core";
import "./app.css";
import Cursor from "./components/Cursor";

type TranscriptionJob = {
  data: Float32Array;
  reason: "speech_end" | "timer";
};

const MIN_TRANSCRIBE_SAMPLES = Math.ceil(16000 * 0.1);
const SILENCE_THRESHOLD = 10;

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

  const isRecording = status === "recording";

  const updateStatus = (newStatus: AppStatus, errorMsg?: string) => {
    setStatus(newStatus);
    if (newStatus === "error" && errorMsg) {
      setCurrentError(errorMsg);
    } else {
      setCurrentError(null);
    }
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
      const response = await fetch("http://localhost:3000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ input: transcription }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      const choice = data?.choices?.[0];
      const message = choice?.message ?? null;
      const responseLines: string[] = [];

      const reasoning =
        typeof message?.reasoning === "string" ? message.reasoning.trim() : "";
      if (reasoning) {
        responseLines.push(`Reasoning: ${reasoning}`);
      }

      const content = message?.content;
      if (Array.isArray(content)) {
        const contentText = content
          .map((entry: unknown) => {
            if (typeof entry === "string") return entry;
            if (entry && typeof entry === "object") {
              const maybeText = (entry as { text?: string }).text;
              if (typeof maybeText === "string") return maybeText;
            }
            return "";
          })
          .join(" ")
          .trim();
        if (contentText) {
          responseLines.push(contentText);
        }
      } else if (typeof content === "string" && content.trim()) {
        responseLines.push(content.trim());
      }

      const toolCalls = Array.isArray(message?.tool_calls)
        ? message.tool_calls
        : [];
      for (const call of toolCalls) {
        const toolName = call?.function?.name ?? call?.name ?? "unknown_tool";
        let args = call?.function?.arguments ?? call?.arguments ?? "";
        if (typeof args === "string" && args.trim()) {
          try {
            args = JSON.parse(args);
          } catch (error) {
            console.error("Failed to parse tool arguments", error);
          }
        }
        const argsStr = typeof args === "string" ? args : JSON.stringify(args);
        responseLines.push(`Called ${toolName}(${argsStr})`);
        if (toolName === "open") {
          const result = await window.electronAPI.openTool(args);
          console.log("Result from openTool:", result);
        }
      }

      if (!responseLines.length) {
        const fallback = choice?.text ?? data?.choices?.[0]?.text;
        if (typeof fallback === "string" && fallback.trim()) {
          responseLines.push(fallback.trim());
        } else {
          responseLines.push(JSON.stringify(data));
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

          const maxSamples = 8 * 16000;
          let totalSamples = turnDetectionBufferRef.current.reduce(
            (sum, chunk) => sum + chunk.length,
            0,
          );
          while (
            totalSamples > maxSamples &&
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
      const vad = new VadIterator("http://localhost:3000/models/silero_vad.onnx");
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
        recordedChunksRef.current.push(audioChunk);
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
    if (isRecording) {
      await stopRecording();
    } else {
      await startRecording();
    }
  };

  const stopRecording = async () => {
    if (!isRecording || !audioContextRef.current) {
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

    await audioContext.close();

    if (ringBufferRef.current) {
      ringBufferRef.current.clear();
      ringBufferRef.current = null;
    }

    if (recordedChunksRef.current.length === 0) {
      updateStatus("error", "No audio recorded");
      return;
    }

    try {
      updateStatus("loading");

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
      if (!vad) {
        updateStatus("error", "VAD unavailable");
        return;
      }

      await vad.process(combinedAudio);
      vad.getSpeechTimestamps();

      updateStatus("idle");
    } catch (error) {
      console.error("Error processing recorded audio:", error);
      updateStatus("error", "Error processing recorded audio");
    }
  };

  useEffect(() => {
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
    };
  }, []);


  return (
    <main className="relative flex min-h-screen w-full flex-col overflow-hidden font-sans leading-relaxed text-gray-800">
      <div className="pointer-events-none absolute inset-0 -z-10">
        <Cursor analyser={visualizerAnalyser} status={status} currentResponse={currentResponse} />
      </div>

      {/* Recording toggle button - positioned at lower right */}
      <div className="absolute bottom-4 right-4 z-10 md:bottom-8 md:right-8">
        <button
          onClick={toggleRecording}
          disabled={status === "loading"}
          className={`w-16 h-16 rounded-full flex items-center justify-center text-white shadow-lg transition-all duration-200 ease-in-out hover:shadow-xl disabled:opacity-60 disabled:cursor-not-allowed ${
            isRecording
              ? 'bg-red-500 hover:bg-red-600'
              : 'bg-blue-500 hover:bg-blue-600'
          }`}
          aria-label={isRecording ? 'Stop Recording' : 'Start Recording'}
        >
            <svg
              className="w-6 h-6"
              fill="currentColor"
              viewBox="0 0 20 20"
              xmlns="http://www.w3.org/2000/svg"
            >
              {isRecording ? (
                // Stop icon (square)
                <rect x="6" y="6" width="8" height="8" />
              ) : (
                // Microphone icon
                <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z" clipRule="evenodd" />
              )}
            </svg>
        </button>
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
