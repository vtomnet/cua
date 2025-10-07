import { useEffect, useMemo, useRef, useState } from "react";
import {
  VadIterator,
  SmartTurnV3,
  OpenAIRealtimeTranscription,
  RingBuffer,
  Timestamp,
} from "frontend-core";
import "./app.css";
import OrbVisualizer from "./components/OrbVisualizer";

type Message = {
  text: string;
  timestamp: number;
  role: "user" | "assistant";
};

type TranscriptionJob = {
  data: Float32Array;
  reason: "speech_end" | "timer";
};

const MIN_TRANSCRIBE_SAMPLES = Math.ceil(16000 * 0.1);
const SILENCE_THRESHOLD = 10;

const App = (): JSX.Element => {
  const [isRecording, setIsRecording] = useState(false);
  const [status, setStatus] = useState(
    "Status: Ready - click Start Recording to begin",
  );
  const [statusClass, setStatusClass] = useState(
    "bg-gray-100 text-gray-800 border-gray-300",
  );
  const [speechSegments, setSpeechSegments] = useState<Timestamp[]>([]);
  const [transcriptions, setTranscriptions] = useState<
    Array<{ text: string; timestamp: number }>
  >([]);
  const [openaiResponses, setOpenaiResponses] = useState<
    Array<{ text: string; timestamp: number }>
  >([]);
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

  const updateStatus = (msg: string, className = "") => {
    setStatus(`Status: ${msg}`);
    switch (className) {
      case "loading":
        setStatusClass("bg-yellow-100 text-yellow-800 border-yellow-300");
        break;
      case "recording":
        setStatusClass("bg-red-100 text-red-800 border-red-300");
        break;
      case "success":
        setStatusClass("bg-green-100 text-green-800 border-green-300");
        break;
      case "error":
        setStatusClass("bg-red-100 text-red-800 border-red-300");
        break;
      default:
        setStatusClass("bg-gray-100 text-gray-800 border-gray-300");
        break;
    }
  };

  const displayTranscription = (transcription: string, timestamp: number) => {
    setTranscriptions((prev) => [...prev, { text: transcription, timestamp }]);
  };

  const displayOpenAIResponse = (responseText: string, timestamp: number) => {
    setOpenaiResponses((prev) => [...prev, { text: responseText, timestamp }]);
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
        setOpenaiResponses((prev) => [
          ...prev,
          { text: "Error getting response from generator", timestamp: Date.now() },
        ]);
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
      updateStatus("Initializing VAD model...", "loading");
      const vad = new VadIterator("http://localhost:3000/models/silero_vad.onnx");
      await vad.init();
      vadRef.current = vad;

      updateStatus("VAD loaded, initializing Smart Turn v3...", "loading");
      const smartTurn = new SmartTurnV3();
      await smartTurn.init();
      smartTurnRef.current = smartTurn;

      updateStatus("Smart Turn v3 loaded, loading ASR...", "loading");

      const apiKey = import.meta.env.VITE_OPENAI_API_KEY;
      if (!apiKey) {
        throw new Error("VITE_OPENAI_API_KEY not found in environment variables");
      }

      const openaiTranscription = new OpenAIRealtimeTranscription(apiKey, 24000);
      await openaiTranscription.init();
      openaiTranscriptionRef.current = openaiTranscription;

      updateStatus(
        "OpenAI Transcription loaded, ready to generate responses locally",
        "success",
      );
      return true;
    } catch (error) {
      console.error("Failed to initialize models:", error);
      updateStatus("Failed to load models or connect to OpenAI", "error");
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
      updateStatus("Requesting microphone access...", "loading");

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

      setIsRecording(true);
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

      updateStatus("Recording... Speak into your microphone", "recording");
    } catch (error) {
      console.error("Error starting recording:", error);
      clearVisualizerAnalyser();
      setIsRecording(false);
      updateStatus("Failed to access microphone", "error");
    }
  };

  const stopRecording = async () => {
    if (!isRecording || !audioContextRef.current) {
      return;
    }

    setIsRecording(false);
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
      updateStatus("No audio recorded", "error");
      return;
    }

    try {
      updateStatus("Processing recorded audio...", "loading");

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
        updateStatus("VAD unavailable", "error");
        return;
      }

      await vad.process(combinedAudio);
      const timestamps = vad.getSpeechTimestamps();
      setSpeechSegments(timestamps);

      updateStatus(
        `Processing complete - ${timestamps.length} speech segments found`,
        "success",
      );
    } catch (error) {
      console.error("Error processing recorded audio:", error);
      updateStatus("Error processing recorded audio", "error");
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

  const messages = useMemo<Message[]>(() => {
    return [
      ...transcriptions.map((entry) => ({ ...entry, role: "user" as const })),
      ...openaiResponses.map((entry) => ({ ...entry, role: "assistant" as const })),
    ].sort((a, b) => a.timestamp - b.timestamp);
  }, [openaiResponses, transcriptions]);

  return (
    <main className="p-4 md:p-8 font-sans leading-relaxed text-gray-800 bg-white">
      <h1 className="text-4xl font-semibold mb-4 text-gray-900 leading-tight">
        Voice Activity Detection
      </h1>

      <div
        className={`status p-4 my-4 rounded-lg font-semibold border transition-all duration-200 ease-in-out ${statusClass}`}
      >
        {status}
      </div>

      <div className="relative mb-8 h-64 w-full overflow-hidden rounded-2xl border border-cyan-400/30 bg-gradient-to-br from-slate-900 via-slate-950 to-black shadow-inner md:h-80">
        <OrbVisualizer analyser={visualizerAnalyser} isRecording={isRecording} />
      </div>

      <div className="bg-slate-50 p-6 my-8 rounded-lg border border-gray-200">
        <h2 className="text-3xl font-medium mb-4 text-gray-700">Microphone Recording</h2>
        <button
          onClick={startRecording}
          disabled={isRecording}
          className="bg-blue-500 text-white px-6 py-3 rounded-md cursor-pointer mr-3 text-sm font-medium transition-all duration-200 ease-in-out shadow-sm hover:bg-blue-600 hover:shadow-md disabled:bg-gray-400 disabled:cursor-not-allowed disabled:opacity-60 md:w-auto w-full mb-2 md:mb-0"
        >
          Start Recording
        </button>
        <button
          onClick={stopRecording}
          disabled={!isRecording}
          className="bg-blue-500 text-white px-6 py-3 rounded-md cursor-pointer mr-3 text-sm font-medium transition-all duration-200 ease-in-out shadow-sm hover:bg-blue-600 hover:shadow-md disabled:bg-gray-400 disabled:cursor-not-allowed disabled:opacity-60 md:w-auto w-full"
        >
          Stop Recording
        </button>
      </div>

      <div className="mt-8 p-6 bg-white rounded-lg border border-gray-200 shadow-sm">
        <h3 className="text-2xl font-medium mb-4 text-gray-600">Speech Segments:</h3>
        <div className="mt-4">
          {speechSegments.length === 0 ? (
            <p className="text-gray-500">No speech segments detected.</p>
          ) : (
            speechSegments.map((segment, index) => (
              <div
                key={`${segment.start}-${segment.end}-${index}`}
                className="px-4 py-3 my-2 bg-slate-100 border-l-4 border-blue-500 rounded-r text-sm font-mono"
              >
                Segment {index + 1}: {Math.round((segment.start / 16000) * 100) / 100}s -
                {" "}
                {Math.round((segment.end / 16000) * 100) / 100}s
              </div>
            ))
          )}
        </div>
      </div>

      <div className="mt-8 p-6 bg-white rounded-lg border border-gray-200 shadow-sm">
        <h3 className="text-2xl font-medium mb-4 text-gray-600">Transcriptions:</h3>
        <div className="max-h-96 overflow-y-auto border border-gray-300 rounded-md p-4 bg-gray-50 mt-4">
          {messages.map((message) => (
            message.role === "user" ? (
              <div
                key={`user-${message.timestamp}-${message.text}`}
                className="py-2 border-b border-gray-200 leading-relaxed"
              >
                <strong>[{formatTimestamp(message.timestamp)}]</strong> {message.text}
              </div>
            ) : (
              <div
                key={`assistant-${message.timestamp}-${message.text}`}
                className="bg-blue-50 p-3 mt-1 rounded-md border-l-4 border-blue-500"
                style={{ whiteSpace: "pre-line" }}
              >
                <strong>[{formatTimestamp(message.timestamp)}] GPT:</strong> {message.text}
              </div>
            )
          ))}
        </div>
      </div>

      <div className="mt-8 p-4 text-sm text-gray-500 bg-gray-50 rounded-md border border-gray-200">
        Click "Start Recording" to begin recording from your microphone. The system will
        detect speech segments using VAD, transcribe them with OpenAI's API on speech end or
        every 5 seconds, and forward transcripts to the local generator service for replies.
      </div>
    </main>
  );
};

export default App;
