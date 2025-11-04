import { useEffect, useRef, useState } from "react";
import { RingBuffer } from "frontend-core";
import { VAD } from "./vad";
import vadModelUrl from "../../assets/silero_vad.onnx?url";

export type RecorderStatus = "idle" | "loading" | "recording" | "error";

export interface RecorderCallbacks {
  onSpeechEnd: (audioData: Float32Array) => void | Promise<void>;
  onSpeechStart?: () => void;
  onError?: (error: Error) => void;
}

export interface RecorderConfig {
  /** Sample rate for audio processing (default: 16000) */
  sampleRate?: 8000 | 16000;
  /** Size of the audio buffer in samples (default: 2 seconds of audio) */
  audioBufferSize?: number;
  /** Size of the speech buffer in samples (default: 8 seconds of audio) */
  speechBufferSize?: number;
  /** VAD detection threshold (default: 0.5) */
  vadThreshold?: number;
  /** Audio chunk size for processing (default: 512) */
  chunkSize?: number;
  /** Microphone constraints */
  audioConstraints?: MediaTrackConstraints;
}

export interface RecorderResult {
  status: RecorderStatus;
  error: string | null;
  isSpeaking: boolean;
}

const DEFAULT_CONFIG = {
  sampleRate: 16000 as const,
  audioBufferSize: 16000 * 2, // 2 seconds
  speechBufferSize: 16000 * 8, // 8 seconds
  vadThreshold: 0.5,
  chunkSize: 512,
  audioConstraints: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  },
};

/**
 * Custom hook for managing microphone recording with Voice Activity Detection (VAD).
 *
 * This hook handles:
 * - Microphone access and audio context setup
 * - Real-time audio processing through AudioWorklet
 * - Voice Activity Detection (VAD)
 * - Audio buffering for context preservation
 * - Automatic cleanup on unmount
 *
 * @param callbacks - Callbacks for speech events
 * @param config - Optional configuration for recording parameters
 * @returns Recording status, error state, and speaking state
 *
 * @example
 * ```tsx
 * const { status, error, isSpeaking } = useRecorder({
 *   onSpeechEnd: async (audioData) => {
 *     const transcript = await transcribe(audioData);
 *     await runAgent(transcript);
 *   },
 *   onSpeechStart: () => console.log("User started speaking"),
 *   onError: (error) => console.error("Recording error:", error),
 * });
 * ```
 */
export function useRecorder(
  callbacks: RecorderCallbacks,
  config: RecorderConfig = {}
): RecorderResult {
  const [status, setStatus] = useState<RecorderStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);

  // Merge config with defaults
  const fullConfig = { ...DEFAULT_CONFIG, ...config };
  const {
    sampleRate,
    audioBufferSize,
    speechBufferSize,
    vadThreshold,
    chunkSize,
    audioConstraints,
  } = fullConfig;

  // Audio processing refs
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const vadRef = useRef<VAD | null>(null);
  const isSpeakingRef = useRef<boolean>(false);

  // Audio buffers
  const audioBufferRef = useRef<RingBuffer | null>(null);
  const speechBufferRef = useRef<RingBuffer | null>(null);

  // Callback refs to avoid stale closures
  const callbacksRef = useRef(callbacks);
  useEffect(() => {
    callbacksRef.current = callbacks;
  }, [callbacks]);

  /**
   * Process incoming audio chunks
   */
  const processAudioChunk = (chunk: Float32Array) => {
    // Write to continuous audio buffer (for context preservation)
    if (audioBufferRef.current) {
      audioBufferRef.current.write(chunk);
    }

    // If speaking, also write to speech buffer
    if (isSpeakingRef.current && speechBufferRef.current) {
      speechBufferRef.current.write(chunk);
    }

    // Run VAD on chunk
    if (vadRef.current) {
      vadRef.current.process(chunk);
    }
  };

  /**
   * Handle speech start event from VAD
   */
  const handleSpeechStart = () => {
    console.log("Speech started");
    isSpeakingRef.current = true;
    setIsSpeaking(true);

    // Copy the previous audio from audioBuffer to speechBuffer
    // This ensures we capture the beginning of speech that occurred before VAD detection
    if (audioBufferRef.current && speechBufferRef.current) {
      const previousAudio = audioBufferRef.current.read();
      speechBufferRef.current.write(previousAudio);
      console.log(
        `Prepended ${previousAudio.length} samples (${(previousAudio.length / sampleRate).toFixed(2)}s) from ring buffer`
      );
    }

    // Call user callback if provided
    callbacksRef.current.onSpeechStart?.();
  };

  /**
   * Handle speech end event from VAD
   */
  const handleSpeechEnd = async () => {
    console.log("Speech ended");
    isSpeakingRef.current = false;
    setIsSpeaking(false);

    if (speechBufferRef.current) {
      try {
        const audioData = speechBufferRef.current.read();
        console.log(
          `Processing speech: ${audioData.length} samples (${(audioData.length / sampleRate).toFixed(2)}s)`
        );

        // Call user callback with audio data
        await callbacksRef.current.onSpeechEnd(audioData);

        // Clear speech buffer for next utterance
        speechBufferRef.current.clear();
      } catch (err) {
        console.error("Error processing speech:", err);
        const error = err instanceof Error ? err : new Error(String(err));
        callbacksRef.current.onError?.(error);
      }
    }
  };

  /**
   * Handle errors from VAD or recording
   */
  const handleError = (err: Error) => {
    console.error("Recorder error:", err);
    setError(err.message);
    setStatus("error");
    callbacksRef.current.onError?.(err);
  };

  /**
   * Initialize and start recording
   */
  const startRecording = async () => {
    try {
      setStatus("loading");
      setError(null);

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: audioConstraints,
      });
      mediaStreamRef.current = stream;

      // Create audio context
      const audioContext = new AudioContext({ sampleRate });
      audioContextRef.current = audioContext;

      // Create source node from microphone stream
      const source = audioContext.createMediaStreamSource(stream);
      sourceNodeRef.current = source;

      // Initialize ring buffers
      audioBufferRef.current = new RingBuffer(audioBufferSize);
      speechBufferRef.current = new RingBuffer(speechBufferSize);

      // Initialize VAD
      vadRef.current = new VAD({
        modelPath: vadModelUrl,
        samplingRate: sampleRate,
        threshold: vadThreshold,
        onSpeechStart: handleSpeechStart,
        onSpeechEnd: handleSpeechEnd,
        onError: handleError,
      });

      // Wait for VAD model to load
      await vadRef.current.init();

      // Create AudioWorklet processor
      const processorCode = `
        class ChunkProcessor extends AudioWorkletProcessor {
          constructor() {
            super();
            this.chunkSize = ${chunkSize};
            this.buffer = new Float32Array(this.chunkSize);
            this.bufferIndex = 0;
          }

          process(inputs, outputs, parameters) {
            const input = inputs[0];
            if (input.length > 0) {
              const inputChannel = input[0];

              for (let i = 0; i < inputChannel.length; i++) {
                this.buffer[this.bufferIndex] = inputChannel[i];
                this.bufferIndex++;

                // When we have a full chunk, send it
                if (this.bufferIndex >= this.chunkSize) {
                  this.port.postMessage({
                    type: 'audioChunk',
                    data: this.buffer.slice()
                  });
                  this.bufferIndex = 0;
                }
              }
            }
            return true; // Keep processor alive
          }
        }

        registerProcessor('chunk-processor', ChunkProcessor);
      `;

      // Register AudioWorklet processor
      const blob = new Blob([processorCode], { type: "application/javascript" });
      const processorUrl = URL.createObjectURL(blob);
      await audioContext.audioWorklet.addModule(processorUrl);
      URL.revokeObjectURL(processorUrl);

      // Create AudioWorklet node
      const workletNode = new AudioWorkletNode(audioContext, "chunk-processor");
      workletNodeRef.current = workletNode;

      // Listen for audio chunks from the worklet
      workletNode.port.onmessage = (event) => {
        if (event.data.type === "audioChunk") {
          processAudioChunk(event.data.data);
        }
      };

      // Connect audio graph: microphone → worklet
      source.connect(workletNode);

      setStatus("recording");
      console.log("Recording started");
    } catch (err) {
      console.error("Error starting recording:", err);
      const error = err instanceof Error ? err : new Error(String(err));
      handleError(error);
    }
  };

  /**
   * Stop recording and cleanup resources
   */
  const stopRecording = () => {
    console.log("Stopping recording...");

    if (vadRef.current) {
      vadRef.current.close();
      vadRef.current = null;
    }

    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current.port.onmessage = null;
      workletNodeRef.current = null;
    }

    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    setStatus("idle");
    setIsSpeaking(false);
    console.log("Recording stopped");
  };

  // Start recording on mount, cleanup on unmount
  useEffect(() => {
    startRecording();

    return () => {
      stopRecording();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty deps - only run once on mount

  return {
    status,
    error,
    isSpeaking,
  };
}

