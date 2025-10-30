import { useEffect, useRef, useState } from "react";
import { RingBuffer } from "frontend-core";
import "./app.css";
import Cursor from "./components/Cursor";
import ErrorMessage from "./components/ErrorMessage";
import { VAD } from "./vad";
import { transcribe, disconnectTranscription } from "./transcribe";
import vadModelUrl from "../../assets/silero_vad.onnx?url";
import { sendToLLM } from "./llm";

type AppStatus = "idle" | "loading" | "recording" | "error";

const App = (): JSX.Element => {
  const [status, setStatus] = useState<AppStatus>("idle");
  const [currentError, setCurrentError] = useState<string | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const vadRef = useRef<VAD | null>(null);
  const isSpeakingRef = useRef<boolean>(false);

  const audioBufferRef = useRef<RingBuffer | null>(null);
  const speechBufferRef = useRef<RingBuffer | null>(null);

  const processAudioChunk = (chunk: Float32Array) => {
    if (audioBufferRef.current) {
      audioBufferRef.current.write(chunk);
    }

    if (isSpeakingRef.current && speechBufferRef.current) {
      speechBufferRef.current.write(chunk);
    }

    if (vadRef.current) {
      vadRef.current.process(chunk);
    }
  };

  const speechStart = () => {
    console.log("Speech started");
    isSpeakingRef.current = true;

    // Copy the previous 2 seconds from audioBuffer to speechBuffer
    // This ensures we capture the beginning of speech that occurred before VAD detection
    if (audioBufferRef.current && speechBufferRef.current) {
      const previousAudio = audioBufferRef.current.read();
      speechBufferRef.current.write(previousAudio);
      console.log(`Prepended ${previousAudio.length} samples (${(previousAudio.length / 16000).toFixed(2)}s) from ring buffer`);
    }
  };

  const speechEnd = async () => {
    console.log("Speech ended");
    isSpeakingRef.current = false;

    // Transcribe the speech buffer
    if (speechBufferRef.current) {
      try {
        const audioData = speechBufferRef.current.read();
        console.log(`Transcribing ${audioData.length} samples (${(audioData.length / 16000).toFixed(2)}s)`);

        const transcript = await transcribe(audioData);
        console.log("Transcript:", transcript);

        // Send transcript to LLM
        if (transcript.trim()) {
          await sendToLLM(transcript);
        }

        // Clear the speech buffer after processing
        speechBufferRef.current.clear();
      } catch (error) {
        console.error("Transcription failed:", error);
      }
    }
  };

  // Initialize and start recording
  const startRecording = async () => {
    try {
      setStatus("loading");
      setCurrentError(null);

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });

      mediaStreamRef.current = stream;

      // Create audio context
      const audioContext = new AudioContext({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      // Create source node from microphone stream
      const source = audioContext.createMediaStreamSource(stream);
      sourceNodeRef.current = source;

      const AUDIO_BUFFER_SIZE = 16000 * 2; // 2 seconds
      const SPEECH_BUFFER_SIZE = 16000 * 8; // 8 seconds
      audioBufferRef.current = new RingBuffer(AUDIO_BUFFER_SIZE);
      speechBufferRef.current = new RingBuffer(SPEECH_BUFFER_SIZE);

      // Initialize VAD
      vadRef.current = new VAD({
        modelPath: vadModelUrl,
        samplingRate: 16000,
        threshold: 0.5,
        onSpeechStart: speechStart,
        onSpeechEnd: speechEnd,
        onError: (error) => {
          console.error("VAD error:", error);
          setCurrentError(`VAD error: ${error.message}`);
          setStatus("error");
        }
      });

      // Wait for VAD model to load
      await vadRef.current.init();

      // Create AudioWorklet processor code
      const processorCode = `
        class ChunkProcessor extends AudioWorkletProcessor {
          constructor() {
            super();
            this.chunkSize = 512; // VAD chunk size
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

                // When we have 512 samples, send them
                if (this.bufferIndex >= this.chunkSize) {
                  // Send a copy of the buffer
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
      const blob = new Blob([processorCode], { type: 'application/javascript' });
      const processorUrl = URL.createObjectURL(blob);
      await audioContext.audioWorklet.addModule(processorUrl);
      URL.revokeObjectURL(processorUrl);

      // Create AudioWorklet node
      const workletNode = new AudioWorkletNode(audioContext, 'chunk-processor');
      workletNodeRef.current = workletNode;

      // Listen for audio chunks from the worklet
      workletNode.port.onmessage = (event) => {
        if (event.data.type === 'audioChunk') {
          processAudioChunk(event.data.data);
        }
      };

      // Connect: microphone → worklet
      source.connect(workletNode);

      setStatus("recording");
      console.log("Recording started");
    } catch (error) {
      console.error("Error starting recording:", error);
      setCurrentError(error instanceof Error ? error.message : "Failed to start recording");
      setStatus("error");
    }
  };

  // Stop recording and cleanup
  const stopRecording = () => {
    // Disconnect transcription session
    disconnectTranscription();

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
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    setStatus("idle");
    console.log("Recording stopped");
  };

  // Start recording on mount
  useEffect(() => {
    startRecording();

    return () => {
      stopRecording();
    };
  }, []);

  return (
    <main>
      <Cursor status={status}/>
      <ErrorMessage error={currentError}/>
    </main>
  )
};

export default App;
