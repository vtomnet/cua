// Transcription module using OpenAI's Realtime Transcription API
// Automatically manages WebSocket sessions with 14-minute timeout

const SESSION_TIMEOUT_MS = 14 * 60 * 1000; // 14 minutes (safety margin before 15min limit)
const TRANSCRIPTION_TIMEOUT_MS = 30000; // 30 seconds per transcription
const MIN_SAMPLES_100MS = 2400; // 100ms at 24kHz (OpenAI minimum)
const SAMPLE_RATE = 24000; // OpenAI's default sample rate

class TranscriptionSession {
  private ws: WebSocket | null = null;
  private sessionStartTime: number = 0;
  private apiKey: string;
  private initialized: boolean = false;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async init(): Promise<void> {
    try {
      const uri = "wss://api.openai.com/v1/realtime?intent=transcription";

      this.ws = await new Promise((resolve, reject) => {
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

        // Connection timeout
        setTimeout(() => {
          if (!this.initialized) {
            wsWithAuth.close();
            reject(new Error('Connection timeout'));
          }
        }, 10000);
      });

      // Send session configuration for transcription
      const sessionUpdate = {
        type: "session.update",
        session: {
          type: "transcription",
          audio: {
            input: {
              format: {
                rate: SAMPLE_RATE,
                type: "audio/pcm"
              },
              noise_reduction: {
                type: "far_field"
              },
              transcription: {
                language: "en",
                model: "gpt-4o-mini-transcribe"
              },
              turn_detection: null
            }
          },
          include: [
            "item.input_audio_transcription.logprobs"
          ]
        }
      };

      if (this.ws) {
        this.ws.send(JSON.stringify(sessionUpdate));
        this.initialized = true;
        this.sessionStartTime = Date.now();
      } else {
        throw new Error('WebSocket connection failed');
      }
    } catch (error) {
      console.error('Failed to initialize transcription session:', error);
      this.disconnect();
      throw error;
    }
  }

  isActive(): boolean {
    if (!this.ws || !this.initialized) {
      return false;
    }

    // Check if WebSocket is still open
    if (this.ws.readyState !== WebSocket.OPEN) {
      return false;
    }

    // Check if session hasn't exceeded 14-minute limit
    const elapsedTime = Date.now() - this.sessionStartTime;
    if (elapsedTime >= SESSION_TIMEOUT_MS) {
      console.log('Session expired, will reconnect');
      return false;
    }

    return true;
  }

  async transcribe(audioData: Float32Array): Promise<string> {
    if (!this.ws || !this.initialized) {
      throw new Error('Transcription session not initialized');
    }

    // Check if we have enough audio data (minimum 100ms)
    if (audioData.length < MIN_SAMPLES_100MS) {
      console.warn(`Audio too short for transcription: ${audioData.length} samples (${(audioData.length / SAMPLE_RATE * 1000).toFixed(1)}ms), minimum required: ${MIN_SAMPLES_100MS} samples (100ms)`);
      return "";
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
        const timeout = setTimeout(() => {
          if (this.ws) {
            this.ws.removeEventListener('message', messageHandler);
          }
          reject(new Error('Transcription timeout'));
        }, TRANSCRIPTION_TIMEOUT_MS);

        // Clean up timeout if we finish early
        const originalHandler = messageHandler;
        const wrappedHandler = (event: MessageEvent) => {
          clearTimeout(timeout);
          originalHandler(event);
        };
        this.ws.removeEventListener('message', messageHandler);
        this.ws.addEventListener('message', wrappedHandler);
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
      this.sessionStartTime = 0;
    }
  }
}

// Module-level singleton session
let currentSession: TranscriptionSession | null = null;

/**
 * Transcribe audio using OpenAI's Realtime Transcription API.
 * Automatically manages WebSocket sessions with connection pooling and timeout handling.
 *
 * @param audio - Float32Array of audio samples at 16kHz (will be resampled to 24kHz internally)
 * @returns Promise<string> - Transcribed text
 */
export async function transcribe(audio: Float32Array): Promise<string> {
  try {
    // Get API key from environment
    const apiKey = import.meta.env.VITE_OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error('OPENAI_API_KEY environment variable not set');
    }

    // Resample from 16kHz to 24kHz (OpenAI's expected rate)
    const resampledAudio = resampleAudio(audio, 16000, SAMPLE_RATE);

    // Check if current session is active, create new one if needed
    if (!currentSession || !currentSession.isActive()) {
      console.log('Creating new transcription session');
      if (currentSession) {
        currentSession.disconnect();
      }
      currentSession = new TranscriptionSession(apiKey);
      await currentSession.init();
    }

    // Transcribe using current session
    const result = await currentSession.transcribe(resampledAudio);
    return result;

  } catch (error) {
    console.error('Transcription error:', error);

    // On error, invalidate current session
    if (currentSession) {
      currentSession.disconnect();
      currentSession = null;
    }

    throw error;
  }
}

/**
 * Simple linear resampling from source to target sample rate
 */
function resampleAudio(audio: Float32Array, sourceSampleRate: number, targetSampleRate: number): Float32Array {
  if (sourceSampleRate === targetSampleRate) {
    return audio;
  }

  const ratio = targetSampleRate / sourceSampleRate;
  const targetLength = Math.floor(audio.length * ratio);
  const resampled = new Float32Array(targetLength);

  for (let i = 0; i < targetLength; i++) {
    const sourceIndex = i / ratio;
    const index0 = Math.floor(sourceIndex);
    const index1 = Math.min(index0 + 1, audio.length - 1);
    const fraction = sourceIndex - index0;

    // Linear interpolation
    resampled[i] = audio[index0] * (1 - fraction) + audio[index1] * fraction;
  }

  return resampled;
}

/**
 * Manually disconnect the current session (e.g., on app shutdown)
 */
export function disconnectTranscription(): void {
  if (currentSession) {
    currentSession.disconnect();
    currentSession = null;
  }
}

