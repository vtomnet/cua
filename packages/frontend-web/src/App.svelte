<script lang="ts">
  import { onMount } from 'svelte';
  import {
    VadIterator,
    SmartTurnV3,
    Moonshine,
    OpenAIRealtime,
    RingBuffer,
    Timestamp,
    combineAudioChunks
  } from './main';

  // Application state
  let isRecording = false;
  let status = 'Ready - click Start Recording to begin';
  let statusClass = '';
  let speechSegments: Timestamp[] = [];
  let transcriptions: { text: string; timestamp: string }[] = [];
  let openaiResponses: { text: string; timestamp: string }[] = [];

  // Global instances (will be available from main.ts)
  let vad: VadIterator | null = null;
  let moonshine: Moonshine | null = null;
  let openaiRealtime: OpenAIRealtime | null = null;
  let smartTurn: SmartTurnV3 | null = null;
  let audioContext: AudioContext | null = null;
  let recordedChunks: Float32Array[] = [];
  let lastTranscriptionTime = 0;
  let currentSpeechBuffer: Float32Array[] = [];
  let silenceTimeout: NodeJS.Timeout | null = null;
  let ringBuffer: RingBuffer | null = null;
  let turnDetectionBuffer: Float32Array[] = [];
  let pendingTranscription: string = "";
  let isProcessingOpenAI = false;
  let lastRequestTimestamp = 0;
  let lastUserSpeechTime = 0;
  let lastTranscribedAudioLength = 0;
  // Minimum samples required before attempting a Moonshine transcription.
  // Prevents tiny slices (e.g. length 1) that cause ONNX shape errors.
  const MIN_TRANSCRIBE_SAMPLES = 256; // ~16ms at 16kHz
  // Guard to prevent overlapping ASR transcriptions (duplicates/races)
  let isTranscribing = false;
  type TranscriptionJob = { data: Float32Array; reason: 'speech_end' | 'timer' };
  let transcriptionQueue: TranscriptionJob[] = [];

  // Update status display
  function updateStatus(msg: string, className = '') {
    status = 'Status: ' + msg;
    // Map old class names to Tailwind classes
    switch (className) {
      case 'loading':
        statusClass = 'bg-yellow-100 text-yellow-800 border-yellow-300';
        break;
      case 'recording':
        statusClass = 'bg-red-100 text-red-800 border-red-300';
        break;
      case 'success':
        statusClass = 'bg-green-100 text-green-800 border-green-300';
        break;
      case 'error':
        statusClass = 'bg-red-100 text-red-800 border-red-300';
        break;
      default:
        statusClass = 'bg-gray-100 text-gray-800 border-gray-300';
    }
  }

  // Display transcriptions
  function displayTranscription(transcription: string, timestamp: number) {
    const timeStr = new Date(timestamp).toLocaleTimeString();
    transcriptions = [...transcriptions, { text: transcription, timestamp: timeStr }];
  }

  // Display OpenAI responses
  function displayOpenAIResponse(response: any, timestamp: number) {
    let responseText = "";
    for (const item of response.response.output) {
      switch (item.type) {
        case "message":
          const content = item.content.map((x: any) => x.text).join(' ');
          responseText += content;
          break;
        case "function_call":
          const functionMsg = `Called ${item.name}(${JSON.stringify(item.arguments)})`;
          responseText += "\n" + functionMsg;
          break;
        case "function_call_output":
          const outputMsg = `${item.name} finished`;
          responseText += "\n" + outputMsg;
          break;
        default:
          console.log(`Unhandled item type: ${item.type}`);
      }
    }

    const timeStr = new Date(timestamp).toLocaleTimeString();
    openaiResponses = [...openaiResponses, { text: responseText, timestamp: timeStr }];
  }

  // Send transcription to OpenAI and get response
  async function sendToOpenAI(transcription: string) {
    if (!openaiRealtime || !transcription.trim() || isProcessingOpenAI) return;

    // Prevent duplicate requests
    isProcessingOpenAI = true;
    lastRequestTimestamp = Date.now();

    try {
      console.log('Sending to OpenAI:', transcription);
      const response = await openaiRealtime.sendTextAndGetResponse(transcription);
      console.log('OpenAI response:', response);

      // Only display response if user hasn't spoken since the request was sent
      if (lastUserSpeechTime <= lastRequestTimestamp) {
        displayOpenAIResponse(response, Date.now());
      } else {
        console.log('Ignoring OpenAI response - user has spoken since request was sent');
      }
    } catch (error) {
      console.error('Error getting OpenAI response:', error);
      // Only display error if user hasn't spoken since the request was sent
      if (lastUserSpeechTime <= lastRequestTimestamp) {
        openaiResponses = [...openaiResponses, { text: 'Error getting response from OpenAI', timestamp: new Date().toLocaleTimeString() }];
      }
    } finally {
      isProcessingOpenAI = false;
    }
  }

  // Process audio chunk for real-time VAD and transcription
  let speechStarted = false;
  let silenceCounter = 0;
  const SILENCE_THRESHOLD = 10; // chunks of silence to detect speech end

  async function processAudioChunkLocal(audioChunk: Float32Array) {
    if (!vad || !ringBuffer) return;

    // Process chunk with VAD to detect speech activity
    try {
      // Use a sliding window approach for real-time detection
      const windowSize = 512; // Match VAD window size
      for (let i = 0; i < audioChunk.length; i += windowSize) {
        const chunk = audioChunk.slice(i, Math.min(i + windowSize, audioChunk.length));
        if (chunk.length === windowSize) {
          await vad.predict(chunk);
        }
      }

      // Check VAD state for speech activity
      const hasCurrentSpeech = vad.triggered;

      // Always write incoming audio to the ring buffer after VAD processing
      // This ensures the ring buffer contains the chunk that triggered speech detection
      ringBuffer.write(audioChunk);

      if (hasCurrentSpeech && !speechStarted) {
        // Speech started - include buffered audio to prevent losing first bit of speech
        speechStarted = true;
        silenceCounter = 0;
        lastUserSpeechTime = Date.now(); // Track when user starts speaking
        lastTranscribedAudioLength = 0; // Reset for new speech segment

        // Get pre-speech audio from ring buffer and start current speech buffer
        const bufferedAudio = ringBuffer.read();
        currentSpeechBuffer = [bufferedAudio, audioChunk];
      } else if (speechStarted) {
        // Speech is ongoing - add chunk to current speech buffer
        currentSpeechBuffer.push(audioChunk);

        if (!hasCurrentSpeech) {
          // Potential speech end, count silence
          silenceCounter++;

          if (silenceCounter >= SILENCE_THRESHOLD) {
            // Speech ended, transcribe the current buffer
            const currentAudio = combineAudioChunksLocal(currentSpeechBuffer);
            if (currentAudio.length > 0 && currentAudio.length > lastTranscribedAudioLength) {
              // Only transcribe new audio that hasn't been transcribed yet.
              // Advance lastTranscribedAudioLength BEFORE awaiting to avoid race where
              // overlapping processAudioChunkLocal calls resend the same slice.
              const previousLength = lastTranscribedAudioLength;
              const targetLength = currentAudio.length;
              const newAudio = currentAudio.slice(previousLength);
              if (newAudio.length > 0) {
              // Skip if too small to transcribe yet; wait for more audio
              if (newAudio.length >= MIN_TRANSCRIBE_SAMPLES) {
                lastTranscribedAudioLength = targetLength;
                try {
                  await transcribeSpeechLocal(newAudio, 'speech_end');
                } catch (e) {
                  // Roll back so audio can be retried on failure
                  lastTranscribedAudioLength = previousLength;
                  throw e;
                }
              }
              }
            }
            // Reset for next speech segment
            currentSpeechBuffer = [];
            // Don't reset lastTranscribedAudioLength here - keep it to prevent duplicate transcriptions
            lastTranscriptionTime = Date.now();
            speechStarted = false;
            silenceCounter = 0;
          }
        } else {
          // Reset silence counter if speech continues
          silenceCounter = 0;
        }
      }

      // Check for 5-second interval transcription
      const now = Date.now();
      if (now - lastTranscriptionTime >= 5000 && currentSpeechBuffer.length > 0) {
        // Check if there's ongoing speech to transcribe
        const currentAudio = combineAudioChunksLocal(currentSpeechBuffer);
        if (currentAudio.length > lastTranscribedAudioLength) {
          // Advance pointer before awaiting to avoid duplicate slices
            const previousLength = lastTranscribedAudioLength;
            const targetLength = currentAudio.length;
            const newAudio = currentAudio.slice(previousLength);
            if (newAudio.length > 0) {
              if (newAudio.length >= MIN_TRANSCRIBE_SAMPLES) {
                lastTranscribedAudioLength = targetLength;
                try {
                  await transcribeSpeechLocal(newAudio, 'timer');
                } catch (e) {
                  lastTranscribedAudioLength = previousLength;
                  throw e;
                }
              }
              lastTranscriptionTime = now;
            }
        }
      }

    } catch (error) {
      console.error('Error processing audio chunk:', error);
    }
  }

  // Helper function to combine audio chunks into a single Float32Array
  function combineAudioChunksLocal(chunks: Float32Array[]): Float32Array {
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

  // Handle transcription for speech segment with turn detection
  async function transcribeSpeechLocal(audioData: Float32Array, reason: 'speech_end' | 'timer') {
    if (!moonshine || !smartTurn) return;

    // Queue if a transcription is already in progress
    if (isTranscribing) {
      transcriptionQueue.push({ data: audioData, reason });
      return;
    }

    isTranscribing = true;

    const processOne = async (data: Float32Array, why: 'speech_end' | 'timer') => {
      try {
        const transcription = await moonshine!.generate(data);
        if (transcription.trim()) {
          displayTranscription(transcription.trim(), Date.now());

          // Accumulate transcription for turn detection
          pendingTranscription += (pendingTranscription ? ' ' : '') + transcription.trim();

          // Always accumulate audio for turn detection so we have the full window
          turnDetectionBuffer.push(data);

          // Keep only last 8 seconds of audio for turn detection (16000 samples per second)
          const maxSamples = 8 * 16000;
          let totalSamples = turnDetectionBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
          while (totalSamples > maxSamples && turnDetectionBuffer.length > 1) {
            const removed = turnDetectionBuffer.shift();
            if (removed) totalSamples -= removed.length;
          }

          // Only run endpoint detection (and thus send to OpenAI) on explicit speech-end events
          if (why === 'speech_end') {
            const combinedAudio = combineAudioChunksLocal(turnDetectionBuffer);
            const turnResult = await smartTurn!.predictEndpoint(combinedAudio);
            console.log(`Turn detection (reason=${why}): prediction=${turnResult.prediction}, probability=${turnResult.probability.toFixed(3)}`);
            if (turnResult.prediction === 1) {
              if (pendingTranscription.trim() && !isProcessingOpenAI) {
                const transcriptionToSend = pendingTranscription.trim();
                pendingTranscription = ""; // Clear immediately to prevent duplicates
                await sendToOpenAI(transcriptionToSend);
                turnDetectionBuffer = [];
              }
            }
          } else {
            // For timer-based partials, we explicitly skip endpoint decision
            console.log('Timer-based transcription buffered (no endpoint check)');
          }
        }
      } catch (error) {
        console.error('Error transcribing speech or detecting turn:', error);
      }
    };

    try {
      // Process initial audio
      await processOne(audioData, reason);
      // Drain queue sequentially
      while (transcriptionQueue.length > 0) {
        const next = transcriptionQueue.shift();
        if (next) {
          await processOne(next.data, next.reason);
        }
      }
    } finally {
      isTranscribing = false;
    }
  }

  // Initialize models
  async function initializeModelsLocal() {
    try {
      updateStatus('Initializing VAD model...', 'loading');
      vad = new VadIterator("/models/silero_vad.onnx");
      await vad.init();
      updateStatus('VAD loaded, initializing Smart Turn v3...', 'loading');

      smartTurn = new SmartTurnV3();
      await smartTurn.init();
      updateStatus('Smart Turn v3 loaded, loading ASR...', 'loading');

      moonshine = new Moonshine("moonshine-base");
      await moonshine.loadModel();
      updateStatus('ASR loaded, connecting to OpenAI...', 'loading');

      const apiKey = import.meta.env.VITE_OPENAI_API_KEY;
      if (!apiKey) {
        throw new Error('VITE_OPENAI_API_KEY not found in environment variables');
      }

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

  // Create audio processor worklet as blob
  function createAudioProcessorBlobLocal(): string {
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
        const initialized = await initializeModelsLocal();
        if (!initialized) return;
      }

      // Create audio worklet for real-time processing
      await audioContext.audioWorklet.addModule(createAudioProcessorBlobLocal());
      const processorNode = new AudioWorkletNode(audioContext, 'vad-processor');

      processorNode.port.onmessage = async (event) => {
        const { audioData } = event.data;
        const audioChunk = new Float32Array(audioData);
        recordedChunks.push(audioChunk);

        // Real-time processing for speech detection and transcription
        await processAudioChunkLocal(audioChunk);
      };

      source.connect(processorNode);
      processorNode.connect(audioContext.destination);

      isRecording = true;
      recordedChunks = [];
      currentSpeechBuffer = [];
      lastTranscriptionTime = Date.now();
      speechStarted = false;
      silenceCounter = 0;
      turnDetectionBuffer = [];
      pendingTranscription = "";
      isProcessingOpenAI = false;
      lastRequestTimestamp = 0;
      lastUserSpeechTime = 0;
      lastTranscribedAudioLength = 0;

      // Initialize ring buffer to store 1 second of audio (16000 samples at 16kHz)
      ringBuffer = new RingBuffer(16000);

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

    // Clear any pending timeout
    if (silenceTimeout) {
      clearTimeout(silenceTimeout);
      silenceTimeout = null;
    }

    // Send any remaining transcription to OpenAI
    if (pendingTranscription.trim() && !isProcessingOpenAI) {
      const transcriptionToSend = pendingTranscription.trim();
      pendingTranscription = ""; // Clear immediately to prevent duplicates
      await sendToOpenAI(transcriptionToSend);
    }

    // Stop audio context
    await audioContext.close();
    audioContext = null;

    // Clear ring buffer
    if (ringBuffer) {
      ringBuffer.clear();
      ringBuffer = null;
    }

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
      speechSegments = timestamps;

      updateStatus(`Processing complete - ${timestamps.length} speech segments found`, 'success');
    } catch (error) {
      console.error('Error processing recorded audio:', error);
      updateStatus('Error processing recorded audio', 'error');
    }
  }

  onMount(() => {
    updateStatus('Ready - click Start Recording to begin');
  });
</script>

<main class="max-w-4xl mx-auto p-4 md:p-8 font-sans leading-relaxed text-gray-800 bg-white">
  <h1 class="text-4xl font-semibold mb-4 text-gray-900 leading-tight">Voice Activity Detection</h1>

  <div class="status p-4 my-4 rounded-lg font-semibold border transition-all duration-200 ease-in-out {statusClass}">{status}</div>

  <div class="bg-slate-50 p-6 my-8 rounded-lg border border-gray-200">
    <h2 class="text-3xl font-medium mb-4 text-gray-700">Microphone Recording</h2>
    <button
      on:click={startRecording}
      disabled={isRecording}
      class="bg-blue-500 text-white px-6 py-3 rounded-md cursor-pointer mr-3 text-sm font-medium transition-all duration-200 ease-in-out shadow-sm hover:bg-blue-600 hover:shadow-md disabled:bg-gray-400 disabled:cursor-not-allowed disabled:opacity-60 md:w-auto w-full mb-2 md:mb-0"
    >
      Start Recording
    </button>
    <button
      on:click={stopRecording}
      disabled={!isRecording}
      class="bg-blue-500 text-white px-6 py-3 rounded-md cursor-pointer mr-3 text-sm font-medium transition-all duration-200 ease-in-out shadow-sm hover:bg-blue-600 hover:shadow-md disabled:bg-gray-400 disabled:cursor-not-allowed disabled:opacity-60 md:w-auto w-full"
    >
      Stop Recording
    </button>
  </div>

  <div class="mt-8 p-6 bg-white rounded-lg border border-gray-200 shadow-sm">
    <h3 class="text-2xl font-medium mb-4 text-gray-600">Speech Segments:</h3>
    <div class="mt-4">
      {#if speechSegments.length === 0}
        <p class="text-gray-500">No speech segments detected.</p>
      {:else}
        {#each speechSegments as segment, index}
          <div class="px-4 py-3 my-2 bg-slate-100 border-l-4 border-blue-500 rounded-r text-sm font-mono">
            Segment {index + 1}: {Math.round((segment.start / 16000) * 100) / 100}s - {Math.round((segment.end / 16000) * 100) / 100}s
          </div>
        {/each}
      {/if}
    </div>
  </div>

  <div class="mt-8 p-6 bg-white rounded-lg border border-gray-200 shadow-sm">
    <h3 class="text-2xl font-medium mb-4 text-gray-600">Transcriptions:</h3>
    <div class="max-h-96 overflow-y-auto border border-gray-300 rounded-md p-4 bg-gray-50 mt-4">
      {#each transcriptions as transcription}
        <div class="py-2 border-b border-gray-200 leading-relaxed">
          <strong>[{transcription.timestamp}]</strong> {transcription.text}
        </div>
      {/each}
      {#each openaiResponses as response}
        <div class="bg-blue-50 p-3 mt-1 rounded-md border-l-4 border-blue-500">
          <strong>[{response.timestamp}] GPT:</strong> {response.text}
        </div>
      {/each}
    </div>
  </div>

  <div class="mt-8 p-4 text-sm text-gray-500 bg-gray-50 rounded-md border border-gray-200">
    Click "Start Recording" to begin recording from your microphone. The system will detect speech segments using VAD and transcribe them using Moonshine ASR on speech end or every 5 seconds.
  </div>
</main>
