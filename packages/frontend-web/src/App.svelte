<script lang="ts">
  import { onMount } from 'svelte';
  import {
    VadIterator,
    SmartTurnV3,
    OpenAIRealtimeTranscription,
    RingBuffer,
    Timestamp,
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
  let openaiTranscription: OpenAIRealtimeTranscription | null = null;
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
  // Minimum samples required before attempting transcription.
  // OpenAI requires at least 100ms of audio at 24kHz, which equals ~100ms at 16kHz too
  const MIN_TRANSCRIBE_SAMPLES = Math.ceil(16000 * 0.1); // 100ms at 16kHz = 1600 samples
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
  function displayOpenAIResponse(responseText: string, timestamp: number) {
    const timeStr = new Date(timestamp).toLocaleTimeString();
    openaiResponses = [...openaiResponses, { text: responseText, timestamp: timeStr }];
  }

  // Send transcription to OpenAI and get response
  async function sendToOpenAI(transcription: string) {
    if (!transcription.trim() || isProcessingOpenAI) return;

    // Prevent duplicate requests
    isProcessingOpenAI = true;
    lastRequestTimestamp = Date.now();

    try {
      console.log('Sending transcription to local generator:', transcription);
      const response = await fetch('http://localhost:3000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: transcription }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Local generator response:', data);

      const choice = data?.choices?.[0];
      const message = choice?.message ?? null;
      const responseLines: string[] = [];

      const reasoning = typeof message?.reasoning === 'string' ? message.reasoning.trim() : '';
      if (reasoning) {
        responseLines.push(`Reasoning: ${reasoning}`);
      }

      const content = message?.content;
      if (Array.isArray(content)) {
        const contentText = content
          .map((entry: unknown) => {
            if (typeof entry === 'string') return entry;
            if (entry && typeof entry === 'object') {
              const maybeText = (entry as { text?: string }).text;
              if (typeof maybeText === 'string') return maybeText;
            }
            return '';
          })
          .join(' ')
          .trim();
        if (contentText) {
          responseLines.push(contentText);
        }
      } else if (typeof content === 'string' && content.trim()) {
        responseLines.push(content.trim());
      }

      const toolCalls = Array.isArray(message?.tool_calls) ? message.tool_calls : [];
      for (const call of toolCalls) {
        const toolName = call?.function?.name ?? call?.name ?? 'unknown_tool';
        const args = call?.function?.arguments ?? call?.arguments ?? '';
        const argsStr = typeof args === 'string' ? args : JSON.stringify(args);
        responseLines.push(`Called ${toolName}(${argsStr})`);
      }

      if (!responseLines.length) {
        const fallback = choice?.text ?? data?.choices?.[0]?.text;
        if (typeof fallback === 'string' && fallback.trim()) {
          responseLines.push(fallback.trim());
        } else {
          responseLines.push(JSON.stringify(data));
        }
      }

      const responseText = responseLines.join('\n');

      // Only display response if user hasn't spoken since the request was sent
      if (lastUserSpeechTime <= lastRequestTimestamp) {
        displayOpenAIResponse(responseText, Date.now());
      } else {
        console.log('Ignoring generator response - user has spoken since request was sent');
      }
    } catch (error) {
      console.error('Error getting generator response:', error);
      // Only display error if user hasn't spoken since the request was sent
      if (lastUserSpeechTime <= lastRequestTimestamp) {
        openaiResponses = [...openaiResponses, { text: 'Error getting response from generator', timestamp: new Date().toLocaleTimeString() }];
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
    // We want to capture a small amount of pre-speech audio that occurs
    // just before VAD.triggered flips to true. To do that we snapshot
    // the ring buffer BEFORE writing the current chunk. If this chunk
    // is the one that triggers speech, we then append the chunk after
    // the pre-speech audio snapshot so we don't lose the leading edge.
    const preChunkBufferedAudio = ringBuffer.read();

    try {
      // Run VAD on the chunk (window size matches worklet buffer size)
      const windowSize = 512;
      for (let i = 0; i < audioChunk.length; i += windowSize) {
        const chunk = audioChunk.slice(i, Math.min(i + windowSize, audioChunk.length));
        if (chunk.length === windowSize) {
          await vad.predict(chunk);
        }
      }

      const hasCurrentSpeech = vad.triggered;

      if (hasCurrentSpeech && !speechStarted) {
        // Rising edge of speech detection: start a new speech buffer.
        // Use the snapshot taken BEFORE writing this chunk so we include
        // the preceding context (potentially early speech not yet over threshold).
        speechStarted = true;
        silenceCounter = 0;
        lastUserSpeechTime = Date.now();
        lastTranscribedAudioLength = 0;

        currentSpeechBuffer = [];
        if (preChunkBufferedAudio.length > 0) {
          currentSpeechBuffer.push(preChunkBufferedAudio);
        }
        currentSpeechBuffer.push(audioChunk);
      } else if (speechStarted) {
        // Ongoing speech
        currentSpeechBuffer.push(audioChunk);

        if (!hasCurrentSpeech) {
          // Potential end-of-speech: count silent chunks
          silenceCounter++;
          if (silenceCounter >= SILENCE_THRESHOLD) {
            const currentAudio = combineAudioChunksLocal(currentSpeechBuffer);
            if (currentAudio.length > 0 && currentAudio.length > lastTranscribedAudioLength) {
              const previousLength = lastTranscribedAudioLength;
              const targetLength = currentAudio.length;
              const newAudio = currentAudio.slice(previousLength);
              if (newAudio.length >= MIN_TRANSCRIBE_SAMPLES) {
                lastTranscribedAudioLength = targetLength;
                try {
                  await transcribeSpeechLocal(newAudio, 'speech_end');
                } catch (e) {
                  lastTranscribedAudioLength = previousLength;
                  throw e;
                }
              }
            }
            currentSpeechBuffer = [];
            lastTranscriptionTime = Date.now();
            speechStarted = false;
            silenceCounter = 0;
          }
        } else {
          // Speech continues; reset silence counter
            silenceCounter = 0;
        }
      }

      // Periodic (timer) partial transcription every 5 seconds of ongoing speech
      const now = Date.now();
      if (now - lastTranscriptionTime >= 5000 && currentSpeechBuffer.length > 0) {
        const currentAudio = combineAudioChunksLocal(currentSpeechBuffer);
        if (currentAudio.length > lastTranscribedAudioLength) {
          const previousLength = lastTranscribedAudioLength;
          const targetLength = currentAudio.length;
          const newAudio = currentAudio.slice(previousLength);
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

      // Finally write the current chunk into the ring buffer so it is available
      // as pre-speech context for the NEXT chunk if needed.
      ringBuffer.write(audioChunk);
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

  // Helper function to resample audio from 16kHz to 24kHz for OpenAI
  function resample16to24(input: Float32Array): Float32Array {
    const inputSampleRate = 16000;
    const outputSampleRate = 24000;
    const ratio = outputSampleRate / inputSampleRate; // 1.5
    const outputLength = Math.floor(input.length * ratio);
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const inputIndex = i / ratio;
      const inputIndexFloor = Math.floor(inputIndex);
      const inputIndexCeil = Math.min(inputIndexFloor + 1, input.length - 1);
      const fraction = inputIndex - inputIndexFloor;

      // Linear interpolation
      output[i] = input[inputIndexFloor] * (1 - fraction) + input[inputIndexCeil] * fraction;
    }

    return output;
  }

  // Handle transcription for speech segment with turn detection
  async function transcribeSpeechLocal(audioData: Float32Array, reason: 'speech_end' | 'timer') {
    if (!openaiTranscription || !smartTurn) return;

    // Queue if a transcription is already in progress
    if (isTranscribing) {
      transcriptionQueue.push({ data: audioData, reason });
      return;
    }

    isTranscribing = true;

    const processOne = async (data: Float32Array, why: 'speech_end' | 'timer') => {
      try {
        // Resample audio from 16kHz to 24kHz for OpenAI
        const resampledData = resample16to24(data);
        const transcription = await openaiTranscription!.transcribe(resampledData);
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

      const apiKey = import.meta.env.VITE_OPENAI_API_KEY;
      if (!apiKey) {
        throw new Error('VITE_OPENAI_API_KEY not found in environment variables');
      }

      // Initialize OpenAI transcription (uses 24kHz but we'll resample if needed)
      openaiTranscription = new OpenAIRealtimeTranscription(apiKey, 24000);
      await openaiTranscription.init();
      updateStatus('OpenAI Transcription loaded, ready to generate responses locally', 'success');
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
      if (!vad || !openaiTranscription) {
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
        <div class="bg-blue-50 p-3 mt-1 rounded-md border-l-4 border-blue-500" style="white-space: pre-line;">
          <strong>[{response.timestamp}] GPT:</strong> {response.text}
        </div>
      {/each}
    </div>
  </div>

  <div class="mt-8 p-4 text-sm text-gray-500 bg-gray-50 rounded-md border border-gray-200">
    Click "Start Recording" to begin recording from your microphone. The system will detect speech segments using VAD, transcribe them with OpenAI's API on speech end or every 5 seconds, and forward transcripts to the local generator service for replies.
  </div>
</main>
