// Core WebRTC audio streaming functionality

// Types for WebSocket messages
export interface AnswerMessage {
  type: 'answer';
  answer: RTCSessionDescriptionInit;
}

export interface CandidateMessage {
  type: 'candidate';
  candidate: RTCIceCandidateInit;
}

export interface ErrorMessage {
  type: 'error';
  error: string;
}

export type WebSocketMessage = AnswerMessage | CandidateMessage | ErrorMessage;

// Extend Window interface for webkit compatibility
declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext;
  }
}

export interface AudioStreamingOptions {
  onStatusUpdate?: (msg: string, className?: string) => void;
  onWebSocketMessage?: (message: WebSocketMessage) => void;
  audioProcessorPath?: string;
}

export class WebRTCAudioStreamer {
  private ws: WebSocket | null = null;
  private pc: RTCPeerConnection | null = null;
  private dataChannel: RTCDataChannel | null = null;
  private mediaStream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private audioWorkletNode: AudioWorkletNode | null = null;
  private animationFrameId: number | null = null;
  private options: AudioStreamingOptions;

  constructor(options: AudioStreamingOptions = {}) {
    this.options = options;
  }

  private updateStatus(msg: string, className = '') {
    this.options.onStatusUpdate?.(msg, className);
  }

  connectToServer(host: string = window.location.host) {
    if (this.ws) {
      return; // Already connected or connecting
    }

    this.ws = new WebSocket(`ws://${host}`);

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.updateStatus('Ready', '');
    };

    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.updateStatus('Disconnected', '');
      this.ws = null;
    };

    this.ws.onerror = (err) => {
      console.error('WebSocket error:', err);
      this.updateStatus('Connection error', '');
    };

    this.ws.onmessage = async (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data) as WebSocketMessage;

        if (this.options.onWebSocketMessage) {
          this.options.onWebSocketMessage(message);
        }

        switch (message.type) {
          case 'answer':
            await this.handleAnswer(message.answer);
            break;
          case 'candidate':
            await this.handleCandidate(message.candidate);
            break;
          case 'error':
            console.error('Server error:', message.error);
            this.updateStatus('Server error: ' + message.error, '');
            break;
        }
      } catch (err) {
        console.error('Error handling message:', err);
      }
    };
  }

  private async handleAnswer(answer: RTCSessionDescriptionInit) {
    try {
      if (this.pc) {
        await this.pc.setRemoteDescription(new RTCSessionDescription(answer));
        console.log('Remote description set');
        this.updateStatus('Connected', 'connected');
      }
    } catch (err) {
      console.error('Error setting remote description:', err);
      this.updateStatus('Error: ' + (err as Error).message, '');
    }
  }

  private async handleCandidate(candidate: RTCIceCandidateInit) {
    try {
      if (this.pc) {
        await this.pc.addIceCandidate(new RTCIceCandidate({
          candidate: candidate.candidate,
          sdpMid: candidate.sdpMid,
          sdpMLineIndex: candidate.sdpMLineIndex
        }));
      }
    } catch (err) {
      console.error('Error adding ICE candidate:', err);
    }
  }

  async startStreaming() {
    try {
      // Connect to server first
      if (!this.ws) {
        this.connectToServer();
      }

      this.updateStatus('Requesting microphone access...', 'connecting');

      // Get microphone access
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      this.updateStatus('Creating peer connection...', 'connecting');

      // Create RTCPeerConnection
      this.pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      });

      // Create data channel for audio
      this.dataChannel = this.pc.createDataChannel('audio');

      this.dataChannel.onopen = () => {
        console.log('Data channel opened');
        this.updateStatus('Streaming audio...', 'recording');
        this.startAudioProcessing().catch(err => {
          console.error('Failed to start audio processing:', err);
          this.updateStatus('Error starting audio processing', '');
        });
      };

      this.dataChannel.onclose = () => {
        console.log('Data channel closed');
        this.updateStatus('Connection closed', '');
      };

      this.dataChannel.onerror = (err: Event) => {
        console.error('Data channel error:', err);
        this.updateStatus('Error: Data channel error', '');
      };

      // Handle ICE candidates
      this.pc.onicecandidate = (event) => {
        if (event.candidate) {
          if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
              type: 'candidate',
              candidate: {
                candidate: event.candidate.candidate,
                sdpMid: event.candidate.sdpMid,
                sdpMLineIndex: event.candidate.sdpMLineIndex
              }
            }));
          }
        }
      };

      this.pc.onconnectionstatechange = () => {
        if (this.pc) {
          console.log('Connection state:', this.pc.connectionState);
        }
      };

      // Create and send offer
      const offer = await this.pc.createOffer();
      await this.pc.setLocalDescription(offer);

      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({
          type: 'offer',
          offer: {
            type: offer.type,
            sdp: offer.sdp
          }
        }));
      }

      return true;
    } catch (err) {
      console.error('Error starting stream:', err);
      this.updateStatus('Error: ' + (err as Error).message, '');
      this.cleanup();
      throw err;
    }
  }

  private async startAudioProcessing() {
    if (!this.mediaStream) return;

    // Initialize AudioContext
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
      throw new Error('AudioContext not supported');
    }
    this.audioContext = new AudioContextClass();

    try {
      // Load the AudioWorklet processor
      const processorPath = this.options.audioProcessorPath || './audio-processor.js';
      await this.audioContext.audioWorklet.addModule(processorPath);

      // Create AudioWorklet node
      this.audioWorkletNode = new AudioWorkletNode(this.audioContext, 'audio-processor');

      // Handle messages from AudioWorklet
      this.audioWorkletNode.port.onmessage = (event) => {
        if (event.data.type === 'audioData' && this.dataChannel && this.dataChannel.readyState === 'open') {
          try {
            this.dataChannel.send(event.data.data);
          } catch (err) {
            console.error('Error sending audio:', err);
          }
        }
      };

      // Connect audio graph
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      source.connect(this.audioWorkletNode);
    } catch (err) {
      console.error('Error setting up AudioWorklet:', err);
      // Fallback to AnalyserNode approach if AudioWorklet fails
      console.warn('Falling back to AnalyserNode-based audio processing');
      this.startAudioProcessingFallback();
    }
  }

  private startAudioProcessingFallback() {
    if (!this.mediaStream || !this.audioContext) return;

    const source = this.audioContext.createMediaStreamSource(this.mediaStream);

    // Use AnalyserNode as a modern alternative for audio capture
    const analyser = this.audioContext.createAnalyser();
    analyser.fftSize = 8192; // This gives us 4096 samples in the time domain

    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);

    let lastSendTime = 0;
    const sendInterval = (4096 / this.audioContext.sampleRate) * 1000; // ~93ms at 44.1kHz

    const processAudio = () => {
      if (!this.audioContext || !this.dataChannel || this.dataChannel.readyState !== 'open') {
        return;
      }

      const now = performance.now();
      if (now - lastSendTime >= sendInterval) {
        // Get time domain data (audio samples)
        analyser.getFloatTimeDomainData(dataArray);

        // Take first 4096 samples to match our expected buffer size
        const sampleCount = Math.min(4096, dataArray.length);
        const pcmData = new Int16Array(sampleCount);

        for (let i = 0; i < sampleCount; i++) {
          const s = Math.max(-1, Math.min(1, dataArray[i]));
          pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        try {
          this.dataChannel.send(pcmData.buffer);
          lastSendTime = now;
        } catch (err) {
          console.error('Error sending audio:', err);
        }
      }

      this.animationFrameId = requestAnimationFrame(processAudio);
    };

    // Start the processing loop
    this.animationFrameId = requestAnimationFrame(processAudio);
  }

  stopStreaming() {
    this.cleanup();
    this.updateStatus('Stopped', '');
  }

  cleanup() {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.audioWorkletNode) {
      this.audioWorkletNode.disconnect();
      this.audioWorkletNode = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    if (this.dataChannel) {
      this.dataChannel.close();
      this.dataChannel = null;
    }

    if (this.pc) {
      this.pc.close();
      this.pc = null;
    }

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track: MediaStreamTrack) => track.stop());
      this.mediaStream = null;
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  get isStreaming(): boolean {
    return this.pc?.connectionState === 'connected' && this.dataChannel?.readyState === 'open';
  }
}