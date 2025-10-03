// AudioWorklet processor for handling real-time audio processing
class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 4096;
    this.bufferIndex = 0;
    this.buffer = new Float32Array(this.bufferSize);
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input.length > 0) {
      const inputChannel = input[0]; // Get first channel

      for (let i = 0; i < inputChannel.length; i++) {
        this.buffer[this.bufferIndex] = inputChannel[i];
        this.bufferIndex++;

        // When buffer is full, send it to main thread
        if (this.bufferIndex >= this.bufferSize) {
          // Convert Float32Array to Int16Array (PCM)
          const pcmData = new Int16Array(this.bufferSize);
          for (let j = 0; j < this.bufferSize; j++) {
            const s = Math.max(-1, Math.min(1, this.buffer[j]));
            pcmData[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }

          // Send audio data to main thread
          this.port.postMessage({
            type: 'audioData',
            data: pcmData.buffer
          });

          // Reset buffer
          this.bufferIndex = 0;
        }
      }
    }

    // Keep the processor alive
    return true;
  }
}

registerProcessor('audio-processor', AudioProcessor);