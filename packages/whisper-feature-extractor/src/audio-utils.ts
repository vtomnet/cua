/**
 * Audio processing functions for feature extraction
 * Pure TypeScript implementation using Float64Array for high precision
 */

/**
 * Convert frequency from hertz to mels
 */
export function hertzToMel(freq: number | number[], melScale: string = "htk"): number | number[] {
  if (!["slaney", "htk", "kaldi"].includes(melScale)) {
    throw new Error('mel_scale should be one of "htk", "slaney" or "kaldi"');
  }

  const convertSingle = (f: number): number => {
    if (melScale === "htk") {
      return 2595.0 * Math.log10(1.0 + (f / 700.0));
    } else if (melScale === "kaldi") {
      return 1127.0 * Math.log(1.0 + (f / 700.0));
    }

    // slaney
    const minLogHertz = 1000.0;
    const minLogMel = 15.0;
    const logstep = 27.0 / Math.log(6.4);
    const mels = 3.0 * f / 200.0;

    if (f >= minLogHertz) {
      return minLogMel + Math.log(f / minLogHertz) * logstep;
    } else {
      return mels;
    }
  };

  if (Array.isArray(freq)) {
    return freq.map(convertSingle);
  } else {
    return convertSingle(freq);
  }
}

/**
 * Convert frequency from mels to hertz
 */
export function melToHertz(mels: number | number[], melScale: string = "htk"): number | number[] {
  if (!["slaney", "htk", "kaldi"].includes(melScale)) {
    throw new Error('mel_scale should be one of "htk", "slaney" or "kaldi"');
  }

  const convertSingle = (m: number): number => {
    if (melScale === "htk") {
      return 700.0 * (Math.pow(10, m / 2595.0) - 1.0);
    } else if (melScale === "kaldi") {
      return 700.0 * (Math.exp(m / 1127.0) - 1.0);
    }

    // slaney
    const minLogHertz = 1000.0;
    const minLogMel = 15.0;
    const logstep = Math.log(6.4) / 27.0;
    const freq = 200.0 * m / 3.0;

    if (m >= minLogMel) {
      return minLogHertz * Math.exp(logstep * (m - minLogMel));
    } else {
      return freq;
    }
  };

  if (Array.isArray(mels)) {
    return mels.map(convertSingle);
  } else {
    return convertSingle(mels);
  }
}

/**
 * Creates a triangular filter bank
 */
function createTriangularFilterBank(fftFreqs: number[], filterFreqs: number[]): number[][] {
  const numFreqBins = fftFreqs.length;
  const numFilters = filterFreqs.length - 2;
  const filterBank: number[][] = [];

  for (let i = 0; i < numFreqBins; i++) {
    filterBank[i] = [];
  }

  for (let j = 0; j < numFilters; j++) {
    const left = filterFreqs[j];
    const center = filterFreqs[j + 1];
    const right = filterFreqs[j + 2];

    if (left === undefined || center === undefined || right === undefined) {
      continue;
    }

    for (let i = 0; i < numFreqBins; i++) {
      const freq = fftFreqs[i];
      if (freq === undefined) continue;

      let weight = 0;

      if (freq >= left && freq <= center) {
        weight = (freq - left) / (center - left);
      } else if (freq > center && freq <= right) {
        weight = (right - freq) / (right - center);
      }

      filterBank[i]![j] = Math.max(0, weight);
    }
  }

  return filterBank;
}

/**
 * Creates a mel filter bank
 */
export function melFilterBank(
  numFrequencyBins: number,
  numMelFilters: number,
  minFrequency: number,
  maxFrequency: number,
  samplingRate: number,
  norm: string | null = null,
  melScale: string = "htk",
  triangularizeInMelSpace: boolean = false
): number[][] {
  if (norm !== null && norm !== "slaney") {
    throw new Error('norm must be one of null or "slaney"');
  }

  if (numFrequencyBins < 2) {
    throw new Error(`Require numFrequencyBins: ${numFrequencyBins} >= 2`);
  }

  if (minFrequency > maxFrequency) {
    throw new Error(`Require minFrequency: ${minFrequency} <= maxFrequency: ${maxFrequency}`);
  }

  // Center points of the triangular mel filters
  const melMin = hertzToMel(minFrequency, melScale) as number;
  const melMax = hertzToMel(maxFrequency, melScale) as number;
  const melFreqs: number[] = [];

  for (let i = 0; i <= numMelFilters + 1; i++) {
    melFreqs.push(melMin + (i * (melMax - melMin)) / (numMelFilters + 1));
  }

  const filterFreqs = melToHertz(melFreqs, melScale) as number[];

  let fftFreqs: number[];
  let filterFreqsForTriangular: number[];

  if (triangularizeInMelSpace) {
    // frequencies of FFT bins in Hz, but filters triangularized in mel space
    const fftBinWidth = samplingRate / ((numFrequencyBins - 1) * 2);
    const fftFreqsHz: number[] = [];
    for (let i = 0; i < numFrequencyBins; i++) {
      fftFreqsHz.push(fftBinWidth * i);
    }
    fftFreqs = hertzToMel(fftFreqsHz, melScale) as number[];
    filterFreqsForTriangular = melFreqs;
  } else {
    // frequencies of FFT bins in Hz
    fftFreqs = [];
    for (let i = 0; i < numFrequencyBins; i++) {
      fftFreqs.push((i * samplingRate) / (2 * (numFrequencyBins - 1)));
    }
    filterFreqsForTriangular = filterFreqs;
  }

  const triangularFilters = createTriangularFilterBank(fftFreqs, filterFreqsForTriangular);

  // Transpose to get melFilters[numMelFilters][numFrequencyBins]
  const melFilters: number[][] = [];
  for (let j = 0; j < numMelFilters; j++) {
    melFilters[j] = [];
    for (let i = 0; i < numFrequencyBins; i++) {
      melFilters[j]![i] = triangularFilters[i]?.[j] || 0;
    }
  }

  if (norm === "slaney") {
    // Slaney-style mel is scaled to be approx constant energy per channel
    for (let j = 0; j < numMelFilters; j++) {
      const freqHigh = filterFreqs[j + 2] ?? 0;
      const freqLow = filterFreqs[j] ?? 0;
      const enorm = 2.0 / (freqHigh - freqLow);
      for (let i = 0; i < numFrequencyBins; i++) {
        melFilters[j]![i] = (melFilters[j]![i] ?? 0) * enorm;
      }
    }
  }

  return melFilters;
}

/**
 * Returns a window function array
 */
export function windowFunction(
  windowLength: number,
  name: string = "hann",
  periodic: boolean = true,
  frameLength?: number,
  center: boolean = true
): Float64Array {
  const length = periodic ? windowLength + 1 : windowLength;
  let window: Float64Array;

  if (name === "boxcar") {
    window = new Float64Array(length).fill(1);
  } else if (name === "hamming" || name === "hamming_window") {
    window = new Float64Array(length);
    for (let i = 0; i < length; i++) {
      window[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (length - 1));
    }
  } else if (name === "hann" || name === "hann_window") {
    window = new Float64Array(length);
    for (let i = 0; i < length; i++) {
      window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (length - 1));
    }
  } else if (name === "povey") {
    window = new Float64Array(length);
    for (let i = 0; i < length; i++) {
      const hann = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (length - 1));
      window[i] = Math.pow(hann, 0.85);
    }
  } else {
    throw new Error(`Unknown window type '${name}'`);
  }

  if (periodic) {
    window = window.slice(0, -1);
  }

  if (frameLength === undefined) {
    return window;
  }

  if (windowLength > frameLength) {
    throw new Error(
      `Length of the window (${windowLength}) may not be larger than frame_length (${frameLength})`
    );
  }

  const paddedWindow = new Float64Array(frameLength);
  const offset = center ? Math.floor((frameLength - windowLength) / 2) : 0;
  paddedWindow.set(window, offset);

  return paddedWindow;
}

/**
 * Complex number operations
 */
export class Complex {
  constructor(public real: number, public imag: number) {}

  multiply(other: Complex): Complex {
    return new Complex(
      this.real * other.real - this.imag * other.imag,
      this.real * other.imag + this.imag * other.real
    );
  }

  add(other: Complex): Complex {
    return new Complex(this.real + other.real, this.imag + other.imag);
  }

  magnitude(): number {
    return Math.sqrt(this.real * this.real + this.imag * this.imag);
  }

  magnitudeSquared(): number {
    return this.real * this.real + this.imag * this.imag;
  }
}

/**
 * Fast Fourier Transform implementation
 */
export function fft(x: Complex[] | Float64Array): Complex[] {
  // Convert Float64Array to Complex array if needed
  let complexInput: Complex[];
  if (x instanceof Float64Array) {
    complexInput = Array.from(x, val => new Complex(val, 0));
  } else {
    complexInput = x;
  }

  const N = complexInput.length;
  if (N <= 1) return complexInput;

  // Divide
  const even: Complex[] = [];
  const odd: Complex[] = [];
  for (let i = 0; i < N; i++) {
    if (i % 2 === 0) {
      even.push(complexInput[i]!);
    } else {
      odd.push(complexInput[i]!);
    }
  }

  // Conquer
  const evenFft = fft(even);
  const oddFft = fft(odd);

  // Combine
  const result: Complex[] = new Array(N);
  for (let k = 0; k < N / 2; k++) {
    const t = oddFft[k]!.multiply(new Complex(Math.cos(-2 * Math.PI * k / N), Math.sin(-2 * Math.PI * k / N)));
    result[k] = evenFft[k]!.add(t);
    result[k + N / 2] = evenFft[k]!.add(t.multiply(new Complex(-1, 0)));
  }

  return result;
}

/**
 * Real FFT implementation
 */
function rfft(x: Float64Array): Complex[] {
  const N = x.length;
  // Pad to next power of 2 for efficiency
  const paddedLength = Math.pow(2, Math.ceil(Math.log2(N)));
  const paddedX: Complex[] = [];

  for (let i = 0; i < paddedLength; i++) {
    paddedX.push(new Complex(i < N ? (x[i] ?? 0) : 0, 0));
  }

  const fullFft = fft(paddedX);
  // Return only positive frequencies (including DC and Nyquist)
  return fullFft.slice(0, Math.floor(paddedLength / 2) + 1);
}

/**
 * Calculates a spectrogram using STFT
 */
export function spectrogram(
  waveform: Float64Array,
  window: Float64Array,
  frameLength: number,
  hopLength: number,
  fftLength?: number,
  power?: number,
  center: boolean = true,
  padMode: string = "reflect",
  onesided: boolean = true,
  dither: number = 0,
  preemphasis?: number,
  melFilters?: number[][],
  melFloor: number = 1e-10,
  logMel?: string,
  reference: number = 1.0,
  minValue: number = 1e-10,
  dbRange?: number,
  removeDcOffset: boolean = false
): number[][] {
  if (fftLength === undefined) {
    fftLength = frameLength;
  }

  if (frameLength > fftLength) {
    throw new Error(`frame_length (${frameLength}) may not be larger than fft_length (${fftLength})`);
  }

  if (window.length !== frameLength) {
    throw new Error(`Length of the window (${window.length}) must equal frame_length (${frameLength})`);
  }

  if (hopLength <= 0) {
    throw new Error("hop_length must be greater than zero");
  }

  let paddedWaveform = waveform;

  // Center pad the waveform
  if (center) {
    const padLength = Math.floor(frameLength / 2);
    const newLength = waveform.length + 2 * padLength;
    paddedWaveform = new Float64Array(newLength);

    if (padMode === "reflect") {
      // Reflect padding
      for (let i = 0; i < padLength; i++) {
        paddedWaveform[i] = waveform[padLength - 1 - i] ?? 0;
      }
      paddedWaveform.set(waveform, padLength);
      for (let i = 0; i < padLength; i++) {
        paddedWaveform[padLength + waveform.length + i] = waveform[waveform.length - 1 - i] ?? 0;
      }
    } else if (padMode === "constant") {
      // Zero padding
      paddedWaveform.set(waveform, padLength);
    } else if (padMode === "edge") {
      // Edge padding
      paddedWaveform.fill(waveform[0] ?? 0, 0, padLength);
      paddedWaveform.set(waveform, padLength);
      paddedWaveform.fill(waveform[waveform.length - 1] ?? 0, padLength + waveform.length);
    }
  }

  // Split waveform into frames
  const numFrames = Math.floor(1 + (paddedWaveform.length - frameLength) / hopLength);
  const numFrequencyBins = onesided ? Math.floor(fftLength / 2) + 1 : fftLength;

  const spectrogramResult: number[][] = [];
  const buffer = new Float64Array(fftLength);

  for (let frameIdx = 0; frameIdx < numFrames; frameIdx++) {
    const timestep = frameIdx * hopLength;

    // Clear buffer
    buffer.fill(0);

    // Copy frame data
    for (let i = 0; i < frameLength; i++) {
      buffer[i] = paddedWaveform[timestep + i] ?? 0;
    }

    // Add dithering
    if (dither !== 0) {
      for (let i = 0; i < frameLength; i++) {
        buffer[i]! += dither * (Math.random() * 2 - 1); // Simple uniform noise approximation
      }
    }

    // Remove DC offset
    if (removeDcOffset) {
      const mean = buffer.slice(0, frameLength).reduce((a, b) => a + b, 0) / frameLength;
      for (let i = 0; i < frameLength; i++) {
        buffer[i]! -= mean;
      }
    }

    // Apply preemphasis
    if (preemphasis !== undefined) {
      for (let i = frameLength - 1; i > 0; i--) {
        buffer[i]! -= preemphasis * (buffer[i - 1] ?? 0);
      }
      buffer[0]! *= 1 - preemphasis;
    }

    // Apply window
    for (let i = 0; i < frameLength; i++) {
      buffer[i]! *= (window[i] ?? 1);
    }

    // Compute FFT
    const fftResult = onesided ? rfft(buffer) : fft(buffer);

    // Convert to magnitude/power
    const frameResult: number[] = [];
    for (let i = 0; i < numFrequencyBins; i++) {
      const complex = fftResult[i];
      if (!complex) continue;

      if (power === undefined) {
        // Return complex values (not implemented in this simplified version)
        throw new Error("Complex-valued spectrograms not supported in this implementation");
      } else if (power === 1.0) {
        frameResult.push(complex.magnitude());
      } else if (power === 2.0) {
        frameResult.push(complex.magnitudeSquared());
      } else {
        frameResult.push(Math.pow(complex.magnitude(), power));
      }
    }

    spectrogramResult.push(frameResult);
  }

  // Transpose to get (frequency, time) format
  const transposed: number[][] = [];
  for (let freqIdx = 0; freqIdx < numFrequencyBins; freqIdx++) {
    transposed[freqIdx] = [];
    for (let timeIdx = 0; timeIdx < numFrames; timeIdx++) {
      transposed[freqIdx]![timeIdx] = spectrogramResult[timeIdx]?.[freqIdx] ?? 0;
    }
  }

  let result = transposed;

  // Apply mel filters
  if (melFilters) {
    const melResult: number[][] = [];
    for (let melIdx = 0; melIdx < melFilters.length; melIdx++) {
      melResult[melIdx] = [];
      for (let timeIdx = 0; timeIdx < numFrames; timeIdx++) {
        let sum = 0;
        for (let freqIdx = 0; freqIdx < numFrequencyBins; freqIdx++) {
          sum += (melFilters[melIdx]?.[freqIdx] ?? 0) * (transposed[freqIdx]?.[timeIdx] ?? 0);
        }
        melResult[melIdx]![timeIdx] = Math.max(melFloor, sum);
      }
    }
    result = melResult;
  }

  // Apply log scaling
  if (power !== undefined && logMel) {
    for (let i = 0; i < result.length; i++) {
      for (let j = 0; j < (result[i]?.length ?? 0); j++) {
        if (logMel === "log") {
          result[i]![j] = Math.log(Math.max(minValue, result[i]![j]!));
        } else if (logMel === "log10") {
          result[i]![j] = Math.log10(Math.max(minValue, result[i]![j]!));
        } else if (logMel === "dB") {
          if (power === 1.0) {
            result[i]![j] = 20.0 * Math.log10(Math.max(minValue, result[i]![j]!) / reference);
          } else if (power === 2.0) {
            result[i]![j] = 10.0 * Math.log10(Math.max(minValue, result[i]![j]!) / reference);
          }
        }
      }
    }

    // Apply dB range if specified
    if (dbRange !== undefined && logMel === "dB") {
      // Find max value
      let maxVal = -Infinity;
      for (let i = 0; i < result.length; i++) {
        for (let j = 0; j < (result[i]?.length ?? 0); j++) {
          maxVal = Math.max(maxVal, result[i]![j]!);
        }
      }

      // Apply range constraint
      const minAllowed = maxVal - dbRange;
      for (let i = 0; i < result.length; i++) {
        for (let j = 0; j < (result[i]?.length ?? 0); j++) {
          result[i]![j] = Math.max(minAllowed, result[i]![j]!);
        }
      }
    }
  }

  return result;
}

/**
 * Short-Time Fourier Transform
 */
export function stft(
  signal: Float64Array,
  window: Float64Array,
  frameLength: number,
  hopLength: number,
  center: boolean = true
): Complex[][] {
  // Pad signal if centering
  let paddedSignal = signal;
  if (center) {
    const padLength = Math.floor(frameLength / 2);
    paddedSignal = new Float64Array(signal.length + 2 * padLength);

    // Reflect padding
    for (let i = 0; i < padLength; i++) {
      paddedSignal[i] = signal[padLength - i - 1] ?? 0;
      paddedSignal[signal.length + padLength + i] = signal[signal.length - i - 1] ?? 0;
    }

    for (let i = 0; i < signal.length; i++) {
      paddedSignal[padLength + i] = signal[i] ?? 0;
    }
  }

  const numFrames = Math.floor((paddedSignal.length - frameLength) / hopLength) + 1;
  const result: Complex[][] = [];

  for (let frame = 0; frame < numFrames; frame++) {
    const start = frame * hopLength;
    const frameData = new Float64Array(frameLength);

    for (let i = 0; i < frameLength; i++) {
      if (start + i < paddedSignal.length) {
        frameData[i] = (paddedSignal[start + i] ?? 0) * (window[i] ?? 1);
      }
    }

    const complexFrame = Array.from(frameData, x => new Complex(x, 0));
    const fftResult = fft(complexFrame);
    result.push(fftResult);
  }

  return result;
}