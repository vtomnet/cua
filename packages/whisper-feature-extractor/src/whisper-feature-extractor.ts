import { SequenceFeatureExtractor } from "./sequence-feature-extractor";
import { BatchFeature } from "./batch-feature";
import { melFilterBank, windowFunction, spectrogram } from "./audio-utils";
import type {
  WhisperFeatureExtractorConfig,
  FeatureExtractionCallOptions,
  TensorType
} from "./types";

/**
 * Whisper feature extractor for extracting mel-filter bank features from raw speech
 */
export class WhisperFeatureExtractor extends SequenceFeatureExtractor {
  private nFft: number;
  private hopLength: number;
  private chunkLength: number;
  private nSamples: number;
  private nbMaxFrames: number;
  private dither: number;
  private melFilters: number[][];

  constructor(config: WhisperFeatureExtractorConfig = {}) {
    const {
      featureSize = 80,
      samplingRate = 16000,
      hopLength = 160,
      chunkLength = 30,
      nFft = 400,
      paddingValue = 0.0,
      dither = 0.0,
      returnAttentionMask = false
    } = config;

    super(featureSize, samplingRate, paddingValue, {
      returnAttentionMask
    });

    this.nFft = nFft;
    this.hopLength = hopLength;
    this.chunkLength = chunkLength;
    this.nSamples = chunkLength * samplingRate;
    this.nbMaxFrames = Math.floor(this.nSamples / hopLength);
    this.dither = dither;

    // Create mel filter bank
    this.melFilters = melFilterBank(
      1 + Math.floor(nFft / 2), // numFrequencyBins
      featureSize,              // numMelFilters
      0.0,                      // minFrequency
      8000.0,                   // maxFrequency
      samplingRate,             // samplingRate
      "slaney",                 // norm
      "slaney",                 // melScale
      false                     // triangularizeInMelSpace
    );
  }




  /**
   * Normalize input to consistent Float64Array[] format
   */
  private normalizeInput(rawSpeech: Float64Array | Float64Array[] | number[] | number[][]): Float64Array[] {
    if (rawSpeech instanceof Float64Array) {
      return [rawSpeech];
    }

    if (!Array.isArray(rawSpeech)) {
      throw new Error("Invalid input format for raw_speech");
    }

    // Empty array
    if (rawSpeech.length === 0) {
      return [];
    }

    // Array of arrays (number[][])
    if (Array.isArray(rawSpeech[0])) {
      return (rawSpeech as number[][]).map(speech => new Float64Array(speech));
    }

    // Array of Float64Arrays
    if (rawSpeech[0] instanceof Float64Array) {
      return rawSpeech as Float64Array[];
    }

    // Single array of numbers
    return [new Float64Array(rawSpeech as number[])];
  }

  /**
   * Extract features from raw speech using NumPy-style processing
   */
  private extractFbankFeatures(waveformBatch: Float64Array[]): number[][][] {
    const logSpecBatch: number[][][] = [];

    for (const waveform of waveformBatch) {
      // Create Hann window
      const window = windowFunction(this.nFft, "hann");

      // Compute spectrogram
      const logSpec = spectrogram(
        waveform,
        window,
        this.nFft,           // frame_length
        this.hopLength,      // hop_length
        this.nFft,          // fft_length
        2.0,                // power (power spectrogram)
        true,               // center
        "reflect",          // pad_mode
        true,               // onesided
        this.dither,        // dither
        undefined,          // preemphasis
        this.melFilters,    // mel_filters
        1e-10,              // mel_floor
        "log10"             // log_mel
      );

      // Remove last frame (to match PyTorch STFT behavior)
      const trimmedLogSpec = logSpec.map(freqBin => freqBin.slice(0, -1));

      // Apply dynamic range compression: max(log_spec, log_spec.max() - 8.0)
      let maxVal = -Infinity;
      for (const freqBin of trimmedLogSpec) {
        for (const val of freqBin) {
          maxVal = Math.max(maxVal, val);
        }
      }

      const clampedLogSpec = trimmedLogSpec.map(freqBin =>
        freqBin.map(val => Math.max(val, maxVal - 8.0))
      );

      // Normalize: (log_spec + 4.0) / 4.0
      const normalizedLogSpec = clampedLogSpec.map(freqBin =>
        freqBin.map(val => (val + 4.0) / 4.0)
      );

      logSpecBatch.push(normalizedLogSpec);
    }

    return logSpecBatch;
  }

  /**
   * Normalize features using zero-mean unit-variance normalization
   */
  private normalizeFeatures(
    inputFeatures: number[][],
    attentionMask?: number[]
  ): number[][] {
    // Reshape inputFeatures from number[][] to number[][][] for normalization
    // inputFeatures is [numFeatures][numTimeSteps], we need [1][numFeatures][numTimeSteps]
    const reshapedForNorm: number[][][] = [inputFeatures];

    const normalized = WhisperFeatureExtractor.zeroMeanUnitVarNorm(
      reshapedForNorm,
      attentionMask ? [attentionMask] : undefined,
      this.paddingValue
    );

    return normalized[0] ?? inputFeatures;
  }

  /**
   * Zero-mean unit-variance normalization
   */
  private static zeroMeanUnitVarNorm(
    inputValues: number[][][],
    attentionMask?: number[][],
    paddingValue: number = 0.0
  ): number[][][] {
    const normedInputValues: number[][][] = [];

    for (let i = 0; i < inputValues.length; i++) {
      const vector = inputValues[i];
      if (!vector || !vector[0]) continue;

      let length = vector[0].length;

      if (attentionMask && attentionMask[i]) {
        length = attentionMask[i]!.reduce((sum, mask) => sum + mask, 0);
      }

      // Flatten the first `length` time steps for computing statistics
      const flattenedVector: number[] = [];
      for (let t = 0; t < length; t++) {
        for (let f = 0; f < vector.length; f++) {
          const value = vector[f]?.[t];
          if (value !== undefined) {
            flattenedVector.push(value);
          }
        }
      }

      // Compute mean and variance
      const mean = flattenedVector.reduce((sum, val) => sum + val, 0) / flattenedVector.length;
      const variance = flattenedVector.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / flattenedVector.length;
      const std = Math.sqrt(variance + 1e-7);

      // Normalize
      const normedSlice = vector.map(freqBin =>
        freqBin.map((val, timeIdx) => {
          if (timeIdx < length) {
            return (val - mean) / std;
          } else {
            return paddingValue;
          }
        })
      );

      normedInputValues.push(normedSlice);
    }

    return normedInputValues;
  }

  /**
   * Extract features from raw speech audio
   */
  extractFeatures(
    rawSpeech: Float64Array | Float64Array[] | number[] | number[][],
    options: FeatureExtractionCallOptions = {}
  ): BatchFeature {
    const {
      truncation = true,
      padToMultipleOf,
      returnTensors,
      returnAttentionMask,
      padding = "maxLength",
      maxLength,
      samplingRate,
      doNormalize,
      device = "cpu",
      returnTokenTimestamps
    } = options;

    // Validate sampling rate
    if (samplingRate !== undefined && samplingRate !== this.samplingRate) {
      throw new Error(
        `The model corresponding to this feature extractor was trained using a sampling rate of ${this.samplingRate}. ` +
        `Please make sure that the provided raw_speech input was sampled with ${this.samplingRate} and not ${samplingRate}.`
      );
    }

    if (device !== "cpu") {
      throw new Error(
        `Got device \`${device}\` for feature extraction, but this TypeScript implementation only supports CPU processing.`
      );
    }

    // Convert input to consistent format
    const processedSpeech = this.normalizeInput(rawSpeech);

    // Validate mono channel
    // (This implementation assumes input is already mono)

    // Convert to batch format for padding
    const batchedSpeech = new BatchFeature({
      input_features: processedSpeech.map(speech => Array.from(speech))
    });

    // Pad inputs
    const paddedInputs = this.pad(batchedSpeech, {
      padding,
      maxLength: maxLength || this.nSamples,
      truncation,
      padToMultipleOf,
      returnAttentionMask: returnAttentionMask || doNormalize
    });

    // Zero-mean unit-variance normalization
    let inputFeatures = paddedInputs.get("input_features") as number[][] | undefined;
    if (!inputFeatures) {
      throw new Error("input_features not found in padded inputs");
    }

    if (doNormalize) {
      inputFeatures = this.normalizeFeatures(inputFeatures, paddedInputs.get("attention_mask"));
    }

    // Convert to Float64Array for feature extraction
    const inputFeaturesFloat64 = inputFeatures.map(seq => new Float64Array(seq));

    // Extract mel-filterbank features
    const extractedFeatures = this.extractFbankFeatures(inputFeaturesFloat64);

    // Set the processed features
    paddedInputs.set("input_features", extractedFeatures);

    // Handle attention mask rescaling
    if (returnAttentionMask && paddedInputs.has("attention_mask")) {
      this.rescaleAttentionMask(paddedInputs);
    }

    // Handle deprecated returnTokenTimestamps
    if (returnTokenTimestamps !== undefined) {
      console.warn(
        "`returnTokenTimestamps` is deprecated. Use `returnAttentionMask` instead, " +
        "as the number of frames can be inferred from it."
      );

      const numFrames = processedSpeech.map(speech => Math.floor(speech.length / this.hopLength));
      paddedInputs.set("num_frames", numFrames);
    }

    // Convert to tensors if requested
    if (returnTensors) {
      paddedInputs.convertToTensors(returnTensors);
    }

    return paddedInputs;
  }

  /**
   * Rescale attention mask to match feature frames
   */
  private rescaleAttentionMask(paddedInputs: BatchFeature): void {
    const attentionMask = paddedInputs.get("attention_mask") as number[][];
    const rescaledAttentionMask = attentionMask.map(mask =>
      mask.filter((_, idx) => idx % this.hopLength === 0)
    );

    // Trim if needed to match feature frames
    const trimmedAttentionMask = rescaledAttentionMask.map(mask => {
      const expectedFrames = Math.floor((mask.length * this.hopLength) / this.hopLength);
      return mask.length > expectedFrames ? mask.slice(0, expectedFrames) : mask;
    });

    paddedInputs.set("attention_mask", trimmedAttentionMask);
  }

  /**
   * Get configuration object
   */
  getConfig(): WhisperFeatureExtractorConfig {
    return {
      featureSize: this.featureSize,
      samplingRate: this.samplingRate,
      hopLength: this.hopLength,
      chunkLength: this.chunkLength,
      nFft: this.nFft,
      paddingValue: this.paddingValue,
      dither: this.dither,
      returnAttentionMask: this.returnAttentionMask
    };
  }

  /**
   * Create from configuration
   */
  static fromConfig(config: WhisperFeatureExtractorConfig): WhisperFeatureExtractor {
    return new WhisperFeatureExtractor(config);
  }
}
