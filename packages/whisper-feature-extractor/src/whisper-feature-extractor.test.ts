import { test, expect, describe, beforeEach } from "bun:test";
import { WhisperFeatureExtractor } from "./whisper-feature-extractor";
import { BatchFeature } from "./batch-feature";

describe("WhisperFeatureExtractor", () => {
  let extractor: WhisperFeatureExtractor;

  beforeEach(() => {
    extractor = new WhisperFeatureExtractor();
  });

  test("should initialize with default config", () => {
    const config = extractor.getConfig();
    expect(config.featureSize).toBe(80);
    expect(config.samplingRate).toBe(16000);
    expect(config.hopLength).toBe(160);
    expect(config.chunkLength).toBe(30);
    expect(config.nFft).toBe(400);
    expect(config.paddingValue).toBe(0.0);
    expect(config.dither).toBe(0.0);
    expect(config.returnAttentionMask).toBe(false);
  });

  test("should initialize with custom config", () => {
    const customExtractor = new WhisperFeatureExtractor({
      featureSize: 64,
      samplingRate: 22050,
      hopLength: 256,
      chunkLength: 20,
      nFft: 512,
      paddingValue: -1.0,
      dither: 0.1,
      returnAttentionMask: true
    });

    const config = customExtractor.getConfig();
    expect(config.featureSize).toBe(64);
    expect(config.samplingRate).toBe(22050);
    expect(config.hopLength).toBe(256);
    expect(config.chunkLength).toBe(20);
    expect(config.nFft).toBe(512);
    expect(config.paddingValue).toBe(-1.0);
    expect(config.dither).toBe(0.1);
    expect(config.returnAttentionMask).toBe(true);
  });

  test("should create from config", () => {
    const config = {
      featureSize: 64,
      samplingRate: 22050,
      hopLength: 256
    };
    const configExtractor = WhisperFeatureExtractor.fromConfig(config);
    const extractorConfig = configExtractor.getConfig();
    expect(extractorConfig.featureSize).toBe(64);
    expect(extractorConfig.samplingRate).toBe(22050);
    expect(extractorConfig.hopLength).toBe(256);
  });

  test("should process single audio array", () => {
    // Create a simple sine wave test signal
    const sampleRate = 16000;
    const duration = 1; // 1 second
    const frequency = 440; // A4 note
    const samples = new Float64Array(sampleRate * duration);

    for (let i = 0; i < samples.length; i++) {
      samples[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
    }

    const result = extractor.extractFeatures(samples);

    expect(result).toBeInstanceOf(BatchFeature);
    expect(result.has("input_features")).toBe(true);

    const features = result.get("input_features");
    expect(Array.isArray(features)).toBe(true);
    expect(features.length).toBe(1); // Single batch
    expect(features[0].length).toBe(80); // 80 mel filters
    expect(features[0][0].length).toBeGreaterThan(0); // Time frames
  });

  test("should process batch of audio arrays", () => {
    const sampleRate = 16000;
    const duration = 0.5; // 0.5 seconds
    const batchSize = 3;
    const batch: Float64Array[] = [];

    for (let b = 0; b < batchSize; b++) {
      const samples = new Float64Array(sampleRate * duration);
      const frequency = 440 + b * 110; // Different frequencies

      for (let i = 0; i < samples.length; i++) {
        samples[i] = Math.sin(2 * Math.PI * frequency * i / sampleRate);
      }
      batch.push(samples);
    }

    const result = extractor.extractFeatures(batch);

    expect(result).toBeInstanceOf(BatchFeature);
    expect(result.has("input_features")).toBe(true);

    const features = result.get("input_features");
    expect(Array.isArray(features)).toBe(true);
    expect(features.length).toBe(batchSize);

    for (let b = 0; b < batchSize; b++) {
      expect(features[b].length).toBe(80); // 80 mel filters
      expect(features[b][0].length).toBeGreaterThan(0); // Time frames
    }
  });

  test("should handle padding with maxLength", () => {
    const shortSamples = new Float64Array(8000); // 0.5 seconds
    shortSamples.fill(0.1);

    const result = extractor.extractFeatures(shortSamples, {
      padding: "maxLength",
      maxLength: 16000 // 1 second
    });

    const features = result.get("input_features");
    expect(features[0][0].length).toBeGreaterThan(0);
  });

  test("should handle truncation", () => {
    const longSamples = new Float64Array(32000); // 2 seconds
    longSamples.fill(0.1);

    const result = extractor.extractFeatures(longSamples, {
      truncation: true,
      maxLength: 16000 // 1 second
    });

    const features = result.get("input_features");
    expect(features[0][0].length).toBeGreaterThan(0);
  });

  test("should return attention mask when requested", () => {
    const samples = new Float64Array(16000); // 1 second
    samples.fill(0.1);

    const result = extractor.extractFeatures(samples, {
      returnAttentionMask: true
    });

    expect(result.has("attention_mask")).toBe(true);
    const attentionMask = result.get("attention_mask");
    expect(Array.isArray(attentionMask)).toBe(true);
    expect(attentionMask.length).toBe(1); // Single batch
    expect(attentionMask[0].length).toBeGreaterThan(0);
  });

  test("should apply normalization when requested", () => {
    const samples = new Float64Array(16000);
    // Create a signal with non-zero mean
    for (let i = 0; i < samples.length; i++) {
      samples[i] = Math.sin(2 * Math.PI * 440 * i / 16000) + 0.5;
    }

    const resultWithNorm = extractor.extractFeatures(samples, {
      doNormalize: true,
      returnAttentionMask: true
    });

    const resultWithoutNorm = extractor.extractFeatures(samples, {
      doNormalize: false
    });

    const featuresWithNorm = resultWithNorm.get("input_features");
    const featuresWithoutNorm = resultWithoutNorm.get("input_features");

    // Features should be different when normalization is applied
    expect(featuresWithNorm[0][0][0]).not.toBe(featuresWithoutNorm[0][0][0]);
  });

  test("should validate sampling rate", () => {
    const samples = new Float64Array(16000);
    samples.fill(0.1);

    expect(() => {
      extractor.extractFeatures(samples, {
        samplingRate: 22050 // Different from extractor's 16000
      });
    }).toThrow();
  });

  test("should reject non-CPU device", () => {
    const samples = new Float64Array(16000);
    samples.fill(0.1);

    expect(() => {
      extractor.extractFeatures(samples, {
        device: "cuda"
      });
    }).toThrow();
  });

  test("should handle returnTokenTimestamps deprecation warning", () => {
    const samples = new Float64Array(16000);
    samples.fill(0.1);

    // Capture console.warn
    const originalWarn = console.warn;
    let warnMessage = "";
    console.warn = (message: string) => {
      warnMessage = message;
    };

    const result = extractor.extractFeatures(samples, {
      returnTokenTimestamps: true
    });

    // Restore console.warn
    console.warn = originalWarn;

    expect(warnMessage).toContain("returnTokenTimestamps");
    expect(result.has("num_frames")).toBe(true);
  });

  test("should handle different input formats", () => {
    const sampleData = Array.from({ length: 8000 }, (_, i) =>
      Math.sin(2 * Math.PI * 440 * i / 16000)
    );

    // Test number array
    const result1 = extractor.extractFeatures(sampleData);
    expect(result1.has("input_features")).toBe(true);

    // Test Float64Array
    const result2 = extractor.extractFeatures(new Float64Array(sampleData));
    expect(result2.has("input_features")).toBe(true);

    // Test array of number arrays
    const result3 = extractor.extractFeatures([sampleData, sampleData]);
    expect(result3.has("input_features")).toBe(true);
    expect(result3.get("input_features").length).toBe(2);
  });

  test("should reject invalid input format", () => {
    expect(() => {
      extractor.extractFeatures("invalid" as any);
    }).toThrow("Invalid input format");
  });

  test("should handle empty input gracefully", () => {
    const result = extractor.extractFeatures(new Float64Array(0));
    expect(result).toBeInstanceOf(BatchFeature);
  });

  test("should convert to tensors when requested", () => {
    const samples = new Float64Array(8000);
    samples.fill(0.1);

    const result = extractor.extractFeatures(samples, {
      returnTensors: "np"
    });

    expect(result.has("input_features")).toBe(true);
    const features = result.get("input_features");
    expect(Array.isArray(features)).toBe(true);
  });

  test("should reject PyTorch tensor conversion", () => {
    const samples = new Float64Array(8000);
    samples.fill(0.1);

    expect(() => {
      extractor.extractFeatures(samples, {
        returnTensors: "pt"
      });
    }).toThrow("PyTorch tensor conversion not implemented");
  });
});