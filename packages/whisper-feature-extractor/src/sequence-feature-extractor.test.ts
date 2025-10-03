import { test, expect, describe, beforeEach } from "bun:test";
import { SequenceFeatureExtractor, PaddingStrategy } from "./sequence-feature-extractor";
import { BatchFeature } from "./batch-feature";

// Concrete implementation for testing
class TestSequenceFeatureExtractor extends SequenceFeatureExtractor {
  constructor(featureSize = 1, samplingRate = 16000, paddingValue = 0.0) {
    super(featureSize, samplingRate, paddingValue);
  }
}

describe("SequenceFeatureExtractor", () => {
  let extractor: TestSequenceFeatureExtractor;

  beforeEach(() => {
    extractor = new TestSequenceFeatureExtractor();
  });

  describe("PaddingStrategy", () => {
    test("should have correct enum values", () => {
      expect(PaddingStrategy.DO_NOT_PAD as string).toBe("doNotPad");
      expect(PaddingStrategy.LONGEST as string).toBe("longest");
      expect(PaddingStrategy.MAX_LENGTH as string).toBe("maxLength");
    });
  });

  describe("Constructor", () => {
    test("should initialize with default values", () => {
      const testExtractor = new TestSequenceFeatureExtractor();
      expect(testExtractor["featureSize"]).toBe(1);
      expect(testExtractor["samplingRate"]).toBe(16000);
      expect(testExtractor["paddingValue"]).toBe(0.0);
      expect(testExtractor["paddingSide"]).toBe("right");
      expect(testExtractor["returnAttentionMask"]).toBe(true);
    });

    test("should initialize with custom options", () => {
      const testExtractor = new TestSequenceFeatureExtractor(20, 22050, -1.0);
      testExtractor["paddingSide"] = "left";
      testExtractor["returnAttentionMask"] = false;

      expect(testExtractor["featureSize"]).toBe(20);
      expect(testExtractor["samplingRate"]).toBe(22050);
      expect(testExtractor["paddingValue"]).toBe(-1.0);
    });
  });

  describe("Padding", () => {
    test("should pad to maxLength", () => {
      const features = new BatchFeature({
        input_features: [
          [1, 2, 3],
          [4, 5]
        ]
      });

      const result = extractor.pad(features, {
        padding: "maxLength",
        maxLength: 5
      });

      const inputFeatures = result.get("input_features");
      expect(inputFeatures[0]).toEqual([1, 2, 3, 0, 0]);
      expect(inputFeatures[1]).toEqual([4, 5, 0, 0, 0]);
    });

    test("should pad to longest sequence", () => {
      const features = new BatchFeature({
        input_features: [
          [1, 2, 3, 4, 5],
          [6, 7, 8],
          [9, 10]
        ]
      });

      const result = extractor.pad(features, {
        padding: "longest"
      });

      const inputFeatures = result.get("input_features");
      expect(inputFeatures[0]).toEqual([1, 2, 3, 4, 5]);
      expect(inputFeatures[1]).toEqual([6, 7, 8, 0, 0]);
      expect(inputFeatures[2]).toEqual([9, 10, 0, 0, 0]);
    });

    test("should not pad when padding is false", () => {
      const features = new BatchFeature({
        input_features: [
          [1, 2, 3],
          [4, 5]
        ]
      });

      const result = extractor.pad(features, {
        padding: false
      });

      const inputFeatures = result.get("input_features");
      expect(inputFeatures[0]).toEqual([1, 2, 3]);
      expect(inputFeatures[1]).toEqual([4, 5]);
    });

    test("should pad with custom padding value", () => {
      const customExtractor = new TestSequenceFeatureExtractor(1, 16000, -999);
      const features = new BatchFeature({
        input_features: [
          [1, 2, 3]
        ]
      });

      const result = customExtractor.pad(features, {
        padding: "maxLength",
        maxLength: 5
      });

      const inputFeatures = result.get("input_features");
      expect(inputFeatures[0]).toEqual([1, 2, 3, -999, -999]);
    });

    test("should pad to multiple of specified value", () => {
      const features = new BatchFeature({
        input_features: [
          [1, 2, 3, 4, 5, 6, 7] // length 7
        ]
      });

      const result = extractor.pad(features, {
        padding: "maxLength",
        maxLength: 7,
        padToMultipleOf: 4 // Should pad to 8
      });

      const inputFeatures = result.get("input_features");
      expect(inputFeatures[0].length).toBe(8);
      expect(inputFeatures[0]).toEqual([1, 2, 3, 4, 5, 6, 7, 0]);
    });

    test("should create attention mask", () => {
      const features = new BatchFeature({
        input_features: [
          [1, 2, 3],
          [4, 5]
        ]
      });

      const result = extractor.pad(features, {
        padding: "longest",
        returnAttentionMask: true
      });

      expect(result.has("attention_mask")).toBe(true);
      const attentionMask = result.get("attention_mask");
      expect(attentionMask[0]).toEqual([1, 1, 1]);
      expect(attentionMask[1]).toEqual([1, 1, 0]);
    });

    test("should handle left padding", () => {
      const leftExtractor = new TestSequenceFeatureExtractor();
      leftExtractor["paddingSide"] = "left";

      const features = new BatchFeature({
        input_features: [
          [1, 2, 3]
        ]
      });

      const result = leftExtractor.pad(features, {
        padding: "maxLength",
        maxLength: 5,
        returnAttentionMask: true
      });

      const inputFeatures = result.get("input_features");
      const attentionMask = result.get("attention_mask");

      expect(inputFeatures[0]).toEqual([0, 0, 1, 2, 3]);
      expect(attentionMask[0]).toEqual([0, 0, 1, 1, 1]);
    });

    test("should handle 2D features", () => {
      const extractor2D = new TestSequenceFeatureExtractor(3); // feature_size > 1
      const features = new BatchFeature({
        input_features: [
          [[1, 2, 3], [4, 5, 6]] // 2 time steps, 3 features each
        ]
      });

      const result = extractor2D.pad(features, {
        padding: "maxLength",
        maxLength: 4
      });

      const inputFeatures = result.get("input_features");
      expect(inputFeatures[0]).toEqual([
        [1, 2, 3],
        [4, 5, 6],
        [0, 0, 0],
        [0, 0, 0]
      ]);
    });

    test("should handle array of BatchFeatures", () => {
      const features1 = new BatchFeature({ input_features: [[1, 2, 3]] });
      const features2 = new BatchFeature({ input_features: [[4, 5]] });

      const result = extractor.pad([features1, features2], {
        padding: "longest"
      });

      const inputFeatures = result.get("input_features");
      expect(inputFeatures[0]).toEqual([1, 2, 3]);
      expect(inputFeatures[1]).toEqual([4, 5, 0]);
    });

    test("should handle empty input", () => {
      const features = new BatchFeature({
        input_features: []
      });

      const result = extractor.pad(features, {
        returnAttentionMask: true
      });

      expect(result.get("input_features")).toEqual([]);
      expect(result.get("attention_mask")).toEqual([]);
    });
  });

  describe("Truncation", () => {
    test("should truncate to maxLength", () => {
      const features = new BatchFeature({
        input_features: [
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ]
      });

      const result = extractor.pad(features, {
        truncation: true,
        maxLength: 5,
        padding: false
      });

      const inputFeatures = result.get("input_features");
      expect(inputFeatures[0]).toEqual([1, 2, 3, 4, 5]);
    });

    test("should truncate attention mask", () => {
      const features = new BatchFeature({
        input_features: [
          [1, 2, 3, 4, 5, 6, 7, 8]
        ],
        attention_mask: [
          [1, 1, 1, 1, 1, 1, 1, 1]
        ]
      });

      const result = extractor.pad(features, {
        truncation: true,
        maxLength: 5,
        padding: false
      });

      const inputFeatures = result.get("input_features");
      const attentionMask = result.get("attention_mask");

      expect(inputFeatures[0]).toEqual([1, 2, 3, 4, 5]);
      expect(attentionMask[0]).toEqual([1, 1, 1, 1, 1]);
    });

    test("should adjust for padToMultipleOf in truncation", () => {
      const features = new BatchFeature({
        input_features: [
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ]
      });

      const result = extractor.pad(features, {
        truncation: true,
        maxLength: 7,
        padToMultipleOf: 4, // Should truncate to 8 (next multiple of 4)
        padding: false
      });

      const inputFeatures = result.get("input_features");
      expect(inputFeatures[0]).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    });

    test("should not truncate if sequence is shorter", () => {
      const features = new BatchFeature({
        input_features: [
          [1, 2, 3]
        ]
      });

      const result = extractor.pad(features, {
        truncation: true,
        maxLength: 5,
        padding: false
      });

      const inputFeatures = result.get("input_features");
      expect(inputFeatures[0]).toEqual([1, 2, 3]);
    });
  });

  describe("Error Handling", () => {
    test("should throw when main input is missing", () => {
      const features = new BatchFeature({
        some_other_key: [[1, 2, 3]]
      });

      expect(() => {
        extractor.pad(features);
      }).toThrow("You should supply an instance of BatchFeature that includes input_features");
    });

    test("should throw when maxLength padding without maxLength", () => {
      const features = new BatchFeature({
        input_features: [[1, 2, 3]]
      });

      expect(() => {
        extractor.pad(features, {
          padding: "maxLength"
          // maxLength is missing
        });
      }).toThrow("When setting padding=maxLength, make sure that maxLength is defined");
    });

    test("should throw when truncation without maxLength", () => {
      const features = new BatchFeature({
        input_features: [[1, 2, 3, 4, 5, 6]]
      });

      expect(() => {
        extractor.pad(features, {
          truncation: true
          // maxLength is missing
        });
      }).toThrow("When setting truncation=true, make sure that maxLength is defined");
    });

    test("should throw for unknown padding strategy", () => {
      const features = new BatchFeature({
        input_features: [[1, 2, 3]]
      });

      expect(() => {
        extractor.pad(features, {
          padding: "unknown_strategy" as any
        });
      }).toThrow("Unknown padding strategy");
    });

    test("should throw for invalid padding side", () => {
      const invalidExtractor = new TestSequenceFeatureExtractor();
      invalidExtractor["paddingSide"] = "invalid";

      const features = new BatchFeature({
        input_features: [[1, 2, 3]]
      });

      expect(() => {
        invalidExtractor.pad(features, {
          padding: "maxLength",
          maxLength: 5
        });
      }).toThrow("Invalid padding strategy");
    });

    test("should throw when padding without padding value", () => {
      const noPaddingExtractor = new TestSequenceFeatureExtractor();
      noPaddingExtractor["paddingValue"] = undefined as any;

      const features = new BatchFeature({
        input_features: [[1, 2, 3]]
      });

      expect(() => {
        noPaddingExtractor.pad(features, {
          padding: "maxLength",
          maxLength: 5
        });
      }).toThrow("Asking to pad but the feature_extractor does not have a padding value");
    });
  });

  describe("Complex Scenarios", () => {
    test("should handle padding and truncation together", () => {
      const features = new BatchFeature({
        input_features: [
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], // Will be truncated
          [1, 2, 3] // Will be padded
        ]
      });

      const result = extractor.pad(features, {
        padding: "maxLength",
        maxLength: 5,
        truncation: true,
        returnAttentionMask: true
      });

      const inputFeatures = result.get("input_features");
      const attentionMask = result.get("attention_mask");

      expect(inputFeatures[0]).toEqual([1, 2, 3, 4, 5]);
      expect(inputFeatures[1]).toEqual([1, 2, 3, 0, 0]);
      expect(attentionMask[0]).toEqual([1, 1, 1, 1, 1]);
      expect(attentionMask[1]).toEqual([1, 1, 1, 0, 0]);
    });

    test("should handle multiple features with mixed operations", () => {
      const features = new BatchFeature({
        input_features: [
          [1, 2, 3, 4, 5, 6, 7, 8],
          [9, 10, 11]
        ],
        extra_features: [
          [100, 200, 300, 400, 500, 600, 700, 800],
          [900, 1000, 1100]
        ]
      });

      const result = extractor.pad(features, {
        padding: "maxLength",
        maxLength: 5,
        truncation: true,
        returnAttentionMask: true
      });

      const inputFeatures = result.get("input_features");
      const extraFeatures = result.get("extra_features");
      const attentionMask = result.get("attention_mask");

      expect(inputFeatures[0]).toEqual([1, 2, 3, 4, 5]);
      expect(inputFeatures[1]).toEqual([9, 10, 11, 0, 0]);
      expect(extraFeatures[0]).toEqual([100, 200, 300, 400, 500]);
      expect(extraFeatures[1]).toEqual([900, 1000, 1100, 0, 0]);
      expect(attentionMask[0]).toEqual([1, 1, 1, 1, 1]);
      expect(attentionMask[1]).toEqual([1, 1, 1, 0, 0]);
    });
  });
});