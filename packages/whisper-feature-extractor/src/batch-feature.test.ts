import { test, expect, describe } from "bun:test";
import { BatchFeature } from "./batch-feature";

describe("BatchFeature", () => {
  test("should initialize empty BatchFeature", () => {
    const batch = new BatchFeature();
    expect(batch.keys()).toBeDefined();
    expect(Array.from(batch.keys())).toEqual([]);
  });

  test("should initialize with data", () => {
    const data = {
      input_features: [[1, 2, 3], [4, 5, 6]],
      attention_mask: [1, 1, 0]
    };

    const batch = new BatchFeature(data);
    expect(batch.has("input_features")).toBe(true);
    expect(batch.has("attention_mask")).toBe(true);
    expect(batch.get("input_features")).toEqual([[1, 2, 3], [4, 5, 6]]);
    expect(batch.get("attention_mask")).toEqual([1, 1, 0]);
  });

  test("should get and set values", () => {
    const batch = new BatchFeature();

    batch.set("test_key", [1, 2, 3]);
    expect(batch.has("test_key")).toBe(true);
    expect(batch.get("test_key")).toEqual([1, 2, 3]);

    batch.set("test_key", [4, 5, 6]);
    expect(batch.get("test_key")).toEqual([4, 5, 6]);
  });

  test("should iterate over keys", () => {
    const data = {
      input_features: [[1, 2, 3]],
      attention_mask: [1, 1, 0],
      labels: [0, 1, 2]
    };

    const batch = new BatchFeature(data);
    const keys = Array.from(batch.keys());

    expect(keys).toContain("input_features");
    expect(keys).toContain("attention_mask");
    expect(keys).toContain("labels");
    expect(keys.length).toBe(3);
  });

  test("should iterate over values", () => {
    const data = {
      input_features: [[1, 2, 3]],
      attention_mask: [1, 1, 0]
    };

    const batch = new BatchFeature(data);
    const values = Array.from(batch.values());

    expect(values).toEqual([[[1, 2, 3]], [1, 1, 0]]);
    expect(values.length).toBe(2);
  });

  test("should iterate over entries", () => {
    const data = {
      input_features: [[1, 2, 3]],
      attention_mask: [1, 1, 0]
    };

    const batch = new BatchFeature(data);
    const entries = Array.from(batch.entries());

    expect(entries).toContainEqual(["input_features", [[1, 2, 3]]]);
    expect(entries).toContainEqual(["attention_mask", [1, 1, 0]]);
    expect(entries.length).toBe(2);
  });

  test("should convert to object", () => {
    const data = {
      input_features: [[1, 2, 3], [4, 5, 6]],
      attention_mask: [1, 1, 0],
      labels: [0, 1, 2]
    };

    const batch = new BatchFeature(data);
    const obj = batch.toObject();

    expect(obj).toEqual(data);
    expect(obj.input_features).toEqual([[1, 2, 3], [4, 5, 6]]);
    expect(obj.attention_mask).toEqual([1, 1, 0]);
    expect(obj.labels).toEqual([0, 1, 2]);
  });

  test("should convert to NumPy-like tensors", () => {
    const data = {
      input_features: [["1", "2", "3"], ["4", "5", "6"]],
      attention_mask: ["1", "1", "0"]
    };

    const batch = new BatchFeature(data);
    batch.convertToTensors("np");

    const inputFeatures = batch.get("input_features");
    const attentionMask = batch.get("attention_mask");

    // Should convert strings to numbers
    expect(inputFeatures).toEqual([[1, 2, 3], [4, 5, 6]]);
    expect(attentionMask).toEqual([1, 1, 0]);
  });

  test("should handle nested arrays in tensor conversion", () => {
    const data = {
      features: [[["1.5", "2.5"], ["3.5", "4.5"]], [["5.5", "6.5"], ["7.5", "8.5"]]]
    };

    const batch = new BatchFeature(data);
    batch.convertToTensors("np");

    const features = batch.get("features");
    expect(features).toEqual([[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]]);
  });

  test("should reject PyTorch tensor conversion", () => {
    const batch = new BatchFeature({ input_features: [[1, 2, 3]] });

    expect(() => {
      batch.convertToTensors("pt");
    }).toThrow("PyTorch tensor conversion not implemented");
  });

  test("should initialize with tensor type", () => {
    const data = {
      input_features: [["1", "2", "3"]],
      attention_mask: ["1", "0", "1"]
    };

    const batch = new BatchFeature(data, "np");

    // Should automatically convert to numeric arrays
    expect(batch.get("input_features")).toEqual([[1, 2, 3]]);
    expect(batch.get("attention_mask")).toEqual([1, 0, 1]);
  });

  test("should handle mixed data types", () => {
    const data = {
      numbers: [1, 2, 3],
      strings: ["a", "b", "c"],
      nested: [[1, 2], [3, 4]],
      mixed: ["1", 2, "3.5"]
    };

    const batch = new BatchFeature(data);
    batch.convertToTensors("np");

    expect(batch.get("numbers")).toEqual([1, 2, 3]);
    expect(batch.get("strings")).toEqual([NaN, NaN, NaN]); // Non-numeric strings become NaN
    expect(batch.get("nested")).toEqual([[1, 2], [3, 4]]);
    expect(batch.get("mixed")).toEqual([1, 2, 3.5]);
  });

  test("should handle empty arrays", () => {
    const data = {
      empty_array: [],
      nested_empty: [[], []],
      mixed_empty: [[], [1, 2, 3]]
    };

    const batch = new BatchFeature(data);
    batch.convertToTensors("np");

    expect(batch.get("empty_array")).toEqual([]);
    expect(batch.get("nested_empty")).toEqual([[], []]);
    expect(batch.get("mixed_empty")).toEqual([[], [1, 2, 3]]);
  });

  test("should preserve non-array values", () => {
    const data = {
      single_number: 42,
      single_string: "hello",
      boolean_value: true,
      null_value: null
    };

    const batch = new BatchFeature(data);

    expect(batch.get("single_number")).toBe(42);
    expect(batch.get("single_string")).toBe("hello");
    expect(batch.get("boolean_value")).toBe(true);
    expect(batch.get("null_value")).toBe(null);
  });

  test("should handle complex nested structures", () => {
    const data = {
      complex: [
        {
          values: [1, 2, 3],
          metadata: { type: "test" }
        },
        {
          values: [4, 5, 6],
          metadata: { type: "validation" }
        }
      ]
    };

    const batch = new BatchFeature(data);
    expect(batch.get("complex")).toEqual(data.complex);
  });

  test("should be chainable", () => {
    const batch = new BatchFeature();
    const result = batch.convertToTensors("np");

    expect(result).toBe(batch); // Should return the same instance
  });
});