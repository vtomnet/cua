import { BatchFeature } from "./batch-feature";
import type { PaddingOptions, TensorType } from "./types";

export enum PaddingStrategy {
  DO_NOT_PAD = "doNotPad",
  LONGEST = "longest",
  MAX_LENGTH = "maxLength"
}

/**
 * Base class for sequence feature extractors
 */
export abstract class SequenceFeatureExtractor {
  protected featureSize: number;
  protected samplingRate: number;
  protected paddingValue: number;
  protected paddingSide: string;
  protected returnAttentionMask: boolean;
  protected modelInputNames: string[] = ["input_features"];

  constructor(
    featureSize: number,
    samplingRate: number,
    paddingValue: number,
    options: { paddingSide?: string; returnAttentionMask?: boolean } = {}
  ) {
    this.featureSize = featureSize;
    this.samplingRate = samplingRate;
    this.paddingValue = paddingValue;
    this.paddingSide = options.paddingSide || "right";
    this.returnAttentionMask = options.returnAttentionMask !== undefined ? options.returnAttentionMask : true;
  }

  /**
   * Pad input values up to predefined length or to the max sequence length in the batch
   */
  pad(
    processedFeatures: BatchFeature | BatchFeature[],
    options: PaddingOptions = {}
  ): BatchFeature {
    const {
      padding = true,
      maxLength,
      truncation = false,
      padToMultipleOf,
      returnAttentionMask,
      returnTensors
    } = options;

    // Convert list of BatchFeatures to single BatchFeature with lists
    const features = Array.isArray(processedFeatures)
      ? this.combineBatchFeatures(processedFeatures)
      : processedFeatures;

    // Check that main input is present
    if (!features.has(this.modelInputNames[0] ?? "")) {
      throw new Error(
        `You should supply an instance of BatchFeature that includes ${this.modelInputNames[0]}, ` +
        `but you provided ${Array.from(features.keys())}`
      );
    }

    const requiredInput = features.get(this.modelInputNames[0] ?? "");
    const shouldReturnAttentionMask = returnAttentionMask !== undefined ? returnAttentionMask : this.returnAttentionMask;

    if (requiredInput.length === 0) {
      if (shouldReturnAttentionMask) {
        features.set("attention_mask", []);
      }
      return features;
    }

    // Convert padding strategy
    const paddingStrategy = this.getPaddingStrategy(padding, maxLength);

    const batchSize = requiredInput.length;

    // Truncate inputs first
    const truncatedInputs: any[] = [];
    for (let i = 0; i < batchSize; i++) {
      const inputs: { [key: string]: any } = {};
      for (const key of features.keys()) {
        inputs[key] = features.get(key)[i];
      }

      const truncatedInput = this.truncate(inputs, {
        maxLength,
        padToMultipleOf,
        truncation
      });

      truncatedInputs.push(truncatedInput);
    }

    // Determine max length for padding
    let actualMaxLength = maxLength;
    if (paddingStrategy === PaddingStrategy.LONGEST) {
      actualMaxLength = Math.max(...truncatedInputs.map(input => input[this.modelInputNames[0] ?? ""]?.length ?? 0));
    }

    // Pad each input
    const batchOutputs: { [key: string]: any[] } = {};
    for (let i = 0; i < batchSize; i++) {
      const paddedOutput = this.padSingle(
        truncatedInputs[i],
        actualMaxLength,
        paddingStrategy,
        padToMultipleOf,
        shouldReturnAttentionMask
      );

      for (const [key, value] of Object.entries(paddedOutput)) {
        if (!batchOutputs[key]) {
          batchOutputs[key] = [];
        }
        batchOutputs[key].push(value);
      }
    }

    const result = new BatchFeature(batchOutputs, returnTensors);
    return result;
  }

  /**
   * Pad a single input
   */
  private padSingle(
    processedFeatures: { [key: string]: any },
    maxLength?: number,
    paddingStrategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    padToMultipleOf?: number,
    returnAttentionMask?: boolean
  ): { [key: string]: any } {
    const requiredInput = processedFeatures[this.modelInputNames[0] ?? ""];

    if (paddingStrategy === PaddingStrategy.LONGEST && maxLength === undefined) {
      maxLength = requiredInput.length;
    }

    if (maxLength !== undefined && padToMultipleOf !== undefined && (maxLength % padToMultipleOf !== 0)) {
      maxLength = Math.floor(maxLength / padToMultipleOf + 1) * padToMultipleOf;
    }

    const needsToBePadded = paddingStrategy !== PaddingStrategy.DO_NOT_PAD &&
                           maxLength !== undefined &&
                           requiredInput.length < maxLength;

    if (returnAttentionMask && !("attention_mask" in processedFeatures)) {
      processedFeatures["attention_mask"] = new Array(requiredInput.length).fill(1);
    }

    if (needsToBePadded && maxLength !== undefined) {
      const difference = maxLength - requiredInput.length;

      if (this.paddingSide === "right") {
        if (returnAttentionMask) {
          processedFeatures["attention_mask"] = [
            ...processedFeatures["attention_mask"],
            ...new Array(difference).fill(0)
          ];
        }

        // Pad all sequence features, not just the main input
        for (const [key, value] of Object.entries(processedFeatures)) {
          if (key === "attention_mask") continue; // Already handled above

          if (Array.isArray(value) && value.length === requiredInput.length) {
            if (key === this.modelInputNames[0] && this.featureSize > 1) {
              // 2D padding for main input feature vectors
              const paddingArray = new Array(difference).fill(null).map(() =>
                new Array(this.featureSize).fill(this.paddingValue)
              );
              processedFeatures[key] = [
                ...value,
                ...paddingArray
              ];
            } else {
              // 1D padding for other features
              processedFeatures[key] = [
                ...value,
                ...new Array(difference).fill(this.paddingValue)
              ];
            }
          }
        }
      } else if (this.paddingSide === "left") {
        if (returnAttentionMask) {
          processedFeatures["attention_mask"] = [
            ...new Array(difference).fill(0),
            ...processedFeatures["attention_mask"]
          ];
        }

        // Pad all sequence features, not just the main input
        for (const [key, value] of Object.entries(processedFeatures)) {
          if (key === "attention_mask") continue; // Already handled above

          if (Array.isArray(value) && value.length === requiredInput.length) {
            if (key === this.modelInputNames[0] && this.featureSize > 1) {
              // 2D padding for main input feature vectors
              const paddingArray = new Array(difference).fill(null).map(() =>
                new Array(this.featureSize).fill(this.paddingValue)
              );
              processedFeatures[key] = [
                ...paddingArray,
                ...value
              ];
            } else {
              // 1D padding for other features
              processedFeatures[key] = [
                ...new Array(difference).fill(this.paddingValue),
                ...value
              ];
            }
          }
        }
      } else {
        throw new Error("Invalid padding strategy: " + this.paddingSide);
      }
    }

    return processedFeatures;
  }

  /**
   * Truncate inputs to predefined length
   */
  private truncate(
    processedFeatures: { [key: string]: any },
    options: {
      maxLength?: number;
      padToMultipleOf?: number;
      truncation?: boolean;
    } = {}
  ): { [key: string]: any } {
    const { maxLength, padToMultipleOf, truncation } = options;

    if (!truncation) {
      return processedFeatures;
    }

    if (truncation && maxLength === undefined) {
      throw new Error("When setting truncation=true, make sure that maxLength is defined");
    }

    const requiredInput = processedFeatures[this.modelInputNames[0] ?? ""];

    // Find maxLength that fits padToMultipleOf
    let actualMaxLength = maxLength!;
    if (padToMultipleOf !== undefined && (actualMaxLength % padToMultipleOf !== 0)) {
      actualMaxLength = Math.floor(actualMaxLength / padToMultipleOf + 1) * padToMultipleOf;
    }

    const needsToBeTruncated = requiredInput.length > actualMaxLength;

    if (needsToBeTruncated) {
      // Truncate all sequence features, not just the main input
      for (const [key, value] of Object.entries(processedFeatures)) {
        if (Array.isArray(value) && value.length === requiredInput.length) {
          processedFeatures[key] = value.slice(0, actualMaxLength);
        }
      }
    }

    return processedFeatures;
  }

  /**
   * Combine multiple BatchFeatures into a single BatchFeature
   */
  private combineBatchFeatures(batchFeatures: BatchFeature[]): BatchFeature {
    const firstFeature = batchFeatures[0];
    if (!firstFeature) {
      return new BatchFeature();
    }

    const keys = firstFeature.keys();
    const combinedData: { [key: string]: any[] } = {};

    for (const key of keys) {
      combinedData[key] = batchFeatures.map(bf => {
        const value = bf.get(key);
        // If the BatchFeature contains a single array, extract it, otherwise keep as is
        return Array.isArray(value) && value.length === 1 ? value[0] : value;
      });
    }

    return new BatchFeature(combinedData);
  }

  /**
   * Get padding strategy from options
   */
  private getPaddingStrategy(padding: boolean | string, maxLength?: number): PaddingStrategy {
    if (padding === false) {
      return PaddingStrategy.DO_NOT_PAD;
    }

    if (padding === true) {
      return PaddingStrategy.LONGEST;
    }

    const strategyMap: Record<string, PaddingStrategy> = {
      longest: PaddingStrategy.LONGEST,
      maxLength: PaddingStrategy.MAX_LENGTH,
      doNotPad: PaddingStrategy.DO_NOT_PAD,
    };

    const strategy = strategyMap[padding];
    if (!strategy) {
      throw new Error(`Unknown padding strategy: ${padding}`);
    }

    // Validate maxLength is provided when needed
    if (maxLength === undefined && strategy === PaddingStrategy.MAX_LENGTH) {
      throw new Error("When setting padding=maxLength, make sure that maxLength is defined");
    }

    // Test if we have a padding value
    if (strategy !== PaddingStrategy.DO_NOT_PAD && this.paddingValue === undefined) {
      throw new Error(
        "Asking to pad but the feature_extractor does not have a padding value. " +
        "Please select a value to use as paddingValue."
      );
    }

    return strategy;
  }
}