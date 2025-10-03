import type { BatchFeatureData, TensorType } from "./types";

/**
 * Holds the output of feature extractor methods
 * This class provides a type-safe key-value store for feature data
 */
export class BatchFeature {
  private readonly data: Record<string, any> = {};

  constructor(data?: BatchFeatureData, tensorType?: TensorType) {
    if (data) {
      Object.assign(this.data, data);
    }

    if (tensorType) {
      this.convertToTensors(tensorType);
    }
  }

  get<T = any>(key: string): T | undefined {
    return this.data[key];
  }

  set(key: string, value: any): this {
    this.data[key] = value;
    return this;
  }

  has(key: string): boolean {
    return key in this.data;
  }

  keys(): string[] {
    return Object.keys(this.data);
  }

  values(): any[] {
    return Object.values(this.data);
  }

  entries(): Array<[string, any]> {
    return Object.entries(this.data);
  }

  /**
   * Convert the inner content to tensors
   */
  convertToTensors(tensorType: TensorType): this {
    if (tensorType === "np") {
      // For NumPy-like tensors, we'll keep as nested arrays
      // In a real implementation, you might want to use a proper tensor library
      for (const [key, value] of this.entries()) {
        if (Array.isArray(value)) {
          this.data[key] = this.ensureNumericArray(value);
        }
      }
    } else if (tensorType === "pt") {
      throw new Error("PyTorch tensor conversion not implemented in this TypeScript version");
    }

    return this;
  }

  private ensureNumericArray(arr: unknown): number[][][] | number[][] | number[] | number {
    if (Array.isArray(arr)) {
      return arr.map(item =>
        Array.isArray(item) ? this.ensureNumericArray(item) : Number(item)
      ) as number[][][] | number[][] | number[];
    }
    return Number(arr);
  }

  /**
   * Convert to plain object
   */
  toObject(): BatchFeatureData {
    return { ...this.data };
  }
}