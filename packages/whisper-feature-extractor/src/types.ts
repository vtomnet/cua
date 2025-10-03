export type TensorType = "np" | "pt";

export interface BatchFeatureData {
  [key: string]: number[][][] | number[][] | number[] | any;
}

export interface WhisperFeatureExtractorConfig {
  featureSize?: number;
  samplingRate?: number;
  hopLength?: number;
  chunkLength?: number;
  nFft?: number;
  paddingValue?: number;
  dither?: number;
  returnAttentionMask?: boolean;
}

export interface SpectrogramOptions {
  power?: number;
  center?: boolean;
  padMode?: "reflect" | "constant" | "edge";
  onesided?: boolean;
  dither?: number;
  preemphasis?: number;
  melFilters?: number[][];
  melFloor?: number;
  logMel?: "log" | "log10" | "dB" | null;
  reference?: number;
  minValue?: number;
  dbRange?: number;
  removeDcOffset?: boolean;
}

export interface PaddingOptions {
  padding?: boolean | "longest" | "maxLength" | "doNotPad";
  maxLength?: number;
  truncation?: boolean;
  padToMultipleOf?: number;
  returnAttentionMask?: boolean;
  returnTensors?: TensorType;
}

export interface FeatureExtractionCallOptions extends PaddingOptions {
  samplingRate?: number;
  doNormalize?: boolean;
  device?: string;
  returnTokenTimestamps?: boolean;
}