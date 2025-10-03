import { test, expect, describe } from "bun:test";
import {
  melFilterBank,
  windowFunction,
  spectrogram,
  hertzToMel,
  melToHertz,
  fft,
  stft,
  Complex
} from "./audio-utils";

describe("Audio Utils", () => {
  describe("Mel Scale Conversions", () => {
    test("should convert hertz to mel (HTK)", () => {
      expect(hertzToMel(0)).toBe(0);
      expect(hertzToMel(1000, "htk")).toBeCloseTo(1000.0, 1);
      expect(hertzToMel(4000, "htk")).toBeCloseTo(2146.1, 1);
    });

    test("should convert hertz to mel (Slaney)", () => {
      expect(hertzToMel(0, "slaney")).toBe(0);
      expect(hertzToMel(1000, "slaney")).toBeCloseTo(15.0, 1);
      expect(hertzToMel(2000, "slaney")).toBeCloseTo(25.1, 1);
    });

    test("should convert mel to hertz (HTK)", () => {
      expect(melToHertz(0)).toBe(0);
      expect(melToHertz(1000, "htk")).toBeCloseTo(1000.0, 0);
      expect(melToHertz(2146, "htk")).toBeCloseTo(4000.0, 0);
    });

    test("should convert mel to hertz (Slaney)", () => {
      expect(melToHertz(0, "slaney")).toBe(0);
      expect(melToHertz(15, "slaney")).toBeCloseTo(1000.0, 0);
      expect(melToHertz(25.1, "slaney")).toBeCloseTo(2000.0, -1); // Within 5
    });

    test("should handle round-trip conversion", () => {
      const freqs = [100, 500, 1000, 2000, 4000, 8000];

      for (const freq of freqs) {
        const melHtk = hertzToMel(freq, "htk");
        const backHtk = melToHertz(melHtk, "htk");
        expect(backHtk).toBeCloseTo(freq, 1);

        const melSlaney = hertzToMel(freq, "slaney");
        const backSlaney = melToHertz(melSlaney, "slaney");
        expect(backSlaney).toBeCloseTo(freq, 1);
      }
    });
  });

  describe("Window Functions", () => {
    test("should create Hann window", () => {
      const window = windowFunction(4, "hann", false); // Non-periodic
      expect(window.length).toBe(4);
      expect(window[0]).toBeCloseTo(0, 5);
      expect(window[1]).toBeCloseTo(0.75, 2);
      expect(window[2]).toBeCloseTo(0.75, 2);
      expect(window[3]).toBeCloseTo(0, 5);
    });

    test("should create Hamming window", () => {
      const window = windowFunction(4, "hamming", false); // Non-periodic
      expect(window.length).toBe(4);
      expect(window[0]).toBeCloseTo(0.08, 2);
      expect(window[1]).toBeCloseTo(0.77, 2);
      expect(window[2]).toBeCloseTo(0.77, 2);
      expect(window[3]).toBeCloseTo(0.08, 2);
    });

    test("should create Povey window", () => {
      const window = windowFunction(4, "povey", false); // Non-periodic
      expect(window.length).toBe(4);
      expect(window[0]).toBeCloseTo(0, 5);
      expect(window[3]).toBeCloseTo(0, 5);
    });

    test("should create Boxcar window", () => {
      const window = windowFunction(4, "boxcar");
      expect(window.length).toBe(4);
      expect(window[0]).toBe(1);
      expect(window[1]).toBe(1);
      expect(window[2]).toBe(1);
      expect(window[3]).toBe(1);
    });

    test("should throw for unknown window type", () => {
      expect(() => windowFunction(4, "unknown")).toThrow("Unknown window type");
    });
  });

  describe("FFT", () => {
    test("should compute FFT of simple signal", () => {
      const input = new Float64Array([1, 0, 0, 0]);
      const result = fft(input);

      expect(result.length).toBe(4);
      expect(result[0]).toEqual(new Complex(1, 0));
      expect(result[1]).toEqual(new Complex(1, 0));
      expect(result[2]).toEqual(new Complex(1, 0));
      expect(result[3]).toEqual(new Complex(1, 0));
    });

    test("should compute FFT of sine wave", () => {
      const N = 8;
      const input = new Float64Array(N);
      // Create a single frequency sine wave
      for (let i = 0; i < N; i++) {
        input[i] = Math.sin(2 * Math.PI * i / N);
      }

      const result = fft(input);
      expect(result.length).toBe(N);

      // Check that energy is concentrated at the expected frequency bin
      const magnitudes = result.map(c => Math.sqrt(c.real * c.real + c.imag * c.imag));
      const maxMagnitude = Math.max(...magnitudes);
      expect(maxMagnitude).toBeGreaterThan(1);
    });

    test("should handle power of 2 sizes", () => {
      const sizes = [2, 4, 8, 16, 32];

      for (const size of sizes) {
        const input = new Float64Array(size);
        input.fill(1);
        const result = fft(input);
        expect(result.length).toBe(size);
      }
    });
  });

  describe("STFT", () => {
    test("should compute STFT of simple signal", () => {
      const signal = new Float64Array(16);
      signal.fill(1);

      const window = windowFunction(8, "hann");
      const result = stft(signal, window, 8, 4);

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);
      expect(result[0]?.length).toBeGreaterThan(0);
    });

    test("should handle different hop lengths", () => {
      const signal = new Float64Array(32);
      for (let i = 0; i < signal.length; i++) {
        signal[i] = Math.sin(2 * Math.PI * i / 8);
      }

      const window = windowFunction(8, "hann");

      const result1 = stft(signal, window, 8, 2); // 75% overlap
      const result2 = stft(signal, window, 8, 4); // 50% overlap
      const result3 = stft(signal, window, 8, 8); // No overlap

      expect(result1.length).toBeGreaterThan(result2.length);
      expect(result2.length).toBeGreaterThan(result3.length);
    });

    test("should handle centering", () => {
      const signal = new Float64Array(16);
      signal.fill(1);

      const window = windowFunction(8, "hann");

      const resultCentered = stft(signal, window, 8, 4, true);
      const resultUncentered = stft(signal, window, 8, 4, false);

      expect(resultCentered.length).toBeGreaterThanOrEqual(resultUncentered.length);
    });
  });

  describe("Mel Filter Bank", () => {
    test("should create mel filter bank with default parameters", () => {
      const filters = melFilterBank(201, 80, 0, 8000, 16000);

      expect(filters.length).toBe(80); // num_mel_filters
      expect(filters[0]?.length).toBe(201); // num_frequency_bins

      // Check that filters are non-negative
      for (const filter of filters) {
        for (const value of filter) {
          expect(value).toBeGreaterThanOrEqual(0);
        }
      }
    });

    test("should create filters with different parameters", () => {
      const filters = melFilterBank(129, 40, 50, 4000, 8000, "slaney", "slaney");

      expect(filters.length).toBe(40);
      expect(filters[0]?.length).toBe(129);
    });

    test("should handle HTK mel scale", () => {
      const filters = melFilterBank(201, 80, 0, 8000, 16000, null, "htk");

      expect(filters.length).toBe(80);
      expect(filters[0]?.length).toBe(201);
    });

    test("should normalize filters when requested", () => {
      const filtersNormalized = melFilterBank(201, 80, 0, 8000, 16000, "slaney");
      const filtersUnnormalized = melFilterBank(201, 80, 0, 8000, 16000, null);

      // Normalized filters should have different values
      let different = false;
      for (let filterIdx = 0; filterIdx < 10; filterIdx++) {
        for (let freqIdx = 0; freqIdx < 201; freqIdx++) {
          if ((filtersUnnormalized[filterIdx]?.[freqIdx] ?? 0) > 0) {
            if (Math.abs((filtersNormalized[filterIdx]?.[freqIdx] ?? 0) - (filtersUnnormalized[filterIdx]?.[freqIdx] ?? 0)) > 1e-10) {
              different = true;
              break;
            }
          }
        }
        if (different) break;
      }
      expect(different).toBe(true);
    });

    test("should handle edge frequencies", () => {
      // Test with very low and high frequencies
      const filters1 = melFilterBank(201, 10, 0, 1000, 16000);
      const filters2 = melFilterBank(201, 10, 7000, 8000, 16000);

      expect(filters1.length).toBe(10);
      expect(filters2.length).toBe(10);
    });
  });

  describe("Spectrogram", () => {
    test("should compute power spectrogram", () => {
      const signal = new Float64Array(32);
      for (let i = 0; i < signal.length; i++) {
        signal[i] = Math.sin(2 * Math.PI * i / 8);
      }

      const window = windowFunction(8, "hann");
      const melFilters = melFilterBank(5, 4, 0, 4000, 8000);

      const result = spectrogram(
        signal,
        window,
        8, // frame_length
        4, // hop_length
        8, // fft_length
        2.0, // power
        true, // center
        "reflect", // pad_mode
        true, // onesided
        0.0, // dither
        undefined, // preemphasis
        melFilters, // mel_filters
        1e-10, // mel_floor
        "log10" // log_mel
      );

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(4); // mel_filters.length
      expect(result[0]?.length).toBeGreaterThan(0);

      // Check that values are finite
      for (const freqBin of result) {
        for (const value of freqBin) {
          expect(isFinite(value)).toBe(true);
        }
      }
    });

    test("should handle different power values", () => {
      const signal = new Float64Array(16);
      signal.fill(1);

      const window = windowFunction(8, "hann");
      const melFilters = melFilterBank(5, 4, 0, 4000, 8000);

      const result1 = spectrogram(signal, window, 8, 4, 8, 1.0, true, "reflect", true, 0.0, undefined, melFilters, 1e-10, "log10");
      const result2 = spectrogram(signal, window, 8, 4, 8, 2.0, true, "reflect", true, 0.0, undefined, melFilters, 1e-10, "log10");

      // Results should be different for different power values - check mel filter 1 which has non-zero values
      expect(result1[1]?.[0]).not.toBe(result2[1]?.[0]);
    });

    test("should apply dithering", () => {
      const signal = new Float64Array(16);
      signal.fill(0.1);

      const window = windowFunction(8, "hann");
      const melFilters = melFilterBank(5, 4, 0, 4000, 8000);

      const resultNoDither = spectrogram(signal, window, 8, 4, 8, 2.0, true, "reflect", true, 0.0, undefined, melFilters, 1e-10, "log10");
      const resultWithDither = spectrogram(signal, window, 8, 4, 8, 2.0, true, "reflect", true, 0.1, undefined, melFilters, 1e-10, "log10");

      // Results should be different when dithering is applied
      let different = false;
      for (let i = 0; i < Math.min(resultNoDither.length, resultWithDither.length); i++) {
        for (let j = 0; j < Math.min(resultNoDither[i]?.length ?? 0, resultWithDither[i]?.length ?? 0); j++) {
          if (Math.abs((resultNoDither[i]?.[j] ?? 0) - (resultWithDither[i]?.[j] ?? 0)) > 1e-10) {
            different = true;
            break;
          }
        }
        if (different) break;
      }
      expect(different).toBe(true);
    });

    test("should handle preemphasis", () => {
      const signal = new Float64Array(16);
      for (let i = 0; i < signal.length; i++) {
        signal[i] = Math.sin(2 * Math.PI * i / 8);
      }

      const window = windowFunction(8, "hann");
      const melFilters = melFilterBank(5, 4, 0, 4000, 8000);

      const resultNoPreemph = spectrogram(signal, window, 8, 4, 8, 2.0, true, "reflect", true, 0.0, undefined, melFilters, 1e-10, "log10");
      const resultWithPreemph = spectrogram(signal, window, 8, 4, 8, 2.0, true, "reflect", true, 0.0, 0.97, melFilters, 1e-10, "log10");

      // Results should be different when preemphasis is applied - check mel filter 1 which has non-zero values
      expect(resultNoPreemph[1]?.[0]).not.toBe(resultWithPreemph[1]?.[0]);
    });

    test("should handle different log scales", () => {
      const signal = new Float64Array(16);
      signal.fill(0.1);

      const window = windowFunction(8, "hann");
      const melFilters = melFilterBank(5, 4, 0, 4000, 8000);

      const resultLog10 = spectrogram(signal, window, 8, 4, 8, 2.0, true, "reflect", true, 0.0, undefined, melFilters, 1e-10, "log10");
      const resultLn = spectrogram(signal, window, 8, 4, 8, 2.0, true, "reflect", true, 0.0, undefined, melFilters, 1e-10, "log");

      // Results should be different for different log scales
      expect(resultLog10[0]?.[0]).not.toBe(resultLn[0]?.[0]);
    });

    test("should work without mel filters", () => {
      const signal = new Float64Array(16);
      signal.fill(0.1);

      const window = windowFunction(8, "hann");

      const result = spectrogram(signal, window, 8, 4, 8, 2.0, true, "reflect", true, 0.0, undefined, undefined, 1e-10, "log10");

      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(5); // (fft_length / 2) + 1 for onesided
      expect(result[0]?.length).toBeGreaterThan(0);
    });
  });
});