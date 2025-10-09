import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
  assetsInclude: ["**/*.onnx"],
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  build: {
    target: 'es2020',
    modulePreload: {
      polyfill: false
    }
  }
});
