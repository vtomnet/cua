import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
// const nodeModulesDir = resolve(__dirname, "../../node_modules");

export default defineConfig({
  server: {
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
  // resolve: {
  //   alias: {
  //     "@mediapipe/tasks-vision": resolve(nodeModulesDir, "@mediapipe/tasks-vision/vision_bundle.mjs"),
  //     "@techstark/opencv-js": resolve(nodeModulesDir, "@techstark/opencv-js/dist/opencv.js"),
  //   },
  // },
  assetsInclude: ["**/*.onnx"],
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
    include: ["@mediapipe/tasks-vision", "@techstark/opencv-js"],
  },
  build: {
    target: "es2020",
    modulePreload: {
      polyfill: false,
    },
    rollupOptions: {
      external: [],
    },
  },
  worker: {
    format: "iife",
    rollupOptions: {
      external: [],
    },
  },
});
