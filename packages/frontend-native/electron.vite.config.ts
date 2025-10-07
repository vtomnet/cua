import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

// const repoRoot = path.resolve(__dirname, "../../");

export default defineConfig({
  main: {
    build: {
      outDir: "dist/main"
    }
  },
  preload: {
    build: {
      outDir: "dist/preload"
    }
  },
  renderer: {
    root: "./src/renderer",
    plugins: [
      react(),
      tailwindcss()
    ],
    build: {
      outDir: "dist/renderer"
    }
  },
  // cacheDir: path.resolve(repoRoot, "node_modules/.vite"),
  // resolve: {
  //   alias: {
  //     "onnxruntime-web": path.resolve(repoRoot, "node_modules/onnxruntime-web"),
  //   },
  //   dedupe: ["onnxruntime-web"],
  // },
  // server: {
  //   fs: {
  //     allow: [
  //       repoRoot,
  //       path.resolve(__dirname),
  //     ],
  //   },
  // },
  // optimizeDeps: {
  //   exclude: ["onnxruntime-web"],
  // },
});
