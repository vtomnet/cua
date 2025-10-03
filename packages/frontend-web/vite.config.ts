import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

const repoRoot = path.resolve(__dirname, "../../");

export default defineConfig({
  cacheDir: path.resolve(repoRoot, "node_modules/.vite"),
  resolve: {
    alias: {
      "onnxruntime-web": path.resolve(repoRoot, "node_modules/onnxruntime-web"),
    },
    dedupe: ["onnxruntime-web"],
  },
  server: {
    fs: {
      allow: [
        repoRoot,
        path.resolve(__dirname),
      ],
    },
  },
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  assetsInclude: ["**/*.wasm"],
  plugins: [
    svelte(),
    tailwindcss(),
  ],
});
