import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import svgr from "vite-plugin-svgr";

export default defineConfig({
  main: {
    plugins: [
      svgr(),
    ],
    build: {
      outDir: "dist/main",
      rollupOptions: {
        external: ["@jitsi/robotjs"]
      }
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
      tailwindcss(),
      svgr(),
    ],
    build: {
      outDir: "dist/renderer"
    }
  },
});
