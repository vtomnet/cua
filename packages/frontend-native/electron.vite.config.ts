import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

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
});
