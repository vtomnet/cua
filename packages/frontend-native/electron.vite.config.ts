import { defineConfig } from "electron-vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import svgr from "vite-plugin-svgr";
import path from "path";
import { resolve } from "node:path";

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
      outDir: "dist/preload",
      rollupOptions: {
        input: {
          index: resolve(__dirname, 'src/preload/index.ts'),
          sandbox: resolve(__dirname, 'src/preload/sandbox.ts'),
        }
      }
    }
  },
  renderer: {
    envDir: path.resolve(__dirname, "../.."),
    root: "./src/renderer",
    publicDir: "./src/renderer/public",
    plugins: [
      react(),
      tailwindcss(),
      svgr(),
    ],
    resolve: {
      conditions: ['onnxruntime-web-use-extern-wasm', 'import', 'module', 'browser', 'default'],
    },
    optimizeDeps: {
      exclude: ["onnxruntime-web"],
    },
    assetsInclude: ["**/*.wasm", "**/*.mjs"],
    build: {
      outDir: "dist/renderer",
      commonjsOptions: {
        transformMixedEsModules: true,
      },
      rollupOptions: {
        input: {
          index: path.resolve(__dirname, 'src/renderer/index.html'),
          control: path.resolve(__dirname, 'src/renderer/control.html'),
        },
        external: ['onnxruntime-web'],
        output: {
          assetFileNames: (assetInfo) => {
            // Keep WASM files without hash for predictable paths
            if (assetInfo.name && assetInfo.name.endsWith('.wasm')) {
              return '[name][extname]';
            }
            if (assetInfo.name && assetInfo.name.endsWith('.mjs')) {
              return '[name][extname]';
            }
            return '[name]-[hash][extname]';
          },
          paths: {
            'onnxruntime-web': './ort.min.mjs'
          },
        },
        treeshake: {
          moduleSideEffects: (id) => {
            // Don't tree-shake onnxruntime-web - preserve all exports
            return id.includes('onnxruntime-web');
          },
        },
      },
    }
  },
});
