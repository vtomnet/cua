import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { copyFileSync, existsSync, mkdirSync } from "fs";
import { join } from "path";

export default defineConfig({
  main: {
    plugins: [
      {
        name: 'copy-icon',
        writeBundle() {
          const destDir = join(process.cwd(), 'dist/main');
          if (!existsSync(destDir)) {
            mkdirSync(destDir, { recursive: true });
          }

          // Copy microphone tray icon
          const micSrcPath = join(process.cwd(), 'microphone-tray.png');
          const micDestPath = join(destDir, 'microphone-tray.png');
          if (existsSync(micSrcPath)) {
            copyFileSync(micSrcPath, micDestPath);
            console.log('Copied microphone-tray.png to dist/main/');
          }

          // Copy microphone tray icon @2x
          const mic2xSrcPath = join(process.cwd(), 'microphone-tray@2x.png');
          const mic2xDestPath = join(destDir, 'microphone-tray@2x.png');
          if (existsSync(mic2xSrcPath)) {
            copyFileSync(mic2xSrcPath, mic2xDestPath);
            console.log('Copied microphone-tray@2x.png to dist/main/');
          }

          // Copy microphone-slash tray icon
          const micSlashSrcPath = join(process.cwd(), 'microphone-slash-tray.png');
          const micSlashDestPath = join(destDir, 'microphone-slash-tray.png');
          if (existsSync(micSlashSrcPath)) {
            copyFileSync(micSlashSrcPath, micSlashDestPath);
            console.log('Copied microphone-slash-tray.png to dist/main/');
          }

          // Copy microphone-slash tray icon @2x
          const mic2xSlashSrcPath = join(process.cwd(), 'microphone-slash-tray@2x.png');
          const mic2xSlashDestPath = join(destDir, 'microphone-slash-tray@2x.png');
          if (existsSync(mic2xSlashSrcPath)) {
            copyFileSync(mic2xSlashSrcPath, mic2xSlashDestPath);
            console.log('Copied microphone-slash-tray@2x.png to dist/main/');
          }
        }
      }
    ],
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
