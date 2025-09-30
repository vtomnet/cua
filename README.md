# Computer Use Agent

This program targets both web and native (via Tauri).
To build for web, you will need [Bun](https://bun.com).
To build for native, you will additionally need [Rust](https://rust-lang.org).

## init

```bash
bun add
```

## run

```bash
bun run dev:web        # or:
bun run dev:native     # or:
TARGET=native bun dev  # or TARGET=web
```

## index

```
core              Turn detection, ASR, LLMs. Audio + face landmarks in, ui actions out.
server            WebRTC server, counterpart to frontend-{web,native}. Builds on 'core'.
frontend-core     Mediapipe, Silero VAD via ORT, maybe audio/camera recording routines.
frontend-web      Builds on frontend-core, includes VNC and maybe a Linux VM runner.
frontend-native   Tauri program, otherwise builds on frontend-core. + UI automation.
frontend-local    Like frontend-web, but no server dependency. Builds on 'core'.
```
