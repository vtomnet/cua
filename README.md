# Computer Use Agent

This program targets both web and native (via Tauri).
To build for web, you will need [Bun](https://bun.com).
To build for native, you will additionally need [Rust](https://rust-lang.org).

You will also need to obtain the following files:

```
packages/frontend-web/public
└── models
    ├── moonshine-base
    │   ├── cached_decode.onnx
    │   ├── encode.onnx
    │   ├── preprocess.onnx
    │   └── uncached_decode.onnx
    ├── silero_vad.onnx
    └── smart-turn-v3.0.onnx
```

## init

```bash
bun add
```

## run

```bash
bun run dev:web     # or:
bun run dev:native  # or:
bun run dev:local   # or:
TARGET=web bun dev  # TARGET={web,native,local,server}
```
