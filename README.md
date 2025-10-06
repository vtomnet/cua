# Computer Use Agent

Install [Bun](https://bun.com).

Obtain the following files from GitHub/HuggingFace:

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

Set VITE_OPENAI_API_KEY and CEREBRAS_API_KEY.

Run once:
```bash
bun install
(cd packages/frontend-native && bun run link-electron)
(cd packages/server && bun run link-ort)
(cd packages/core && bun run build)
(cd packages/frontend-core && bun run build)
(cd packages/frontend-native && bun run build)
```

Now you may run `bun dev` / `bun run build` from any package directory. You may need to rebuild packages/core or packages/frontend-core after updating them.
