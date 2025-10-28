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
for d in whisper-feature-extractor open-things core frontend-core; do bun run --cwd=packages/$d build; done
```

Now you may run `bun dev` from frontend-native, frontend-web, or server. You may need to rebuild core or frontend-core after updating them.
