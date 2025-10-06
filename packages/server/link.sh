#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

ln -sf ../../../node_modules/onnxruntime-web/dist public/onnx
