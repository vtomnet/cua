#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p node_modules
ln -sf ../../../node_modules/electrobun/ node_modules/electrobun
