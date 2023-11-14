#!/usr/bin/env bash

mkdir -p ./src/assets
cp ./node_modules/sql.js/dist/sql-wasm.wasm ./src/assets/sql-wasm.wasm
cp ~/.cache/burn-dataset/mnist.db ./public/mnist.db
