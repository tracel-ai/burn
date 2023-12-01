# Image Classification Web Demo Using Burn and WebAssembly

[![Live Demo](https://img.shields.io/badge/live-demo-brightgreen)](https://antimora.github.io/image-classification/)


## Overview

This demo showcases how to execute an image classification task in a web browser using a model
converted to Rust code. The project utilizes the Burn deep learning framework, WebGPU and
WebAssembly . Specifically, it demonstrates:

1. Converting an ONNX (Open Neural Networks Exchange) model into Rust code compatible with the Burn
   framework.
2. Executing the model within a web browser using WebGPU via the `burn-wgpu` backend and WebAssembly
   through the `burn-ndarray` and `burn-candle` backends.

## Running the Demo

### Step 1: Build the WebAssembly Binary and Other Assets

To compile the Rust code into WebAssembly and build other essential files, execute the following
script:

```bash
./build-for-web.sh
```

### Step 2: Launch the Web Server

Run the following command to initiate a web server on your local machine:

```bash
./run-server.sh
```

### Step 3: Access the Web Demo

Open your web browser and navigate to:

```plaintext
http://localhost:8000
```

## Backend Compatibility

As of now, the WebGPU backend is compatible only with Chrome browsers running on macOS and Windows.
The application will dynamically detect if WebGPU support is available and proceed accordingly.

## SIMD Support

The build targets two sets of binaries, one with SIMD support and one without. The web application
dynamically detects if SIMD support is available and downloads the appropriate binary.

## Model Information

The image classification task is achieved using the SqueezeNet model, a compact Convolutional Neural
Network (CNN). It is trained on the ImageNet dataset and can classify images into 1,000 distinct
categories. The included ONNX model is sourced from the
[ONNX Model Zoo](https://github.com/onnx/models/tree/main/vision/classification/squeezenet). For
further details about the model's architecture and performance, you can refer to the
[original paper](https://arxiv.org/abs/1602.07360).

## Credits

This demo was inspired by the ONNX Runtime web demo featuring the
[SqueezeNet model trained on ImageNet](https://microsoft.github.io/onnxruntime-web-demo/#/squeezenet).

The complete list of credits/attribution can be found in the [NOTICES](NOTICES.md) file.

## Future Enhancements

- [ ] Fall back to WebGL if WebGPU is not supported by the browser. See
      [wgpu's WebGL support ](https://github.com/gfx-rs/wgpu/wiki/Running-on-the-Web-with-WebGPU-and-WebGL)

- [ ] Enable SIMD support for Safari browsers after Release 179.

- [ ] Add image paste functionality to allow users to paste an image from the clipboard.
