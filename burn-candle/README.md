# Burn Candle Backend

This crate provides a backend for [Burn](https://github.com/burn-rs/burn) based on the [Candle](https://github.com/huggingface/candle) framework. 

It is still in alpha stage, not all operations are supported. It is usable for some use cases, like for inference. 

It can be used with CPU or CUDA. On macOS computations can be accelerated by using the Accelerate framework.

## Feature Flags

The following features are supported:

- `cuda` - Cuda GPU device (NVIDIA only)
- `accelerate` - Accelerate framework (macOS only)
