# Burn Tensor

> [Burn](https://github.com/tracel-ai/burn) Tensor Library

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-tensor.svg)](https://crates.io/crates/burn-tensor)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn-tensor/blob/master/README.md)

This library provides the core abstractions required to run tensor operations with Burn.

`Tensor<D, K>` is generic over rank and tensor kind. Its `Device` selects the backend at runtime;
operations pass through an opaque bridge and dispatch to backend primitives. With the `autodiff`
feature, create tensors on `device.autodiff()` and call `require_grad()` on source leaves whose
gradients you need. Models and tensor functions do not need a backend type parameter.
