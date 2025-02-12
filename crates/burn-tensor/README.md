# Burn Tensor

> [Burn](https://github.com/tracel-ai/burn) Tensor Library

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-tensor.svg)](https://crates.io/crates/burn-tensor)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn-tensor/blob/master/README.md)

This library provides the core abstractions required to run tensor operations with Burn.

`Tensor`s are generic over the backend to allow users to perform operations using different
`Backend` implementations. Burn's tensors also support support auto-differentiation thanks to the
`AutodiffBackend` trait.
