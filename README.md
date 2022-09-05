<div align="center">
<img src="https://github.com/burn-rs/burn/blob/main/assets/logo-burn-full.png" width="200px"/>

[![Current Crates.io Version](https://img.shields.io/crates/v/burn.svg)](https://crates.io/crates/burn)
[![Test Status](https://github.com/burn-rs/burn/actions/workflows/test-burn.yml/badge.svg)](https://github.com/burn-rs/burn/actions/workflows/test-burn.yml)
[![Documentation](https://docs.rs/burn/badge.svg)](https://docs.rs/burn)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/burn-rs/burn/blob/master/LICENSE)

<div align="left">

> This library aims to be a complete deep learning framework with extreme flexibility written in Rust. 
> The goal would be to satisfy researchers as well as practitioners making it easier to experiment, train and deploy your solution.

## Why Rust?

A big benefit of using Rust instead of Python is to allow performant multi-threaded deep learning networks which might open new doors for more efficient models.
Scale seems to be very important, but the only tool we currently have to achieve it is big matrix multiplication on GPUs.
This often implies big batch sizes, which is impossible for online learning.
Also, asynchronous sparsely activated networks without copying weights is kind of impossible to achieve with Python (or really hard without proper threading).

## Burn-Tensor

Burn has its own tensor library supporting multiple backends, it can also be used for other scientific computing applications.
Click [here](https://github.com/burn-rs/burn/burn-tensor) for more details.
