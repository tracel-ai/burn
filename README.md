# BURN

> BURN: Burn Unstoppable Rusty Neurons

This library aims to be a complete deep learning framework with extreme flexibility written in Rust.
The goal would be to satisfy researchers as well as practitioners making it easier to experiment, train and deploy your solution.

## Why Rust?

A big benefit of using Rust instead of Python is to allow performant multi-threaded deep learning networks which might open new doors for more efficient models.
Scale seems to be very important, but the only tool we currently have to achieve it is big matrix multiplication on GPUs.
This often implies big batch sizes, which is impossible for online learning.
Also, asynchronus sparsely activated networks without copying weights is kind of impossible to achieve with Python (or really hard without proper threading).

## Burn-Tensor

BURN has its own tensor library supporting multiple backends, it can also be used for other scientific computing applications.
Click [here](./burn-tensor/) for more details.

## Module Definition

Currently working on it ... 💻
