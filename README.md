<div align="center">
<img src="./assets/logo-burn-neutral.webp" width="350px"/>
<div align="left">
&nbsp;

<div align="center">

[![Discord](https://img.shields.io/discord/1038839012602941528.svg?color=7289da&&logo=discord)](https://discord.gg/uPEBbYYDB6)
[![Current Crates.io Version](https://img.shields.io/crates/v/burn.svg)](https://crates.io/crates/burn)
[![Documentation](https://img.shields.io/badge/docs-latest-blue)](https://burn.dev/docs/burn)
[![Test Status](https://github.com/burn-rs/burn/actions/workflows/test.yml/badge.svg)](https://github.com/burn-rs/burn/actions/workflows/test.yml)
[![CodeCov](https://codecov.io/gh/burn-rs/burn/branch/main/graph/badge.svg)](https://codecov.io/gh/burn-rs/burn)
[![Rust Version](https://img.shields.io/badge/Rust-1.71.0+-blue)](https://releases.rs/docs/1.71.0)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)

Burn is a new comprehensive dynamic Deep Learning Framework built using Rust <br /> with extreme flexibilty, compute efficiency and portability as its primary goals.


<div align="left">

## Performance

<div align="left">
<img align="right" src="./assets/logo-burn-small.png" height="96px"/>
Performance is a core pillard of Burn, since the goal of a deep learning framework is to convert computation into usefull intelligence.
We strive to achieve top efficiency by leveraging multiple optmization techniques described bellow 👇
</div>

<br />

<details>
<summary>
Automatic kernel fusion 💥
</summary>
<br />
The goal of Burn is to optmize your models on any backend.
Therefore, we provide a way to create custom compute shaders that minimize memory movements, extremely useful for memory bound code.
As an example, you can write your own GELU activation function using new error function implementation with the high level tensor api, and a custom kernel is going to be created automaticaly that will rivals a custom GPU implementation.

Rust code snipper
```rust
fn my_custom_gelu() {
}
fn my_erf_function() {
}
```

The created kernel.
```wgsl
```

<br />
</details>

<details>
<summary>
Asynchronous execution 🧨
</summary>
<br />

For backends developed from scrach by the Burn team, an asynchronous execution style is used which allows to perform various optimizations, with automatic kernel fusion being one of them.
Asynchronous execution also ensures that the framework code won't block the GPU execution, which implies that the frameowkr overhead doesn't impact the speed of execution.
For more information about this subject, see [this post](https://burn.dev/).

<br />

</details>

<details>
<summary>
Threadsafe building blocks ❤️‍🔥
</summary>
<br />
Burn enphasises threadsafety by leveraging the ownership system of Rust.
With Burn, each module owns its weights, therefore you can send a module to another thread, compute the gradients, send the gradients to another the main thread to aggregate the gradients, and pouf you just implemented multi-device training.
This is very different from PyTorch where backprobagation actually mutate each tensor by adding its grad attribute, which isn't a thread safe operation, therefore requirering lower level synchonisation primitives.

<br />
<br />
</details>

<details>
<summary>
Intelligent memory mangement 🦀
</summary>
<br />

One of the main role of a deep learning framework is to reduce the amount of memory necesasry to run your models.
The most straingforward way of handling memory is that each tensor has its own memory space that is allocated when the tensor is created and deallocated when the tensor gets out of scope.
However, allocating and deallocating data is very costly in general, so a memory pool is often time required to achieve good throughput.
We created an infrastructure that allows for creating more memory mangement strategies that can be selected when creating a backend.
For more information about this subject, see [this post](https://burn.dev/).

Another very important part of our meory mangement strategy, is that we know when a tensor can be mutated in-place just by using the ownership system well.
Eventhough this is a quite small memory optimization, it adds up considerably when training or running inference with bigger models and contributes to reduce the memory usage even more.
For more information about this subject, see [this post](https://burn.dev/tenshandling).

<br />
</details>

<details>
<summary>
Automatic kernel selection 🎯
</summary>
<br />
One of the responsability of a deep learning framework is to ensure that your model runs smoothly on all hardware.
However, not all hardware share the same behavior in term of execution speed.
A matrix multiplication kernels can be lauched with many different properties, which is highly sensitive to the size of the matrices and the hardware.
Using the wrong configuration could reduce the speed of execution by a factor (10 times or even more in extreme cases) so chosing the right kernels become a priority.

<br />
<br />

With our home made backends, we actually run benchmarks automaticaly and chose the best configuration for the current hardware and matrices sizes with a reasonable caching strategy to not re-compute the benchmarks on small shape updates, fully supporting dynamic tensor shapes.
This will increase a bit the warmup time of execution, but it should stability after a few forward/backward passes.

<br />
</details>

<details>
<summary>
Hardware specific features 🔥
</summary>
<br />
It is no secret that deep learning is mosly relying on matrix multiplication as its core operation, since this is how fully connected neurons are modeled.
More and more, hardware manifacturer take advantage of this caracteristique to optimize their chip specificatlly for matrix mutiliplication workloads. 
NVidia has its tensor core, now most cell phone have AI chips, Burn will try to leverage those as much as possible.
https://github.com/gpuweb/gpuweb/issues/4195

<br />

DISCLAMER en italic.
We currently don't have an in-house backend that support Tensor Core yet, since we focus our development on portability though using WGSL shader first.
This decision was made because we already have a pretty efficient CUDA implementation through the use of LibTorch and Candles.
However we will create more backends in the future to nire custom optimizations to those platform such as kernel fusion and automatic kernel selection.

<br />
</details>

<details>
<summary>
Custom Backend Extension 🎒
</summary>
<br />
Burn aims to be the most flexible deep learning framework.
While it's crucial to maintain compatibility with a wide variety of backends, Burn also provides the ability to extend the functionality of a backend implementation to suit your modeling requirements.
This versatility is advantageous in numerous ways, such as supporting custom operations like flash attention or manually writing your own kernel for a specific backend to enhance performance.
See [this section](https://burn.dev/book/advanced/backend-extension/index.html) in the Burn Book 🔥 for more details
<br />
</details>

## Backends

<div align="left">
<img align="right" src="./assets/logo-burn-small.png" height="96px"/>
Burn strives to be as fast as possible on as many hardware as possible with robust reliability.
We believe this flexibilty is crucial for modern needs where you may train your models in the cloud, but deploy on customer hardwares, which varies from user to user.
</div>

<br />

Burn has a very different approach to supporting many backends compared to other frameworks.
By design, your code will be generic over the Backend trait, which allows us to built Burn with backends as plugins, which allows you to compose backends to add additional functionalities such as automatic kernel fusion and autodifferiention.


<details>
<summary>
WGPU: Cross-Platform GPU Backend 🌐
</summary>

Vulkan, Metal, Direct X, link to the blog. Integrated graphics card. 

First in-house backend with tons of special optimizations.
WGPU serves as our research playgound for a variaty of optimizations.
Where we do most of our optimization research for now.

</details>

<details>
<summary>
Candle: Backend using the Candles bindings 🕯
</summary>

Web Browser support CPU wasm.
CUDA
 
Not fully complete yet.
</details>

<details>
<summary>
LibTorch: Backend using the LibTorch bindings 🎆
</summary>

PyTorch doesn't need an introduction in this realm.
We leverage the bindings created by link to repo so that you can use LibTorch C++ to execute kernels operations on CPU, CUDA and Metal.
</details>

<details>
<summary>
NdArray: Backend using the ndarray primitive as data structure. 🦐
</summary>

NdArray isn't the fastest backend, but proves to be extremely portable.
No-std support

</details>

<details>
<summary>
Autodiff: Backend decorator that brings backprobagation to any backend. 🥵
</summary>

You can't makes a mistake and call backward on a model that runs on a backend that doens't support autodiff.
You can simply..

```rust

type Backend = Wgpu;
type ADBackend = Autodiff<Backend>;

fn main() {
   let x = Tensor::rendom
}
```

</details>

<details>
<summary>
Fusion: Backend decorator that brings kernel fusion to backends that support it. 💥
</summary>

```rust

type Backend = Fusion<Wgpu>;
type ADBackend = Autodiff<Backend>;

fn main() {
   let x = Tensor::random
}
```

Of note, we plan to implement automatic gradient checkpointing based on compute bound and memory bound operation, which will compose with the fusion backend so that your code will run ever faster when training your model. Link to issue.

</details>


## Training & Inference

- ONNX
- Burn train
YOUTUBE VIDEO
- Wasm

## Getting Started

  - Book
  - Models
  - Examples
  - Why Rust

## Status

Burn is currently in active development, and there will be breaking changes. While any resulting
issues are likely to be easy to fix, there are no guarantees at this stage.

## License

Burn is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for details. Opening a pull
request is assumed to signal agreement with these licensing terms.
