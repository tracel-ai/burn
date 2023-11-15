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
<img align="right" src="./assets/illu-fast-uni_T07.png" height="96px"/>
Because we believe the goal of a deep learning framework is to convert computation into useful intelligence, we have made performance a core pillar of Burn. 
We strive to achieve top efficiency by leveraging multiple optimization techniques described below 👇
</div>

<br />

<details>
<summary>
Automatic kernel fusion 💥
</summary>
<br />

Using Burn means having your models optimized on any backend.
For that, we provide a way to automatically and dynamically create custom compute shaders that minimize data relocation between different memory spaces, extremely useful for all code where moving memory is a bottleneck.

As an example, you could write your own GELU activation function with the high level tensor api (see Rust code snippet below).
Then, at runtime, a custom low-level kernel will be automatically created for your specific implementation (see WGSL kernel below) and will rival a handcrafted GPU implementation.

```rust
fn gelu_custom<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let x = x.clone() * ((x / SQRT_2).erf() + 1);
    x / 2
}
```

You probably don't want to code your deep learning model with a lower level shader language, since as shown below it is extremely verbose.
Note that the error function (erf) isn't part of the WGSL specifications (<a href="https://www.w3.org/TR/WGSL/https://www.w3.org/TR/WGSL/">WebGPU Shading Language</a>), so we automatically extend the language with our own implementation.

```wgsl
@group(0)
@binding(0)
var<storage, read> input_0_global: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> output_0_global: array<f32>;

@group(0)
@binding(2)
var<storage, read> scalars_f32: array<f32, 3>;

@group(0)
@binding(3)
var<storage, read> info: array<u32>;

const WORKGROUP_SIZE_X = 32u;
const WORKGROUP_SIZE_Y = 32u;
const WORKGROUP_SIZE_Z = 1u;

@compute
@workgroup_size(32, 32, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
    let id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;
let rank: u32 = info[0];


var index_input_0: u32 = 0u;

for (var i: u32 = 1u; i <= rank; i++) {
    let position = 0u * (2u * rank);
    let position_out = 1u * (2u * rank);

    let stride = info[position + i];
    let stride_out = info[position_out + i];
    let shape = info[position + rank + i];

    index_input_0 += id / stride_out % shape * stride;
}

let input_0 = input_0_global[index_input_0];

let local_0 = input_0 / scalars_f32[0];
let local_1 = erf(local_0);
let local_2 = local_1 + scalars_f32[1];
let local_3 = input_0 * local_2;
let local_4 = local_3 / scalars_f32[2];
output_0_global[id] = local_4;

}

/// An approximation of the error function: https://en.wikipedia.org/wiki/Error_function#Numerical_approximations
///
/// > (maximum error: 1.5×10−7)
/// > All of these approximations are valid for x ≥ 0. To use these approximations for negative x, use the fact that erf x is an odd function, so erf x = −erf(−x).
fn erf_positive(x: f32) -> f32 {
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * abs(x));
    let tmp = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1;

    return 1.0 - (tmp * t * exp(-x * x));
}

fn erf(x: f32) -> f32 {
    if (x < 0.0) {
        return -1.0 * erf_positive(-1.0 * x);
    }

    return erf_positive(x);
}
```

> As of now, our fusion strategy is only implemented for our own WGPU backend and supports only a subset of operations.
We plan to add more operations very soon and extend this technique to other future in-house backends.

<br />
</details>

<details>
<summary>
Asynchronous execution 🧨
</summary>
<br />

For [backends developed from scratch by the Burn team](#backends), an asynchronous execution style is used, which allows to perform various optimizations, such as the previously mentioned automatic kernel fusion.

Asynchronous execution also ensures that the normal execution of the framework does not block the model computations, which implies that the framework overhead does not impact the speed of execution. Conversely, the intense computations in the model do not interfere with the responsiveness of the framework.
For more information about our asynchronous backends, see <a href="https://burn.dev/blog/creating-high-performance-asynchronous-backends-with-burn-compute">this blog post</a>.

<br />

</details>

<details>
<summary>
Thread-safe building blocks ❤️‍🔥
</summary>
<br />

Burn emphasizes thread safety by leveraging the <a href="https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html">ownership system of Rust</a>.
With Burn, each module is the owner of its weights. It is therefore possible to send a module to another thread for computing the gradients, then send the gradients to the main thread that can aggregate them, and _voilà_, you get multi-device training.

This is a very different approach from the one of PyTorch, where backpropagation actually mutates each tensor by adding its _grad_ attribute, which is not a thread-safe operation and therefore requires lower level synchronization primitives.
Note that this is still very fast, just not as easy to implement and portable across different backends.

<br />
</details>

<details>
<summary>
Intelligent memory management 🦀
</summary>
<br />

One of the main roles of a deep learning framework is to reduce the amount of memory necessary to run models.
The naive way of handling memory is that each tensor has its own memory space, which is allocated when the tensor is created then deallocated as the tensor gets out of scope.
However, allocating and deallocating data is in general very costly, so a memory pool is oftentimes required to achieve good throughput.
Burn offers an infrastructure that allows for easily creating and selecting memory management strategies when creating a backend.
For more details on memory management in Burn, see <a href="https://burn.dev/blog/creating-high-performance-asynchronous-backends-with-burn-compute">this blog post</a>.

Another very important memory optimization of Burn is that we keep track of when a tensor can be mutated in-place just by using the ownership system well.
Even though it is a rather small memory optimization on its own, it adds up considerably when training or running inference with larger models and contributes to reduce the memory usage even more.
For more information, see <a href="https://burn.dev/blog/burn-rusty-approach-to-tensor-handling">this blog post about tensor handling</a>.

<br />
</details>

<details>
<summary>
Automatic kernel selection 🎯
</summary>
<br />

A good deep learning framework should ensure that models run smoothly on all hardware.
However, not all hardware share the same behavior in terms of execution speed.
For instance, a matrix multiplication kernel can be launched with many different parameters, which are highly sensitive to the size of the matrices and the hardware.
Using the wrong configuration could reduce the speed of execution by a large factor (10 times or even more in extreme cases), so choosing the right kernels becomes a priority.

With our home-made backends, we run benchmarks automatically and choose the best configuration for the current hardware and matrices sizes with a reasonable caching strategy.

This adds a small overhead by increasing the warmup execution time, but stabilizes quickly after a few forward and backward passes, saving lots of time in the long run.
Note that this feature isn't mandatory, and can be disabled when cold starts are a priority over optimized throughput.

<br />
</details>

<details>
<summary>
Hardware specific features 🔥
</summary>
<br />

It is no secret that deep learning is mosly relying on matrix multiplication as its core operation, since this is how fully-connected neural networks are modeled.

More and more, hardware manufacturers optimize their chips specifically for matrix mutiliplication workloads.
For instance, Nvidia has its _Tensor Cores_ and today most cellphones have AI specialized chips. Burn aims at leveraging those as much as possible; you can refer to [this issue](https://github.com/gpuweb/gpuweb/issues/4195).

<br />

> _Disclaimer:_ We do not currently have an in-house backend that support Tensor Cores yet, since we have first chosen to focus our development on portability through the use of WGSL shaders.
> This decision was made because we already support Tensor Cores when using LibTorch and Candle backends.
> However we will create more backends in the future to bring our custom optimizations such as kernel fusion and automatic kernel selection to all platforms.

<br />
</details>

<details>
<summary>
Custom Backend Extension 🎒
</summary>
<br />

Burn aims to be the most flexible deep learning framework.
While it's crucial to maintain compatibility with a wide variety of backends, Burn also provides the ability to extend the functionalities of a backend implementation to suit your personal modeling requirements.

This versatility is advantageous in numerous ways, such as supporting custom operations like flash attention or manually writing your own kernel for a specific backend to enhance performance.
See [this section](https://burn.dev/book/advanced/backend-extension/index.html) in the Burn Book 🔥 for more details.

<br />
</details>

## Backends

<div align="left">
<img align="right" src="./assets/illu-backend-uni_T07.png" height="96px"/>
Burn strives to be as fast as possible on as many hardwares as possible, with robust implementations.
We believe this flexibilty is crucial for modern needs where you may train your models in the cloud, then deploy on customer hardwares, which varies from user to user.
</div>

<br />

Compared to other frameworks, Burn has a very different approach to supporting many backends.
By design, most code is generic over the Backend trait, which allows us to build Burn with backends as plugins.
This makes composing backend possible, augmenting them with additional functionalities such as autodifferentiation and automatic kernel fusion.

We already have many backends implemented, all listed below 👇

<details>
<summary>
WGPU: Cross-Platform GPU Backend 🌐
</summary>
<br />

**The go-to backend for running on any GPU.**

Based on the most popular and well-supported Rust graphics library, [WGPU](https://wgpu.rs), this backend automatically targets Vulkan, OpenGL, Metal, Direct X11/12, and WebGPU.
It can also be compiled to Web Assembly to run in the browser while leveraging the GPU, see [this demo](https://antimora.github.io/image-classification/).
For more information on the benefits of this backend, see [this blog](https://burn.dev/blog/cross-platform-gpu-backend).

The WGPU backend is our first "in-house backend", which means the totality of its functionalities is self-contained within Burn.
It is fully optimized with the [performance characteristics mentioned earlier](#performance), as it serves as our research playgound for a variety of optimizations.

</details>

<details>
<summary>
Candle: Backend using the Candle bindings 🕯
</summary>
<br />

Based on [Candle by Hugging Face](https://github.com/huggingface/candle), a minimalist ML framework for Rust with a focus on performance and ease of use, this backend can run on CPU with support for Web Assembly or on Nvidia GPUs using CUDA.

> _Disclaimer:_ This backend is not fully complete yet, but can work in some contexts like inference.

</details>

<details>
<summary>
LibTorch: Backend using the LibTorch bindings 🎆
</summary>
<br />

PyTorch doesn't need an introduction in the realm of deep learning.
This backend leverages [PyTorch Rust bindings](https://github.com/LaurentMazare/tch-rs), enabling you to use LibTorch C++ kernels on CPU, CUDA and Metal.

</details>

<details>
<summary>
NdArray: Backend using the NdArray primitive as data structure 🦐
</summary>
<br />

This CPU backend is admittedly not our fastest backend, but offers extreme portability.

Its [no_std](https://docs.rust-embedded.org/book/intro/no-std.html) support allows you to run Burn even on embedded devices without an operating system.

</details>

<details>
<summary>
Autodiff: Backend decorator that brings backpropagation to any backend 🥵
</summary>
<br />

Contrary to the aforementioned backends, the Autodiff backend is actually a backend _decorator_. This means that it cannot exist by itself; it must encapsulate some other backend.

The simple act of wrapping a base backend with Autodiff transparently equips it with autodifferentiation support, making it possible to call backward on your model.

```rust
use burn::backend::{Autodiff, Wgpu};
use burn::tensor::{Distribution, Tensor};

fn main() {
    type Backend = Autodiff<Wgpu>;

    let x: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default);
    let y: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default).require_grad();

    let tmp = x.clone() + y.clone();
    let tmp = tmp * x;
    let tmp = tmp.exp();

    let grads = tmp.backward();
    let y_grad = y.grad(&grads).unwrap();
    println!("{y_grad}");

```

Of note, it is impossible to make the mistake of calling backward on a model that runs on a backend that does not support autodiff (for inference), as this method is only offered by an Autodiff backend.

</details>

<details>
<summary>
Fusion: Backend decorator that brings kernel fusion to backends that support it 💥
</summary>
<br />

This backend decorator enhances a backend with kernel fusion, provided that the inner backend supports it.
Note that you can compose this backend with other backend decorators such as `Autodiff`.
For now, only the WGPU backend has support for fused kernels.

```rust
use burn::backend::{Autodiff, Fusion, Wgpu};
use burn::tensor::{Distribution, Tensor};

fn main() {
    type Backend = Autodiff<Fusion<Wgpu>>;

    let x: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default);
    let y: Tensor<Backend, 2> = Tensor::random([32, 32], Distribution::Default).require_grad();

    let tmp = x.clone() + y.clone();
    let tmp = tmp * x;
    let tmp = tmp.exp();

    let grads = tmp.backward();
    let y_grad = y.grad(&grads).unwrap();
    println!("{y_grad}");
}

```

Of note, we plan to implement automatic gradient checkpointing based on compute bound and memory bound operation, which will compose with the fusion backend so that your code will run even faster when training your model. See [this issue](https://github.com/burn-rs/burn/issues/936).

</details>

## Training & Inference

<div align="left">
<img align="right" src="./assets/illu-flexible-altA-uni_T07.png" height="96px"/>

The whole machine learning operations pipeline is made easy with Burn, as you can monitor your training with an ergonomic dashboard, and run inference everywhere from embedded devices to directly within the browser.

</div>

<br />

<details>
<summary>
Training Dashboard
</summary>

As you can see in the following video, a new terminal UI dashboard based on the [Ratatui](https://github.com/ratatui-org/ratatui) crate allows users to follow their training with ease without having to connect to any external application.

<iframe width="420" height="315" src="https://www.youtube.com/watch?v=N9RM5CQbNQc"></iframe>

You can visualize your training and validation metrics updating in real-time and analyze the lifelong progression or recent history of any registered metrics using only the arrow keys. Break from the training loop without crashing, allowing potential checkpoints to be fully written or important pieces of code to complete without interruption 🛡

<br />

</details>

<details>
<summary>
ONNX Support
</summary>
<br />

ONNX (Open Neural Network Exchange) is an open-standard format that exports both the architecture and the weights of a deep learning model.

Burn supports the importation of models that follow the ONNX standard so you can easily port a model you have written in another framework like TensorFlow or PyTorch to Burn to benefit from all the advantage our framework offers.

Our ONNX support is further described [this section of the Burn Book 🔥](https://burn.dev/book/import/onnx-model.html) and [the README related to importations in Burn](./burn-import/README.md)>.

> **Note**: This crate is in active development and currently supports a
> [limited set of ONNX operators](SUPPORTED-ONNX-OPS.md).

</details>

<details>
<summary>
Inference in the Browser
</summary>
<br />

Several of our backends can compile to Web Assembly: Candle and NdArray for CPU, and WGPU for GPU. This means that you can run inference directly within a browser.

</details>

## Getting Started

<div align="left">
<img align="right" src="./assets/illu-community_driven-uni_T07.png" height="96px"/>

Just heard of Burn? Join our community by connecting to our [Discord](https://discord.gg/PbjzCPfs)!

Then have a look at our book, simple examples and pre-trained cutting-edge models.

</div>

<details>
<summary>
The Burn Book 🔥
</summary>
<br />

To begin working effectively with Burn, it is crucial to understand its key components and philosophy. For detailed examples and explanations covering every facet of the framework, including building blocks like tensors, modules and learners, please refer to [The Burn Book 🔥](https://burn.dev/book/).

</details>

<details>
<summary>
Examples
</summary>
<br />

Here is a code snippet showing how intuitive the framework is to use, where we declare a position-wise feed-forward module along with its forward pass.

```rust
use burn::nn;
use burn::module::Module;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
    dropout: Dropout,
    gelu: GELU,
}

impl<B: Backend> PositionWiseFeedForward<B> {
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);

        self.linear_outer.forward(x)
    }
}
```

For more practical insights, you can clone the repository and experiment with the following examples:

- [MNIST](https://github.com/burn-rs/burn/tree/main/examples/mnist) train a model on CPU/GPU using
  different backends.
- [MNIST Inference Web](https://github.com/burn-rs/burn/tree/main/examples/mnist-inference-web) run
  trained model in the browser for inference.
- [Text Classification](https://github.com/burn-rs/burn/tree/main/examples/text-classification)
  train a transformer encoder from scratch on GPU.
- [Text Generation](https://github.com/burn-rs/burn/tree/main/examples/text-generation) train an
  autoregressive transformer from scratch on GPU.

</details>

<details>
<summary>
Pre-trained Models 
</summary>
<br />

We keep an updated and curated list of models and examples built with Burn, see the [burn-rs/models repository](https://github.com/burn-rs/models) for more details.

</details>

<details>
<summary>
Why use Rust for Deep Learning? 🦀
</summary>
<br />

To us, the main reason to use Rust is when you need to go through multiple abstraction boundaries, without having to pay for performance. A deep learning framework must be easy to use at a high level so its users can concentrate on innovating in the AI field, but since running models relies on heavy computations, performance must be maximized.

To this day, the mainstream solution to this problem has been to offer APIs in Python but rely on bindings to low-level languages such as C/C++.

Rust's approach to abstractions makes it versatile enough to tackle this dichotomy. Indeed, thanks to the borrow-checker, which prevents the programmer from using a variable without explicitly stating if it can be changed or just looked at, Rust is able to provide high-level abstractions for concurrent programming and memory safety guarantees without incurring any runtime overhead.An example of the borrow-checker being directly useful is for [intelligent memory management](#performance).

Rust also comes with the Cargo package manager, which makes it incredibly easy to build, test, and deploy. The latter is usually painstaking in a Python environment.

Although it has the reputation of being a difficult language at first, we strongly believe programming in Rust leads to more reliable, bug-free solutions.

</details>

<details>
<summary>
Contributing
</summary>
<br />

Before contributing, please take a moment to review our
[code of conduct](https://github.com/burn-rs/burn/tree/main/CODE-OF-CONDUCT.md). Please see more details in our [contributing guide](/CONTRIBUTING.md).

</details>

## Status

Burn is currently in active development, and there will be breaking changes. While any resulting
issues are likely to be easy to fix, there are no guarantees at this stage.

## License

Burn is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for details. Opening a pull
request is assumed to signal agreement with these licensing terms.
