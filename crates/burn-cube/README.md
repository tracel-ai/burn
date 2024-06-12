<div align="center">
<img src="../burn-cube/assets/CubeCL.webp" width="150px"/>
<span style="font-size:64px;font-weight:bold">CubeCL</span>

[![Rust Version](https://img.shields.io/badge/Rust-1.75.0+-blue)](https://releases.rs/docs/1.75.0)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)

---

**Multi-platform high-performance compute language for Rust.**
<br/>

</div>

<div align="left">

## TL;DR

With CubeCL, you can use Rust to program your GPU.

## Motivation

The goal of CubeCL is to ease the pain of writing highly optimized compute kernels that are portable across hardwares.
There is currently no adequate solution when you want optimal performance while still being multi-platform.
You either have to write custom kernels for different hardwares, often with different languages such as CUDA, Metal or ROCm.
To make it possible, we created a Just-in-Time compiler with three core features: **automatic vectorization**, **comptime** and **autotune**!

Those features are extremely useful for anyone writing high-performance kernels, even when portability is not a concern.
It improves code composability, reusability, maintainability, all while staying optimal.

### Disclaimer & History

CubeCL is currently in **alpha**.
The only supported runtimes are CUDA and WebGPU for now.
It's easy to add more GPU runtimes and we intend to support Metal, ROCm and Vulkan; contributions are welcomed!
We also want to have an optimized JIT CPU runtime with SIMD instructions, leveraging Cranelift.

While CubeCL is currently in use in Burn, the user experience, such as error messages and other edge cases aren't graciously handled.
The project started as a WebGPU-only backend for Burn.
As we optimized it, we realized that we needed an intermediate representation (IR) that could be optimized then compiled to WGSL.
Having an IR made it easy to support another compilation target, so we made a CUDA runtime.
However, writing kernels directly in that IR wasn't easy, so we created a Rust frontend using the `syn` crate.
Navigating the differences between CUDA and WebGPU, while leveraging both platforms, forced us to come up with general concepts that worked everywhere.
Hence, CubeCL was born!

## Design

CubeCL is desied around, you guessed it, Cubes! We chose this since all compute API needs to map to the hardware, which is always tiles that can be access using a 3D representation: the cube.
You can easily map the concepts to other APIs.

| CubeCL       | Description                                                 |
| ------------ | ----------------------------------------------------------- |
| CUBE_COUNT   | The number of cubes launched.                               |
| CUBE_POS     | The cube position in all launched cubes.                    |
| CUBE_DIM     | The total amont of working unit in a cube.                  |
| UNIT_POS     | The position of the working unit inside a cube.             |
| SUBCUBE_DIM  | The total amount of working units in a subcube.             |
| ABSOLUTE_POS | The position of the working unit without regards for cubes. |

## Special Features

#### Automatic Vectorization

When you can use SIMD instructions, you should, but it can get pretty complicate pretty fast!
With CubeCL you can specified the vectorization factor of each input variable when launching a kernel.
There is only one type that is dynamically vectorized that supports automatic broadcasting.
The runtimes are able to compile kernels and have all information necessary to use the best instruction!
However, often you actually need to know the vectorization factor in your algorithm, and you can actually access it directly in the kernel when you need it without any performance loss: it's going to use the comptime system!

#### Comptime

CubeCL isn't just a new compute language, it feels like you are writing GPU kernels, but in fact you are writing compiler plugins that you can fully customize!
Comptime is a way to modify the compiler internal representation (IR) at runtime, when compiling a shader for the first time.

This allows lots of optimizations and flexibility without having to write 10 different variants of the same kernels for max performance.

| Feature                        | Description                                                                                                                                                             |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Instruction Specialisation** | Not all instrustion are available on all hardware, but when a specialized instruction exist, should be able to activate them with a simple if statement.                |
| **Automatic Vectorization**    | When you can use SIMD instructions, you should! But not all hardware support the same vectorization factor, so you can inject the vectorization factor as runtime!      |
| **Loop Unrolling**             | You may want multiple different flavor of the same kernels with loop Unrolling for only a certain range of values, you can do it at runtime easily with Comptime        |
| **Shape Specialisation**       | For deep learning kernels, it's often crucial to have different kernels for different size, you can do it by passing the shape information as comptime values.          |
| **Compile Time Calculation**   | In general, you can calculate a constant using runtime properties and inject them into a kernel during compilation, which avoid recalculating it during each execution. |

#### Autonuning

Autotuning drastically simplify kernel selection by running small benchmarks at runtime to figure out the best kernels with the best configurations to run on the current hardware, essential for portability.
This feature combines gracefully with comptime, since you can test the effect of different comptime on performance, sometime you can get surprised!
Even if the benchmarks may takes some time to run when first running the application, the information gets cache on the hardware.
It is often a no-brainer traideoff when you build throughput oriented programs such as deep learning models.
You can ship the autotune cache with your program, reducing cold start when you have more control over the deployment platform.

## Example

CubeCL is designed to be easy to use for Rust programmers by using the same syntax and being fully integrated with the language.
You can simply add an attribute on the top of a Rust function and then be executed on the GPU.

```rust
#[cube(launch)]
fn pow2<F: Float>(input: Tensor<F>, mut output: Tensor<F>) {
    if ABSOLUTE_POS < input.shape(0) {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * input[ABSOLUTE_POS];
    }
}

fn main() {
    type Runtime = CudaRuntime;

    let device = Default::default();
    let client = Runtime::client(&device);

    let input: &[f32] = &[1.0, 2.0, 3.0, 4.0];
    let (shape, strides) = ([4], [1]);

    let input_handle = client.create(f32::as_bytes(input));
    let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());

    runs_on_the_gpu_launch::<F32, Runtime>(
        client,
        KernelSettings::default(),
        TensorHandle::new(&input_handle, &strides, &shape),
        TensorHandle::new(&output_handle, &strides, &shape),
    );

    let output = client.read(output_handle.binding()).read_sync().unwrap();
    let output = f32::from_bytes(&output);

    assert_eq!(output, &[1.0, 4.0, 9.0, 16.0]);
}

```

The `cube` attributes generate the code that is needed to compile a kernel.
In the case above, the function `pow2_expand` and `pow2_launch` are automatically generated.
By generating the `expand` version as well, you can easily compose your functions together easily.

```rust

#[cube]
fn pow2_scalar<F: Float>(x: F) -> F {
   x * x
}

#[cube(launch)]
fn pow2<F: Float>(input: Tensor<F>, mut output: Tensor<F>) {
    if ABSOLUTE_POS < input.shape(0) {
        output[ABSOLUTE_POS] = pow2_scalar(input[ABSOLUTE_POS]);
    }
}
```

Note that you don't have to specify `launch` in a function that is only used by another one.
In addition, you can have return types without problem, which isn't the case when you are writing an entry point to a kernel using the `launch` attribute.
The function `pow2_expand` will actually use `pow2_scalar_expand`, making it easy to combine your function.

### Custom Type

You can esily create custom types using the `CybeType` and `CubeLaunch` derives.

```rust
/// To use when launching a kernel.
#[derive(CubeLaunch)]
struct LaunchArguments {
    conv_stride: UInt,
    dilation: UInt,
    padding: UInt,
    groups: UInt,
}

/// When only used in functions.
#[derive(CubeType)]
struct KernelState {
    input: Tensor<F32>,
    index: UInt,
    accumulator: F32,
}
```
