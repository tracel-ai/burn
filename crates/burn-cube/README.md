<div align="center">
<img src="../burn-cube/assets/logo.drawio.svg" width="400px"/>

<br />
<br />

[![Rust Version](https://img.shields.io/badge/Rust-1.75.0+-blue)](https://releases.rs/docs/1.75.0)
![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)

---

**Multi-platform high-performance compute language extension for Rust.**
<br/>

</div>

## TL;DR

With CubeCL, you can program your GPU using Rust leveraging zero-cost abstraction to create maintainable, flexible and optimal compute kernels.

## Motivation

The goal of CubeCL is to ease the pain of writing highly optimized compute kernels that are portable across hardware.
There is currently no adequate solution when you want optimal performance while still being multi-platform.
You either have to write custom kernels for different hardware, often with different languages such as CUDA, Metal, or ROCm.
To fix this, we created a Just-in-Time compiler with three core features: **automatic vectorization**, **comptime**, and **autotune**!

These features are extremely useful for anyone writing high-performance kernels, even when portability is not a concern.
They improve code composability, reusability, testability, and maintainability, all while staying optimal.

### Disclaimer & History

CubeCL is currently in **alpha**.
The only supported runtimes are CUDA and WebGPU for now.
It's easy to add more GPU runtimes and we intend to support Metal, ROCm, and Vulkan; contributions are welcome!
We also want to have an optimized JIT CPU runtime with SIMD instructions, leveraging [Cranelift](https://cranelift.dev).

While CubeCL is currently in use in [Burn](https://burn.dev), there are still a lot of rough edges; it isn't refined yet.
The project started as a WebGPU-only backend for Burn.
As we optimized it, we realized that we needed an intermediate representation (IR) that could be optimized then compiled to WGSL.
Having an IR made it easy to support another compilation target, so we made a CUDA runtime.
However, writing kernels directly in that IR wasn't easy, so we created a Rust frontend using the [syn](https://github.com/dtolnay/syn) crate.
Navigating the differences between CUDA and WebGPU, while leveraging both platforms, forced us to come up with general concepts that worked everywhere.
Hence, CubeCL was born!

## Design

CubeCL is designed around - you guessed it - Cubes! More specifically, it's based on cuboids, because not all axes are the same size.
Since all compute APIs need to map to the hardware, which are tiles that can be accessed using a 3D representation, our topology can easily be mapped to concepts from other APIs.

<div align="center">

### CubeCL - Topology

<img src="./assets/cubecl.drawio.svg" width="100%"/>
<br />
</div>
<br />

_A cube is composed of units, so a 3x3x3 cube has 27 units that can be accessed by their positions along the x, y, and z axes.
Similarly, a hyper-cube is composed of cubes, just as a cube is composed of units.
Each cube in the hyper-cube can be accessed by its position relative to the hyper-cube along the x, y, and z axes.
Hence, a hyper-cube of 3x3x3 will have 27 cubes.
In this example, the total number of working units would be 27 x 27 = 729._

<details>
<summary>Topology Equivalence 👇</summary>
<br />

Since all topology variables are constant within the kernel entry point, we chose to use the Rust constant syntax with capital letters.
Often when creating kernels, we don't always care about the relative position of a unit within a cube along each axis, but often we only care about its position in general.
Therefore, each kind of variable also has its own axis-independent variable, which is often not present in other languages, except WebGPU with `local_invocation_index`.

<br />

| CubeCL         | CUDA        | WebGPU                 |
| -------------- | ----------- | ---------------------- |
| CUBE_COUNT     | N/A         | N/A                    |
| CUBE_COUNT_X   | gridDim.x   | num_workgroups.x       |
| CUBE_COUNT_Y   | gridDim.y   | num_workgroups.y       |
| CUBE_COUNT_Z   | gridDim.z   | num_workgroups.z       |
| CUBE_POS       | N/A         | N/A                    |
| CUBE_POS_X     | blockIdx.x  | workgroup.x            |
| CUBE_POS_Y     | blockIdx.y  | workgroup.y            |
| CUBE_POS_Z     | blockIdx.z  | workgroup.z            |
| CUBE_DIM       | N/A         | N/A                    |
| CUBE_DIM_X     | blockDim.x  | workgroup_size.x       |
| CUBE_DIM_Y     | blockDim.y  | workgroup_size.y       |
| CUBE_DIM_Z     | blockDim.z  | workgroup_size.z       |
| UNIT_POS       | N/A         | local_invocation_index |
| UNIT_POS_X     | threadIdx.x | local_invocation_id.x  |
| UNIT_POS_Y     | threadIdx.y | local_invocation_id.y  |
| UNIT_POS_Z     | threadIdx.z | local_invocation_id.z  |
| SUBCUBE_DIM    | warpSize    | subgroup_size          |
| ABSOLUTE_POS   | N/A         | N/A                    |
| ABSOLUTE_POS_X | N/A         | global_id.x            |
| ABSOLUTE_POS_Y | N/A         | global_id.y            |
| ABSOLUTE_POS_Z | N/A         | global_id.z            |

</details>

## Special Features

#### Automatic Vectorization

High-performance kernels should rely on SIMD instructions whenever possible, but doing so can quickly get pretty complicated!
With CubeCL, you can specify the vectorization factor of each input variable when launching a kernel.
Inside the kernel code, you still use only one type, which is dynamically vectorized and supports automatic broadcasting.
The runtimes are able to compile kernels and have all the necessary information to use the best instruction!
However, since the algorithmic behavior may depend on the vectorization factor, CubeCL allows you to access it directly in the kernel when needed, without any performance loss, using the comptime system!

#### Comptime

CubeCL isn't just a new compute language: though it feels like you are writing GPU kernels, you are, in fact, writing compiler plugins that you can fully customize!
Comptime is a way to modify the compiler IR at runtime when compiling a kernel for the first time.

This enables lots of optimizations and flexibility without having to write many separate variants of the same kernels to ensure maximal performance.

| Feature                        | Description                                                                                                                                                                 |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Instruction Specialization** | Not all instructions are available on all hardware, but when a specialized one exists, it should be enabled with a simple if statement.                                     |
| **Automatic Vectorization**    | When you can use SIMD instructions, you should! But since not all hardware supports the same vectorization factors, it can be injected at runtime!                          |
| **Loop Unrolling**             | You may want multiple flavors of the same kernel, with loop unrolling for only a certain range of values. This can be configured easily with Comptime.                      |
| **Shape Specialization**       | For deep learning kernels, it's often crucial to rely on different kernels for different input sizes; you can do it by passing the shape information as Comptime values.    |
| **Compile Time Calculation**   | In general, you can calculate a constant using Rust runtime properties and inject it into a kernel during its compilation, to avoid recalculating it during each execution. |

#### Autotuning

Autotuning drastically simplifies kernel selection by running small benchmarks at runtime to figure out the best kernels with the best configurations to run on the current hardware; an essential feature for portability.
This feature combines gracefully with comptime to test the effect of different comptime values on performance; sometimes it can be surprising!

Even if the benchmarks may add some overhead when running the application for the first time, the information gets cached on the device and will be reused.
It is usually a no-brainer trade-off for throughput-oriented programs such as deep learning models.
You can even ship the autotune cache with your program, reducing cold start time when you have more control over the deployment target.

## Example

CubeCL is designed to be easy to use for Rust programmers: it relies on the same syntax and is fully integrated with the language.
You can simply add an attribute on the top of a Rust function for it to be executed on the GPU.

```rust
#[cube(launch)]
fn gelu<F: Float>(input: &Array<F>, output: &mut Array<F>) {
    if ABSOLUTE_POS < input.len() {
        let x = input[ABSOLUTE_POS]
        let gelu = x * (1 + erf(x / sqrt(2))) / 2;
        output[ABSOLUTE_POS] = gelu;
    }
}

fn main() {
    type Runtime = CudaRuntime;

    let device = Default::default();
    let client = Runtime::client(&device);

    let input_handle = client.create(f32::as_bytes(&[-1., 0., 1., 5.]));
    let output_handle = client.empty(input.len() * core::mem::size_of::<f32>());

    gelu::launch::<F32, Runtime>(
        client,
        CubeCount::new(1, 1, 1),
        CubeDim::new(4, 1, 1),
        &input_handle,
        &output_handle,
    );

    let output = client.read(output_handle.binding()).read_sync().unwrap();
    let output = f32::from_bytes(&output);

    // Should be [-0.1587,  0.0000,  0.8413,  5.0000]
    println!("{output:?}");
}

```

The `cube` attribute generates the code that is needed to compile a kernel.
In the case above, the function `gelu_expand` and `gelu_launch` are automatically generated.
This allows you to compose Cube functions easily:

```rust

#[cube]
fn gelu_scalar<F: Float>(x: F) -> F {
   x * (1 + erf(x / sqrt(2))) / 2
}

#[cube(launch)]
fn gelu<F: Float>(input: Array<F>, mut output: Array<F>) {
    if ABSOLUTE_POS < input.shape(0) {
        output[ABSOLUTE_POS] = gelu_scalar::<F>(input[ABSOLUTE_POS]);
    }
}
```

Note that you don't have to specify `launch` in a function that is only used by another Cube function.
In addition, you can have return types without problem, which isn't the case when you are writing an entry point to a kernel using the `launch` attribute.
The function `gelu_expand` will actually use `gelu_scalar_expand`, making it easy to combine your functions.

## Resource

If you have any questions or want to contribute, don't hesitate to join the [Discord](https://discord.gg/uPEBbYYDB6).
