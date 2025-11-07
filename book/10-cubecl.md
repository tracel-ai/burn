# Chapter 10: CubeCL

Burn's backend system is designed to be extensible, allowing for the integration of various computation engines. One of the most innovative and powerful backends in the Burn ecosystem is **CubeCL**. This chapter provides an in-depth look at what CubeCL is, how it works, and why it's a significant step forward for writing high-performance, cross-platform GPU kernels.

## What is CubeCL?

CubeCL is a "meta-backend" or a generic backend framework. Unlike `burn-wgpu` or `burn-cuda`, which are tied to specific GPU APIs, CubeCL is a more abstract system designed to **generate and compile GPU kernels just-in-time (JIT)**.

The core idea is to write GPU kernels once in a high-level, Rust-like language, and then have CubeCL compile them down to the specific shader language required by the target runtime (like WGSL for `wgpu`, CUDA C++ for NVIDIA GPUs, or even C++ for CPU execution).

Let's look at the `CubeBackend` struct in `crates/burn-cubecl/src/backend.rs`:

```rust
// crates/burn-cubecl/src/backend.rs

#[derive(new)]
pub struct CubeBackend<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> {
    _runtime: PhantomData<R>,
    _float_elem: PhantomData<F>,
    _int_elem: PhantomData<I>,
    _bool_elem: PhantomData<BT>,
}

impl<R, F, I, BT> Backend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    // ... other bounds
{
    type Device = R::Device;
    type FloatTensorPrimitive = CubeTensor<R>;
    // ...
}
```

### Line-by-Line Analysis:

*   **`pub struct CubeBackend<R: CubeRuntime, ...>`**: The `CubeBackend` is highly generic. The most important parameter is `R: CubeRuntime`.
    *   **`R: CubeRuntime`**: This generic parameter represents the **target runtime**. This is what allows CubeCL to be so flexible. You can plug in a `wgpu` runtime, a `cuda` runtime, or others, and the `CubeBackend` will adapt.
*   **`impl<R, F, I, BT> Backend for CubeBackend<R, F, I, BT>`**: Crucially, `CubeBackend` itself implements the `Backend` trait. This means it can be used as a standard Burn backend, and all the high-level `Tensor` APIs will work with it seamlessly.
*   **`type FloatTensorPrimitive = CubeTensor<R>;`**: The tensor primitive for a `CubeBackend` is a `CubeTensor`. This special tensor type doesn't just hold a handle to a buffer; it's aware of the CubeCL JIT compilation system.

## How CubeCL Works: The JIT Compilation Pipeline

CubeCL's power comes from its ability to translate high-level kernel definitions into optimized code for different hardware.

### 1. High-Level Kernel Definition with the `#[cube]` DSL

You don't write raw WGSL or CUDA. Instead, you define your kernel using a Rust macro-based DSL (Domain-Specific Language). The `#[cube]` procedural macro transforms a subset of Rust syntax into a hardware-agnostic representation.

```rust
// This is a conceptual example
#[cube]
pub fn my_kernel<E: CubeElement>(lhs: &Tensor<E>, rhs: &Tensor<E>, output: &mut Tensor<E>) {
    // Special variables provided by the DSL for thread indexing
    let x_coord = ABSOLUTE_POS_X;
    let y_coord = ABSOLUTE_POS_Y;

    // Standard Rust syntax is used for the kernel logic
    let x = lhs[x_coord];
    let y = rhs[y_coord];
    output[x_coord] = x + y;
}
```
The `#[cube(launch)]` attribute is used to mark the main entry point of a kernel that will be launched from the host. The DSL provides special global variables like `ABSOLUTE_POS_X` to know which part of the tensor the current GPU thread should work on.

### 2. Intermediate Representation (IR)

When a `CubeTensor` operation is called, CubeCL doesn't execute it immediately. Instead, it converts the operation and the `#[cube]` kernel into an **Intermediate Representation (IR)**. This IR is a hardware-agnostic representation of the computation to be performed.

### 3. JIT Compilation

The CubeCL runtime takes this IR and, at runtime, compiles it into the specific shader language of the target `CubeRuntime`.

*   If the runtime is `wgpu`, it generates **WGSL**.
*   If the runtime is `cuda`, it generates **CUDA C++**.

This JIT-compiled kernel is then cached. The next time the same operation (with the same tensor shapes and types) is needed, the cached, pre-compiled kernel is used directly, avoiding the compilation overhead.

### An ASCII Diagram of the CubeCL Workflow

```
+------------------------------------------------------+
|            Your High-Level Burn Operation            |
|              e.g., `tensor_a + tensor_b`             |
+------------------------------------------------------+
                         |
                         V
+------------------------------------------------------+
|        CubeCL captures the operation as an IR        |
|  (Intermediate Representation: a hardware-agnostic   |
|               description of the math)               |
+------------------------------------------------------+
                         |
                         V
+------------------------------------------------------+
|      The `CubeRuntime` (e.g., Wgpu, Cuda) JIT-       |
|          compiles the IR into a shader/kernel        |
+------------------------------------------------------+
          /                       \
         /                         \
        V                           V
+-----------------+         +-----------------+
|   WGSL Shader   |         |   CUDA Kernel   |
|  (for WGPU)     |         |  (for NVIDIA)   |
+-----------------+         +-----------------+
        |                           |
        V                           V
+-----------------+         +-----------------+
|  Execute on GPU |         |  Execute on GPU |
+-----------------+         +-----------------+
```

## Why is CubeCL Important?

*   **Write Once, Run Anywhere**: It allows developers to write custom, high-performance GPU kernels without needing to be experts in multiple shader languages. This drastically improves portability.
*   **Performance**: By defining kernels at a high level, CubeCL can perform powerful, whole-program optimizations during the JIT compilation step that are difficult to do when writing raw shaders.
*   **Extensibility**: It makes it much easier to add support for new hardware to Burn. As long as you can create a `CubeRuntime` that compiles the IR to the new hardware's language, all existing CubeCL kernels will work automatically.

CubeCL represents a sophisticated and forward-thinking approach to GPU computing, providing a powerful abstraction layer that combines the performance of low-level kernels with the portability and ease of use of a high-level framework.

---

## Exercises

1.  **Explore the DSL**: In the `crates/burn-cubecl/src/kernel` directory, find a kernel file (e.g., `add.rs` or `matmul.rs`). Look at the code inside the `#[cube]` macro. What familiar Rust syntax do you see? What parts seem to be specific to the CubeCL DSL?
2.  **Kernel Variables**: The CubeCL DSL provides global variables for thread and block indices, like `ABSOLUTE_POS_X`, `LOCAL_POS_X`, `BLOCK_ID_X`, etc. What is the difference between an `ABSOLUTE_POS` and a `LOCAL_POS`? In what kind of algorithm would this distinction be important? (Hint: Think about operations that require communication between nearby threads, like a convolution).
3.  **Thought Experiment**: CubeCL uses JIT (Just-In-Time) compilation. What are the pros and cons of this approach compared to AOT (Ahead-Of-Time) compilation, where kernels are compiled before the program runs?
