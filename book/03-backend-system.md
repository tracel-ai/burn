# Chapter 3: The Backend System

If the `Tensor` is the heart of Burn, then the `Backend` trait is its soul. It is the core abstraction that makes Burn so flexible and powerful. This chapter explores the `Backend` trait, how it works, and how concrete implementations like `burn-wgpu` and `burn-ndarray` bring it to life.

## The `Backend` Trait: A Contract for Computation

The `Backend` trait, defined in `crates/burn-tensor/src/tensor/backend/base.rs`, is a contract that a computation engine must fulfill to be compatible with Burn. Any struct that implements this trait can be used as the `B` generic parameter in `Tensor<B, D, K>`.

Let's examine the trait's definition:

```rust
// crates/burn-tensor/src/tensor/backend/base.rs

pub trait Backend:
    FloatTensorOps<Self>
    + BoolTensorOps<Self>
    + IntTensorOps<Self>
    + ModuleOps<Self>
    // ... and more Ops traits
    + Clone
    + Sized
    + Send
    + Sync
    + 'static
{
    type Device: DeviceOps;

    type FloatTensorPrimitive: TensorMetadata + 'static;
    type FloatElem: Element;

    type IntTensorPrimitive: TensorMetadata + 'static;
    type IntElem: Element;

    // ... other associated types and functions
}
```

### Deeper Dive into the Trait Structure

*   **`pub trait Backend: ...`**: This defines the `Backend` trait. The long list of traits that follow (`FloatTensorOps<Self>`, `BoolTensorOps<Self>`, etc.) are called **supertraits**. This means that to implement `Backend`, a type must *also* implement all of these other traits. This design follows the **Interface Segregation Principle**, where the overall `Backend` functionality is broken down into smaller, more focused traits.

*   **The `...Ops` Supertraits**: Each of these traits groups a specific category of operations. For example, `FloatTensorOps` defines all the functions that a backend must implement for floating-point tensors.

    ```rust
    // A conceptual look inside FloatTensorOps
    pub trait FloatTensorOps<B: Backend> {
        fn float_add(lhs: B::FloatTensorPrimitive, rhs: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive;
        fn float_matmul(lhs: B::FloatTensorPrimitive, rhs: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive;
        // ... many other functions for float tensors
    }
    ```
    This separation makes the backend implementation more organized and modular. A backend developer can focus on implementing one set of operations at a time.

*   **Associated Types**:
    *   `type Device: DeviceOps;`: Each backend must define a `Device` type (e.g., `WgpuDevice` for the WGPU backend) that represents a specific computational device (like a specific GPU or a CPU).
    *   `type FloatTensorPrimitive: ...;`: This defines the concrete, backend-specific struct that will be used to represent a floating-point tensor. This is the type that will be stored in the `primitive` field of a `Tensor` handle.

### Static Dispatch and Monomorphization

Burn's generic, trait-based approach is what makes it fast. It relies on a process called **monomorphization**. At compile time, the Rust compiler looks at how you've used generic functions and creates a specialized, concrete version of that function for each specific type you've used.

Here is a diagram illustrating this process:

```
Your Generic Code:
+-------------------------------------------+
| fn matmul<B: Backend>(...) {              |
|   let t1 = Tensor::<B, 2>::ones(...);     |
|   let t2 = Tensor::<B, 2>::zeros(...);    |
|   t1.matmul(t2);                          |
| }                                         |
|                                           |
| matmul::<NdArray<f32>>(...);               |
| matmul::<Wgpu>(...);                       |
+-------------------------------------------+
                 |
                 V (Rust Compiler)
                 |
+-------------------------------------------+
|      Monomorphized (Specialized) Code     |
|                                           |
| fn matmul_ndarray(...) { // For NdArray   |
|   // Calls to ndarray's `ones`, `zeros`,  |
|   // `matmul` implementations are inlined.|
| }                                         |
|                                           |
| fn matmul_wgpu(...) { // For Wgpu          |
|   // Calls to wgpu's `ones`, `zeros`,     |
|   // `matmul` implementations are inlined.|
| }                                         |
+-------------------------------------------+
```
This process, known as **static dispatch**, means there is no runtime overhead for figuring out which backend's function to call. The decision is made at compile time, and the resulting machine code is just as fast as if you had written the backend-specific code yourself.

## Concrete Backends: `wgpu` and `ndarray`

Burn comes with several built-in backends. Let's briefly look at two of the most important ones.

### `burn-wgpu`: The Cross-Platform GPU Backend

*   **Purpose**: This is Burn's primary GPU backend. It uses the `wgpu` library, which is a Rust implementation of the WebGPU API.
*   **How it works**: `wgpu` can target multiple graphics APIs, including Vulkan, Metal, DirectX 12, and even OpenGL. This means that the *same* `burn-wgpu` backend can run on Windows, macOS, Linux, and even in the browser via WebAssembly. It is often used in conjunction with CubeCL (see Chapter 10) for JIT compilation of kernels.
*   **When to use it**: This should be your default choice for training models or running inference on a GPU, as it provides the best combination of performance and portability.

### `burn-ndarray`: The CPU Backend

*   **Purpose**: This backend is for running computations on the CPU.
*   **How it works**: It uses the `ndarray` crate, which is a popular and highly optimized library for numerical computing in Rust, similar to NumPy in Python.
*   **When to use it**:
    *   When you don't have a supported GPU.
    *   For tasks that are not computationally intensive enough to benefit from a GPU.
    *   In `no_std` environments for embedded devices, where a GPU backend is not an option.

---

## Exercises

1.  **Code Exploration**: Find the `FloatTensorOps` trait definition within the `burn-tensor` crate. What are three different mathematical operations defined in this trait?
2.  **Conceptual Design**: Imagine you want to add support for tensors with `f16` (half-precision floating-point) numbers to Burn. What changes or additions might you need to make to the `Backend` trait to support this new data type?
3.  **Practical Application**: Write a small Rust program with a generic function that takes a `Backend` type `B`, creates a 2x2 tensor of ones on the default device for that backend, and then prints the tensor. In your `main` function, call this generic function with both `burn_ndarray::NdArrayBackend<f32>` and another backend of your choice (if your hardware supports it, like `burn_wgpu::WgpuBackend`).
