# Burn Backend Extension

> [Burn](https://github.com/tracel-ai/burn) backend extension generation

[![Current Crates.io Version](https://img.shields.io/crates/v/burn-backend-extension.svg)](https://crates.io/crates/burn-backend-extension)
[![license](https://shields.io/badge/license-MIT%2FApache--2.0-blue)](https://github.com/tracel-ai/burn-backend-extension/blob/master/README.md)

`#[backend_extension(Cube, Fusion: cfg(feature = "fusion"))]` generates both dispatch and lazy
Fusion implementations. Each method must choose a Fusion behavior:

- `#[fusion(dtype = lhs, shape = lhs)]` describes a single tensor output using field expressions.
- `#[fusion(meta = callable)]` computes output metadata from borrowed tensor specs and borrowed
  ordinary arguments, then registers an opaque deferred operation.
- `#[fusion(default)]` inherits the method's existing default body.

For single outputs, both `dtype` and `shape` are required. Tensor argument names refer to `DType`
values in `dtype` and borrowed `Shape` values in `shape`. A bare tensor name copies its shape;
other shape expressions return an owned `Shape`. Ordinary arguments are borrowed in both expressions.
For example, a shape function can be shared with the backend implementation:

```rust,ignore
#[fusion(dtype = lhs, shape = output_shape(lhs, rhs))]
fn matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self>;
// output_shape takes (&Shape, &Shape) and returns Shape.
```

Inline calculations can use a block:

```rust,ignore
#[fusion(dtype = input, shape = {
    let mut shape = input.clone();
    shape.swap(0, 1);
    shape
})]
fn transpose_2d(input: FloatTensor<Self>) -> FloatTensor<Self>;
```

The fields expect values, so `shape = |input| ...` is not invoked automatically. Use a block
or a function call. Field expressions cannot be combined with `meta` or `default`.

Use `meta` for tuples, structs, and enums, or to compute shape and dtype together. It accepts
function paths or inline closures receiving borrowed arguments in declaration order. Tensor
arguments become `burn::backend::fusion::custom::TensorSpec`; extension arguments become their
companion metadata. Results mirror the output structure.

Opt into `#[derive(ExtensionType)] #[extension_type(fusion)]` (or `fusion: cfg(...)`) for structured
inputs and outputs. The generated `NameMetadata` mirrors fields and variants, replacing tensors
with specs. Mark nested fields and method arguments with `#[extension_type]`. Nested metadata
resolves through the field type, including imported or renamed types. Ordinary fields are cloned
and require `Clone + Debug + PartialEq`; captured metadata must also be `Send + Sync + 'static`.
A callback such as `meta = |cache| cache.clone()` preserves a cache's variant and tensor layout.

The operation ID defaults to the method name; use `id = "custom_matmul"` to override it.
Integer parameters (8–64 bits, `usize`, `isize`), `f32`, `f64`, and `bool` are exposed to custom
optimizers automatically. Mark an alias or a type convertible to `burn::backend::Scalar` with
`#[fusion(scalar)]`, or provide an encoding on the parameter:

```rust,ignore
#[fusion(dtype = lhs, shape = output_shape(lhs, rhs))]
fn matmul(
    lhs: FloatTensor<Self>,
    rhs: FloatTensor<Self>,
    #[fusion(scalar = strategy.to_code())] strategy: MatmulStrategy,
) -> FloatTensor<Self>;
```

The encoding returns a value convertible to `Scalar`. Inferred and marked scalars are recorded
in parameter declaration order; the backend still receives the original arguments. Other ordinary
arguments and extension fields are captured for execution only. Tensor contents are not exposed.

Lazy methods must be synchronous and non-generic, with at least one input tensor, directly or
inside an owned extension value. Primitive inputs may be immutably borrowed. Ordinary arguments
must be owned and `Clone + Send + Sync + 'static`. Output metadata, including enum variants and
ordinary fields, must be known before execution. Use an existing default body for other signatures,
or omit `Fusion` from `#[backend_extension]` and implement the trait for `Fusion<B>` manually.
Wrong dtype categories are rejected before registration; outputs that disagree with metadata
fail at execution before any handles are published. Wrappers do not read tensor data or drain queues.

Fusion generation does not generate gradients or merge custom kernels with neighboring kernels.
Handwritten autodiff implementations continue to compose with the generated wrapper.
