# Backend Extension

Burn aims to be the most flexible deep learning framework. While it's crucial to maintain
compatibility with a wide variety of backends, Burn provides the ability to extend the functionality
of a backend implementation to suit your modeling requirements. This versatility is advantageous in
numerous ways, such as supporting custom operations like flash attention or manually fusing
operations for enhanced performance.

In this section, we will go into the process of extending a backend, providing multiple examples.
But before we proceed, let's establish the fundamental principles that will empower you to craft
your own backend extensions.

Burn's user-facing tensors and modules are runtime-dispatched and don't expose a backend generic.
Backend traits remain part of the lower layer, where they define primitive operations that can
be registered with the Tensor → Bridge → Dispatch → Backend stack. To create an extension, define a
backend trait specifying the new primitive operation, implement it for the backends you support,
and expose a backend-independent `Tensor` function that calls through `Dispatch`.

## Registering an operation

Enable the `extension` feature of `burn`, the backend features you target, and `autodiff` if you
support training. This small example uses `extension`, `flex`, and `autodiff`:

```rust,ignore
use burn::{
    backend::{
        Autodiff, Dispatch, Flex, backend_extension,
        autodiff::checkpoint::strategy::CheckpointStrategy,
        tensor::FloatTensor,
    },
    tensor::{Device, Tensor},
};

#[backend_extension(Autodiff, Flex)]
pub trait SquareBackend: burn::backend::Backend {
    fn square(input: FloatTensor<Self>) -> FloatTensor<Self> {
        Self::float_mul(input.clone(), input)
    }
}

impl SquareBackend for Flex {}

// The default body composes differentiable primitives, so it also supplies the backward pass.
impl<B: SquareBackend, C: CheckpointStrategy> SquareBackend for Autodiff<B, C> {}

pub fn square<const D: usize>(input: Tensor<D>) -> Tensor<D> {
    Tensor::from_dispatch(Dispatch::square(input.into_dispatch()))
}

let device = Device::flex().autodiff();
let input = Tensor::<1>::from_floats([2.0, 3.0], &device).require_grad();
let gradients = square(input.clone()).sum().backward();
let gradient = input.grad(&gradients).unwrap();
assert_eq!(gradient.try_into_vec_as::<f32>().unwrap(), vec![4.0, 6.0]);
```

The macro generates the implementation of your trait for `Dispatch`. It does not implement the
trait for concrete backends or derive a custom kernel's backward pass. A default trait body can
compose existing differentiable operations, as above. To optimize the forward or backward pass,
override that body on the concrete backend or `Autodiff<B, C>` respectively. These are alternative
implementations; do not add overlapping implementations of the same trait.

Execution backend selectors include `Cube`, `Flex`, `NdArray`, `LibTorch`, and `Remote`, plus the
`Autodiff` routing option. `Cube` covers the CubeCL runtimes; `Wgpu` and `Cuda` are backend aliases,
not accepted selectors. A selector may have a condition such as
`Cube: cfg(feature = "wgpu")`. Conditions refer to features of the crate containing the macro;
forward those features to the corresponding Burn dependency features.

Listing a selector requires an implementation for that backend when the condition is enabled.
Calling the extension on an unlisted runtime backend panics. Dispatch does not transfer operands
between devices, and a `Cube` implementation using a WGSL kernel is still limited to a compatible
WGPU runtime/compiler even though `Cube` also represents CUDA and other runtimes.

Autodiff contexts from tensor-bearing inputs are merged. Plain inputs act as constants; enabled
inputs must use the same checkpointing strategy. Inputs still need compatible devices and dtypes
for the operation. Add shape and dtype validation in your public wrapper or kernel implementation;
the macro does not infer your operation's mathematical constraints.

## Fusion

When Burn's `fusion` feature is enabled, the `Cube` dispatch backend uses
`Fusion<CubeBackend>`. Add `Fusion` to `#[backend_extension(...)]` to generate the
extension implementation for `Fusion<B>`, and choose a behavior for each method:

- `#[fusion(dtype = input, shape = input)]` describes a single tensor output.
  A bare operand copies its dtype or shape; a helper such as
  `shape = output_shape(lhs, rhs, bias)` receives borrowed shapes and returns an owned `Shape`.
- `#[fusion(meta = callable)]` describes tensor tuples or structured outputs using a callback.
  Tensor arguments arrive as borrowed `TensorSpec` values; structured arguments arrive as
  borrowed extension metadata. Derive `ExtensionType` with `#[extension_type(fusion)]`
  on structs and enums used by the generated Fusion implementation.
- `#[fusion(default)]` uses the method's existing default body instead of registering a custom
  operation. This is useful for bodies that compose existing backend operations.

Choose exactly one form per method. `Fusion` can be conditional, for example
`Fusion: cfg(feature = "fusion")`, using a feature of your extension crate forwarded to Burn.
Omit the `Fusion` selector if you implement the extension for `Fusion<B>` yourself.

For field expressions and `meta`, output metadata is computed when the call is registered;
the concrete backend runs later. Shapes, dtypes, enum variants, and non-tensor output fields
must be knowable without reading tensor contents and must match direct backend execution.
For structured outputs, metadata supplies the **actual non-tensor return values**. Fusion returns
them immediately and discards the backend's later values without comparison. Return
content-dependent values as tensors, or provide a handwritten Fusion implementation that waits
for execution. Output shapes are also not compared against backend results, even in debug builds.

Fusion generation supplies lazy registration, not derivatives or automatic merging of custom
kernels with neighboring operations. A custom optimizer can recognize their IR. Keep the
handwritten `Autodiff<B, C>` implementation for a custom backward pass, or compose differentiable
primitives in a default body as in the Flex example above.

The [CubeCL tutorial](./custom-cubecl-kernel.md) demonstrates a portable kernel, generated Fusion
registration, and a handwritten backward pass. The [WGPU tutorial](./custom-wgpu-kernel.md)
demonstrates a WGSL source kernel. The
[macro reference source](https://github.com/tracel-ai/burn/blob/main/crates/burn-backend-extension/src/lib.rs)
describes supported signatures, metadata validation, operation IDs, and scalar encodings.

## Passing structs and enums of tensors

An extension operation is not limited to individual tensor arguments. A custom struct or enum whose
fields are tensor primitives can be passed to and returned from an operation by deriving
`ExtensionType`. Fields that are not tensors pass through unchanged, and a field that is itself an
`ExtensionType` can be nested by annotating it with `#[extension_type]`.

```rust, ignore
use burn::backend::{
    ExtensionType, backend_extension,
    tensor::{FloatTensor, IntTensor},
};

#[derive(ExtensionType)]
pub struct Boxes<B: Backend> {
    pub coords: FloatTensor<B>,
    pub scores: FloatTensor<B>,
    pub count: usize, // Non-tensor fields pass through unchanged.
}

#[derive(ExtensionType)]
pub enum Operand<B: Backend> {
    Dense(FloatTensor<B>),
    Sparse { values: FloatTensor<B>, indices: IntTensor<B> },
    Empty,
}
```

Such a type can be returned from an operation directly. To pass one as an input, mark the argument
with `#[extension_type]`:

```rust, ignore
#[backend_extension(Cube, Autodiff)]
pub trait Backend: burn::backend::Backend {
    // Struct as an output.
    fn detect(image: FloatTensor<Self>) -> Boxes<Self>;

    // Struct or enum as an input.
    fn nms(#[extension_type] boxes: Boxes<Self>, iou_threshold: f32) -> Boxes<Self>;
}
```

Inputs marked this way can be freely mixed with plain tensor arguments and with each other, and an
operation can take several of them. The backend is selected by looking at a routing tensor
across the inputs, so an enum currently on a variant that holds no tensor simply defers to the next
input; if no input holds a tensor at all, the backend cannot be resolved and the operation panics.

Struct and enum inputs also work with `Autodiff`. Float fields carry the gradient; other fields do
not. Your `impl ... for Autodiff<B, C>` writes the backward pass by hand, exactly as it does for plain
tensor inputs.
