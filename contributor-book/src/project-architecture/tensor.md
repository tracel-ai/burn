# Tensor

A proper deep learning framework should have a fast tensor implementation with autodiff support, and
Burn is no exception. The tensor API abstracts away backend implementation details and focuses on
usability without compromising performance. To make it as easy as possible to use, there is only one
tensor type, which is different from multiple tensor and deep learning crates in Rust. Generic
parameters are used instead to specialize the tensor type.

- **B: Backend:** The first argument is the backend on which the tensor implementation lies.
- **const D: usize:** The second argument is the dimensionality of the tensor.
- **K: TensorKind:** The third argument is the tensor kind, which can be either Float, Int or Bool.
  By default, the tensor kind is set to Float, so for most tensors, the kind argument is not
  necessary.

Having one struct for tensors reduces the complexity of the tensor API, which also means less
duplicated documentation to write and maintain.

Tensors are thread-safe, which means that you can send a tensor to another thread, and everything
will work, including auto-differentiation. Note that there are no explicit in-place tensor
operations since all tensor operations take owned tensors as parameters, which make it possible to
mutate them. Tensors can be shared simply by cloning them, but if there is only one reference to a
tensor, the backend implementation is free to reuse the tensor's allocated data. For more
information about how it is done, you can have a look at this
[blog post](https://burn.dev/blog/burn-rusty-approach-to-tensor-handling).

## Tensor Operations

Operations on Tensors (sometimes shortened to Ops) are defined in traits (generally part of the
Backend Supertrait) and implemented for the Tensor struct. The appropriate parent trait of an
operation depends on the type of operation:

- `base` => All tensor kinds should implement these operations (reshape, into_data, etc.). The
  implementation is in
  [crates/burn-tensor/src/tensor/api/base.rs](https://github.com/tracel-ai/burn/blob/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/api/base.rs).
- `numeric` => All tensors that are numeric by nature should implement these operations (Add, Sub,
  Div, etc.). The implementation is in
  [crates/burn-tensor/src/tensor/api/numeric.rs](https://github.com/tracel-ai/burn/blob/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/api/numeric.rs).
- `Float` => Tensor operations are only available for float tensors. The implementation is in
  [burn-tensor/src/tensor/api/float.rs](https://github.com/tracel-ai/burn/blob/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/api/float.rs).
- `Int` => Tensor operations are only available for int tensors. The implementation is in
  [burn-tensor/src/tensor/api/int.rs](https://github.com/tracel-ai/burn/blob/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/api/int.rs).
- `bool` => Tensor operations are only available for bool tensors. The implementation is in
  [burn-tensor/src/tensor/api/bool.rs](https://github.com/tracel-ai/burn/blob/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/api/bool.rs).

`Numeric` is directly implemented for `Float` and `Int` tensors, and in general, The implementations
for these methods are calling the corresponding `{Int|Float}` method defined in the backend
supertrait.

Anything that is implemented by numeric should have an implementation in the `{Int|Float}` traits,
though it may be avoidable if the operation for one type requires casting to the other type. To
provide an example, `powf` should be implemented for `Int` tensors, but it should not be an Int
Tensor Operation. The LHS should be converted to a float, and the output should be converted back to
an int. So it's possible to avoid implementing `IntTensorOp` altogether.

Additionally there are some operations that should be defined as functions instead of tensor op
methods. These are:

`module` => These should be exported as functions instead of methods on tensors. The implementation
is in
[crates/burn-tensor/src/tensor/ops/module.rs](https://github.com/tracel-ai/burn/tree/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/ops/modules).
`activation` => These should also be exported as functions instead of methods on tensors. The
implementation is in
[crates/burn-tensor/src/tensor/ops/activation.rs](https://github.com/tracel-ai/burn/blob/6d96e8d8086d2309c425f2c8a43a8246f8c454d2/crates/burn-tensor/src/tensor/ops/activation.rs).
Note that some activations are just a combination of backend operations and are not declared in
there.
