# Adding a new operation to burn

Let's discuss how one might go about adding new operators to Burn, using the example of the pow
operator added in [this PR](https://github.com/tracel-ai/burn/pull/1133/files). In that PR, the
following things took place (albeit not in this order).

## Adding the Op to burn-tensor

`burn-tensor` is the crate that defines all tensor operations that need to be implemented by the
various backends. The core of this lies in `crates/burn-tensor/src/tensor/api/numeric.rs`, which is
home to the numeric trait and its implementation for the different tensor types. The numeric trait
is the home of all tensor operations that are numeric in nature and that are shared by `Int` and
`Float` Tensor types. More information on the relationship between Tensor modules can be found under
the section for [Tensor Architecture](../project-architecture/Tensor.md#tensorops).

Here is where pow was added to `crates/burn-tensor/src/tensor/api/numeric.rs`:

1. for the
   [`Tensor<Backend,Dimension,Kind>` struct](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tensor/src/tensor/api/numeric.rs#L565)
2. for the
   [numeric trait](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tensor/src/tensor/api/numeric.rs#L1922)
3. for the implementation of numeric for
   [float](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tensor/src/tensor/api/numeric.rs#L2677)
   and
   [int](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tensor/src/tensor/api/numeric.rs#L2336)

Tensor is a struct that has a single member: `primitive` (defined
[here](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tensor/src/tensor/api/base.rs#L27)),
that is defined by it's
[`Kind`](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tensor/src/tensor/api/kind.rs#L16):
one of `Bool`, `Float`, or `Int` (those linked in 3). These call the ops for that data type defined
in the
[`Backend`](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tensor/src/tensor/backend/base.rs#L54)
supertrait[^supertrait]. This is the trait that is then implemented by the different `burn-`
backends (such as `burn-ndarray` and `burn-wgpu`) which implement the functions if no default is
provided.

In this case, we don't need to worry about `Bool` Tensors. Ops for `Float` is implemented under
[burn-tensor/src/tensor/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tensor/src/tensor/ops/tensor.rs#L977),
and for `Int` under
[`burn-tensor/src/tensor/ops/int_tensor.rs`](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tensor/src/tensor/ops/int_tensor.rs#L539).
The current convention is ops of each type, if not unique to that type, are prefixed with the type.
So `powf` and sundry would be defined as `int_powf` for `IntTensorOps` and `float_powf` for
`FloatTensorOps`. If an op is unique to a type, then it should be implemented under
`burn-tensor/src/api/{type}.rs`. For example, here is an implementation for
[`sin` under `burn-tensor/src/api/float.rs`](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tensor/src/tensor/api/float.rs#L82)
which obviously doesn't make sense for `Int` or `Bool` tensors.

The `Int` Tensor function uses the ones defined for Float with 2 extra casts (LHS to a `Float`
tensor, Output to an `Int`). Given that the rest of the code will only look at the float
implementations.

### Adding Tests

Additional Tests should be added to `burn-tensor` under
[`crates/burn-tensor/src/tests/ops/{op_name}.rs`](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tensor/src/tests/ops/powf.rs#L1),
inserting the module name into `crates/burn-tensor/src/tests/ops/mod.rs`. Then add it to the
`testgen_all` macro under `crates/burn-tensor/src/tests/mod.rs`. This macro is called from the
`lib.rs` file in each backend, which autogenerates the tests for that specific backend. It isn't
necessary to define tests in the backends directly, save for those that require specific testing
such as`burn-autodiff`

## Adding the Op to burn-autodiff

Since this is probably the hardest and the least straightforward, we'll cover this backend
separately. Burn-autodiff enables other backends to use autodifferentiation[^autodiff]. Ops for
float types are implemented in
[crates/burn-autodiff/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-autodiff/src/ops/tensor.rs#L2172)
and need to:

1. define a unit struct [^absolute_units] that implements a backward (pass) function
2. Within the backward function, as this is an elementwise binary operation it implements the binary
   function (from backward.rs under the same directory), the last 2 arguments are two closures that
   define the left and right partial derivatives.
3. Then defines what happens when a specific operation is tracked or untracked, where untracked just
   calls the function in the normal way, and tracked executes the backward function defined above

Steps 1 and 3 are boilerplate, so much so that you can probably just copy the contents of another op
of the same type (binary, unary) and change the name of the struct, and ensure that either both
sides have the data they need (if they need to have a copy of the opposite sided tensor, clone its
contents).

For those that need it, here is a quick refresher on the necessary calculus. If you are familiar
with how to calculate partial derivatives, you can skip this section.

Since pow is a binary operation, the left and right functions are the partial derivatives with
respect to the left and right sided tensors.

Let's define the operator as a function \\(f(x,y)=x^{y}\\) , where \\(x\\) is the left hand tensor
and \\(y\\) is the right handed tensor. The two closures are defining the partial derivatives of
\\(f\\) with respect to \\(x\\),\\(y\\). Treat the other variables as a constant

$$\frac{\delta }{\delta x} (x^{y})= y \cdot x^{y-1}$$ is the left handed closure, and

$$\frac{\delta }{\delta y} (x^{y}) = x^{y} \cdot ln(x)$$

is the right. If you aren't sure how to calculate these by hand, I recommend using
[symbolab](<https://www.symbolab.com/solver/partial-derivative-calculator/%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%5Cleft(x%5E%7By%7D%5Cright)?or=input>),
plug in your operator in terms of \\(x\\) and \\(y\\), and just swap out the variable
\\(x\\)|\\(y\\) in the partial derivative to get the other side.

### Testing autodiff

Test for autodiff go under
[burn-autodiff/src/tests/{op_name}.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-autodiff/src/tests/pow.rs#L31)
(replacing `op_name` with whatever makes sense for your op), and for tensor operations both the left
and right side need to be verified. The easiest way to do this, is to

1. use small tensors with simple values
2. Compute the expected results for the chosen tensors, using some independent and reliable tool.
   For instance, you can pop open a terminal and launch `ipython` import `numpy` (or just use
   [google colab](https://colab.google/) if you don't have the packages installed and don't want to
   install them), and do the calculations by hand.
3. comparing the actual to expected output for lhs, rhs and regular operation

generally, it seems preferable to use
`actual_output_tensor.to_data().assert_approx_eq(&expected_tensor_data,3)` to `assert_eq!(...` due
to occasional hiccups with floating point calculations.

## Adding the Op to other backends

Most of these are fairly straightforward implementations. For reference here's pow's float
implementation for torch, ndarray and candle backends:

1. Torch implementation in
   [crates/burn-tch/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tch/src/ops/tensor.rs#L458)
   and the Op used in
   [crates/burn-tch/src/ops/base.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-tch/src/ops/base.rs#L481)
2. NdArray in
   [crates/burn-ndarray/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-ndarray/src/ops/tensor.rs#L465)
3. Candle in
   [crates/burn-candle/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-candle/src/ops/tensor.rs#L492)

This is where any calculation happens currently. Playing a guessing game with method names and
seeing what completions are suggested will take you far. If you are having trouble figuring out how
to do it from the docs for that backend,
[try searching github for relevant function calls](https://docs.github.com/en/search-github/github-code-search/understanding-github-code-search-syntax).

## Adding the Op to fusion, jit, and wgpu backends

Adding an operator to these backends can be fairly straightforward, though due to what these
backends are for, involves a bit more indirection. Fusion and jit, like autodiff, are not target
backends as much as backends that enable certain functionality for other backends, in this case
kernel fusion (which is currently only supported for `burn-wgpu`) or just-in-time compilation.
Adding the operator won't involve doing any calculation, you'll just be describing how the generated
code should look. Most of this can be copy/pasted/adjusted from other functions.

Here's how powf was added to burn fusion:

1. added powf to the float ops under
   [`crates/burn-fusion/src/ops/float.rs`](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-fusion/src/ops/float.rs#L1813)
2. added powf to the `NumericOperationDescription` enum under
   [crates/burn-fusion/src/stream/operation.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-fusion/src/stream/operation.rs#L426)
3. added powf to the implementations of `NumericOperationDescription` enum under
   [crates/burn-fusion/src/stream/context.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-fusion/src/stream/context.rs#L764)

The way wgpu handles tensor-scalar operations is by transforming both into a sequence of vectorized
scalar operations. Since powf already existed in burn-wgpu, it was pretty easy to reuse the existing
implementation for the situation where both sides of the operation were tensors. The `burn-wgpu`
crate is primarily concerned with how the operation is compiled and executed by the gpu. The actual
implementation is defined in `burn-jit`.

Here is where code was added for powf in `burn-jit` and `burn-wgpu`:

1. to the implementation of
   [`FloatTensorOps` under `crates/burn-jit/src/ops/float_ops.rs`](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-jit/src/ops/float_ops.rs#L491)
2. the function being called was added to
   [crates/burn-jit/src/ops/numeric.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-jit/src/ops/numeric.rs#L208)
3. the operator was defined in
   [`crates/burn-jit/src/codegen/dialect/gpu/operation.rs`](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-jit/src/codegen/dialect/gpu/operation.rs#L37)
4. the vectorization was added to
   [`crates/burn-jit/src/codegen/dialect/gpu/vectorization.rs`](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-jit/src/codegen/dialect/gpu/vectorization.rs#L55)
5. how the operation looks to the gpu was added to
   [`crates/burn-jit/src/fusion/tracing/builder.rs`](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-jit/src/fusion/tracing/builder.rs#L279)
6. the mapping between the gpu operation and the WGSL instruction was added to
   [`crates/burn-wgpu/src/compiler/wgsl/compiler.rs`](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-wgpu/src/compiler/wgsl/compiler.rs#L455)
7. the WGSL instruction itself was added to the
   [instruction op enum in `crates/burn-wgpu/src/compiler/wgsl/instructions.rs`](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-wgpu/src/compiler/wgsl/instructions.rs#L103),
   and the actual
   [instruction in wgsl here](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-wgpu/src/compiler/wgsl/instructions.rs#L273)

We needed to generate some custom WGSL code for powf, primarily due to issues with proper case
handling of the wgsl pow function, like 0 to the 0 power being 1, and any negative number to an even
power being positive. We reused as much as the existing logic as possible, and then branched at the
last point based off the var type of the rhs.
[See here](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-wgpu/src/compiler/wgsl/compiler.rs#L596).
For most operations, you shouldn't need to add to `crates/burn-wgpu/src/compiler/wgsl/extension.rs`
unless the operation isn't native to WGSL.

## Adding the Op to burn-import

I won't comment on generating the ONNX test files or the tests, as this is already covered
[in the ONNX to burn guide](onnx-to-burn-conversion-tool.md#adding-new-operators), this is more
about the specific changes you need to make when adding new operators after you have generated the
tests.

The crate is divided into two sections `src/burn` and `src/onnx`. The code under the former
corresponds to the operation you've implemented earlier in this guide, and the latter to the
operations defined in the ONNX specification. So when you are loading a model, the operator is first
parsed to an intermediate representation defined by `src/onnx`, and then mapped to a Burn operations
defined under `src/burn/node`.

Let's review the changes made for pow starting from `src/burn` and moving to `src/onnx`:

1. determine the type of operator and add your operator to the appropriate node (operation) type, in
   this case
   [BinaryNode under crates/burn-import/src/burn/node/binary.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-import/src/burn/node/binary.rs#L160)
   along with its
   [`to_str` definition](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-import/src/burn/node/binary.rs#L15)
2. add an arm to the match statement inside the `into_burn` function in
   [crates/burn-import/src/onnx/to_burn.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-import/src/onnx/to_burn.rs#L268)
   for the ONNX `NodeType` (which corresponds to an op in the ONNX spec), and make an
   [`{op}_conversion` function](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-import/src/onnx/to_burn.rs#L682)
   that maps the ONNX node to the binary type
3. specify how dimensions for the output should be derived in
   [crates/burn-import/src/onnx/dim_inference.rs](https://github.com/tracel-ai/burn/blob/e303e31c8bc85486690ff80df65d1e25e16728c4/crates/burn-import/src/onnx/dim_inference.rs#L53)

And you're done! Congrats, you just fully added a new op to burn, and we are all one step closer to
the answer to [are we learning yet?](https://www.arewelearningyet.com/) being "Yes, and it's
freaking fast!". Buy yourself a coffee.

[^supertrait]:
    for more on supertraits see
    [the advanced trait section of the rust book](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-supertraits-to-require-one-traits-functionality-within-another-trait)

[^autodiff]:
    wiki link for
    [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)

[^absolute_units]:
    for more information on unit structs see
    [the defining and instantiating structs section of the rust book](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#unit-like-structs-without-any-fields)
