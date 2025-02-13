# Adding a New Operation to burn

Let's discuss how one might go about adding new operators to Burn, using the example of the pow
operator added in [this PR](https://github.com/tracel-ai/burn/pull/1133/files).

## Adding the Op to burn-tensor

`burn-tensor` is the crate that defines all tensor operations that need to be implemented by the
various backends. The core of this lies in
[crates/burn-tensor/src/tensor/api/numeric.rs](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tensor/api/numeric.rs),
which is home to the numeric trait and its implementation for the different tensor types. The
numeric trait is the home of all tensor operations that are numeric in nature and that are shared by
`Int` and `Float` Tensor types. More information on the relationship between Tensor modules can be
found under the section for [Tensor Architecture](../project-architecture/tensor.md#tensor-operations).

Here is where pow was added to `crates/burn-tensor/src/tensor/api/numeric.rs`:

1. for the
   [`Tensor<Backend, Dimension, Kind>` struct](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tensor/api/numeric.rs#L573)
2. for the
   [numeric trait](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tensor/api/numeric.rs#L1955)
3. for the implementation of numeric for
   [float](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tensor/api/numeric.rs#L2722)
   and
   [int](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tensor/api/numeric.rs#L2375)

Tensor is a struct that has a single member: `primitive` (defined
[here](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tensor/api/base.rs#L27)),
that is defined by its
[`Kind`](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tensor/api/kind.rs#L16):
one of `Bool`, `Float`, or `Int` (those linked in 3). These call the ops for that data type defined
in the
[`Backend`](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tensor/backend/base.rs#L54)
supertrait[^supertrait]. This is the trait that is then implemented by the different `burn-`
backends (such as `burn-ndarray` and `burn-wgpu`) which must implement the functions if no default
is provided.

In this case, we don't need to worry about `Bool` Tensors. `Float` ops are implemented under
[`crates/burn-tensor/src/tensor/ops/tensor.rs`](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tensor/ops/tensor.rs#L991),
and `Int` ops under
[`crates/burn-tensor/src/tensor/ops/int_tensor.rs`](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tensor/ops/int_tensor.rs#L539).
The current convention is ops of each type, if not unique to that type, are prefixed with the type.
So `powf` and sundry would be defined as `int_powf` for `IntTensorOps` and `float_powf` for
`FloatTensorOps`. If an op is unique to a type, then it should be implemented under
`burn-tensor/src/api/{type}.rs`. For example, here is an implementation for
[`sin` under `crates/burn-tensor/src/api/float.rs`](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tensor/api/float.rs#L82)
which obviously doesn't make sense for `Int` or `Bool` tensors.

The `Int` Tensor function uses the ones defined for Float with 2 extra casts (LHS to a `Float`
tensor, Output to an `Int`). Given that the rest of the code will only look at the float
implementations.

With the addition of quantized float tensors, the `Float` tensor primitive is represented by the
[`TensorPrimitive`](https://github.com/tracel-ai/burn/blob/a6a5c22e0db56d947b9165d4dae42783a5a6b689/crates/burn-tensor/src/tensor/api/kind.rs#L69)
enum. This allows us to handle both float and quantized float operations in the `Tensor`
implementation, correctly dispatching to the corresponding op (float or quantized) based on the
variant. Following the same convention, the equivalent
[quantized tensor ops](https://github.com/tracel-ai/burn/blob/a6a5c22e0db56d947b9165d4dae42783a5a6b689/crates/burn-tensor/src/tensor/ops/qtensor.rs#L45)
are prefixed with `q_*` (e.g., `q_reshape` instead of `float_reshape`). Most ops have a default
implementation that simply dequantizes the input into its floating-point representation, performs
the operation on the float tensor, and quantizes the output. Backends can overwrite specific
implementations when required/desired.

### Adding Tests

Additional Tests should be added to `burn-tensor` under
[`crates/burn-tensor/src/tests/ops/{op_name}.rs`](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tensor/src/tests/ops/powf.rs#L1),
inserting the module name into `crates/burn-tensor/src/tests/ops/mod.rs`. Then add it to the
`testgen_all` macro under `crates/burn-tensor/src/tests/mod.rs`. This macro is called from the
`lib.rs` file in each backend, which autogenerates the tests for that specific backend. It isn't
necessary to define tests in the backends directly, save for those that require specific testing
such as `burn-autodiff`.

For float tensor operations, the
[`QTensorOps`](https://github.com/tracel-ai/burn/blob/a6a5c22e0db56d947b9165d4dae42783a5a6b689/crates/burn-tensor/src/tensor/ops/qtensor.rs#L45)
counterpart is usually added at the same time with a default implementation (as mentioned in the
previous section). Tests for `q_*` ops follow a similar procedure: the test is added under
[`crates/burn-tensor/src/tests/quantization/ops/{op_name}.rs`](https://github.com/tracel-ai/burn/tree/a6a5c22e0db56d947b9165d4dae42783a5a6b689/crates/burn-tensor/src/tests/quantization/ops),
the module name is inserted into `crates/burn-tensor/src/tests/quantization/ops/mod.rs` and finally
the test is added to the
[`testgen_quantization` macro](https://github.com/tracel-ai/burn/blob/a6a5c22e0db56d947b9165d4dae42783a5a6b689/crates/burn-tensor/src/tests/mod.rs#L67).
If you take a look at any of the existing tests for an operation on a quantized tensor,
you will see that the inputs and expected outputs are always defined with floating point values.
While it assumes that the quantization and dequantization are correct, it makes the tests much more
readable and easier to understand w.r.t. what is being tested. Effectively, the tests are there to
ensure that a tensor operation is invariant to quantization (up to some quantization error, of
course).

_Note: the tests try to use tensors with floating point values which can be de/quantized without
introducing too much quantization error, but the result always depends on the operation (e.g.,
tensor product of values can grow larger and significantly increase the output tensor range, leading
to more de/quantization error on the results)._

## Adding the Op to burn-autodiff

Since this is probably the hardest and the least straightforward, we'll cover this backend
separately. `burn-autodiff` enables other backends to use autodifferentiation[^autodiff]. Ops for
float types are implemented in
[crates/burn-autodiff/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-autodiff/src/ops/tensor.rs)
and need to:

1. Define a unit struct [^absolute_units] that implements a backward (pass) function
2. Within the backward function, as this is an elementwise binary operation it implements the binary
   function (from `backward.rs` under the same directory), the last 2 arguments are two closures
   that define the left and right partial derivatives.
3. Then define what happens when a specific operation is tracked or untracked, where untracked just
   calls the function in the normal way, and tracked sets the execution the backward function
   defined above.
4. When tracked, operations are part of the autodiff graph and must save the needed information to
   efficiently perform their backward pass later. If the information is light (such as a shape), it
   should be directly saved in the state. If the operation's inputs are needed to compute the
   backward pass, it should be checkpointed rather than saved. This will allow the input to be
   provided lazily at the backward pass depending on the checkpointing strategy.
5. An operation must also be identified as _compute-bound_ (`.computeBound()`) or _memory-bound_
   (`.memoryBound()`) for gradient checkpointing. _Compute-bound_ operation are heavy to compute
   (for instance matmul or convolution), which means that even with checkpointing they will save
   their output for the backward pass and not recompute it. _Memory-bound_ operations are more
   trivial (like `powf` which only performs one small operation per tensor entry), so it can be
   beneficial to recompute them during the backward pass instead of saving their whole forward
   output to memory. Operations registered as _memory-bound_ need to know their parents
   (`.parents()` method) and how to recompute their forward pass during the backward pass (with a
   struct that implements `RetroForward`), using their parents' outputs.

The above steps are mostly boilerplate, so you can often just copy the contents of another similar
op, change the name of the structs, and ensure that either both sides have the data they need (if
they need to have a copy of the opposite sided tensor, clone its contents).

### Computing derivatives

For those that need it, here is a quick refresher on the necessary calculus. If you are familiar
with how to calculate partial derivatives, you can skip this section.

Since `pow` is a binary operation, the left and right functions are the partial derivatives with
respect to the left and right sided tensors.

Let's define the operator as a function \\(f(x,y)=x^{y}\\) , where \\(x\\) is the left hand tensor
and \\(y\\) is the right handed tensor. The two closures are defining the partial derivatives of
\\(f\\) with respect to \\(x\\),\\(y\\). Treat the other variables as a constant

$$\frac{\delta }{\delta x} (x^{y})= y \cdot x^{y-1}$$ is the left handed closure, and

$$\frac{\delta }{\delta y} (x^{y}) = x^{y} \cdot ln(x)$$

is the right. If you aren't sure how to calculate these by hand, it is recommended to use
[symbolab](<https://www.symbolab.com/solver/partial-derivative-calculator/%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%5Cleft(x%5E%7By%7D%5Cright)?or=input>),
plug in your operator in terms of \\(x\\) and \\(y\\), and just swap out the variable
\\(x\\)|\\(y\\) in the partial derivative to get the other side.

### Testing autodiff

For testing the `autodiff` operations, please refer to
[this section](../getting-started/testing.md).

## Adding the Op to other backends

Most of these are fairly straightforward implementations. For reference here's pow's float
implementation for torch, ndarray and candle backends:

1. Torch implementation in
   [crates/burn-tch/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tch/src/ops/tensor.rs#L467)
   and the Op used in
   [crates/burn-tch/src/ops/base.rs](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-tch/src/ops/base.rs#L481)
2. NdArray in
   [crates/burn-ndarray/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-ndarray/src/ops/tensor.rs#L472)
3. Candle in
   [crates/burn-candle/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-candle/src/ops/tensor.rs#L504)

This is where any calculation happens currently. Playing a guessing game with method names and
seeing what completions are suggested will take you far. If you are having trouble figuring out how
to do it from the docs for that backend,
[try searching github for relevant function calls](https://docs.github.com/en/search-github/github-code-search/understanding-github-code-search-syntax).

## Adding the Op to fusion, JIT and cubecl backends

Adding an operator to these backends can be fairly straightforward, though due to what these
backends are for, involves a bit more indirection. Fusion and jit, like autodiff, are not target
backends as much as backends that enable certain functionality for other backends, in this case
kernel fusion or just-in-time compilation. Adding the operator won't involve doing any calculation,
you'll just be describing how the generated code should look. Most of this can be
copy/pasted/adjusted from other functions.

Here's how powf was added to `burn-fusion`:

1. Added powf to the float ops under
   [`crates/burn-fusion/src/ops/float.rs`](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-fusion/src/ops/float.rs#L1838)
2. Added powf to the `NumericOperationIr` enum under
   [crates/burn-fusion/src/stream/operation.rs](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-fusion/src/stream/operation.rs#L433)
3. Added powf to the implementations of `NumericOperationIr` enum under
   [crates/burn-fusion/src/stream/context.rs](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-fusion/src/stream/context.rs#L771)

The way `cubecl` handles tensor-scalar operations is by transforming both into a sequence of
vectorized scalar operations. Since powf already existed in `cubecl`, it was pretty easy to reuse
the existing implementation for the situation where both sides of the operation were tensors. The
`cubecl` crate is primarily concerned with how the operation is compiled and executed by the gpu.
The actual implementation is defined in `burn-cubecl`.

Here is where code was added for powf in `burn-cubecl` and `cubecl`:

1. to the implementation of
   [`FloatTensorOps` under `crates/burn-cubecl/src/ops/float_ops.rs`](https://github.com/tracel-ai/burn/blob/3b51c26958128502d60fb35029c43d9b686b816c/crates/burn-cubecl/src/ops/float_ops.rs#L410)
2. the function being called was added to
   [crates/burn-cubecl/src/ops/numeric.rs](https://github.com/tracel-ai/burn/blob/3b51c26958128502d60fb35029c43d9b686b816c/crates/burn-cubecl/src/ops/numeric.rs#L147)
3. the operator was defined in
   [`cubecl-core/src/ir/operation.rs`](https://github.com/tracel-ai/cubecl/blob/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/crates/cubecl-core/src/ir/operation.rs#L68)
4. how the operation looks to the gpu was added to
   [`crates/burn-cubecl/src/fusion/on_write/ir.rs`](https://github.com/tracel-ai/burn/blob/3b51c26958128502d60fb35029c43d9b686b816c/crates/burn-cubecl/src/fusion/on_write/ir.rs#L52)
5. the mappings between the gpu operation and the CPP, WGSL and SPIR-V instructions were added to
   [`cubecl-cpp/src/shared/base.rs`](https://github.com/tracel-ai/cubecl/blob/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/crates/cubecl-cpp/src/shared/base.rs#L456),
   [`cubecl-wgpu/src/compiler/wgsl/compiler.rs`](https://github.com/tracel-ai/cubecl/blob/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/crates/cubecl-wgpu/src/compiler/wgsl/compiler.rs#L652)
   and
   [`cubecl-spirv/src/instruction.rs`](https://github.com/tracel-ai/cubecl/blob/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/crates/cubecl-spirv/src/instruction.rs#L408)
6. the instructions themselves were added for WGSL to
   [instruction op enum in `cubecl-wgpu/src/compiler/wgsl/instructions.rs`](https://github.com/tracel-ai/cubecl/blob/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/crates/cubecl-wgpu/src/compiler/wgsl/instructions.rs#L124),
   and the actual
   [instruction in wgsl here](https://github.com/tracel-ai/cubecl/blob/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/crates/cubecl-wgpu/src/compiler/wgsl/instructions.rs#L547-L555),
   for CPP in the enum here
   [`cubecl-cpp/src/shared/instruction.rs`](https://github.com/tracel-ai/cubecl/blob/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/crates/cubecl-cpp/src/shared/instruction.rs#L127)
   and the actual instruction here
   [`cubecl-cpp/src/shared/binary.rs`](https://github.com/tracel-ai/cubecl/blob/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/crates/cubecl-cpp/src/shared/binary.rs#L137)

We needed to generate some custom WGSL code for powf in WGSL, primarily due to issues with proper
case handling of the wgsl pow function, like 0 to the 0 power being 1, and any negative number to an
even power being positive. We reused as much as the existing logic as possible, and then branched at
the last point based off the var type of the rhs.
[See here](https://github.com/tracel-ai/cubecl/blob/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/crates/cubecl-wgpu/src/compiler/wgsl/compiler.rs#L911).
For most operations, you shouldn't need to add to `cubecl-wgpu/src/compiler/wgsl/extension.rs`
unless the operation isn't native to WGSL.

For functions that need a complex kernel without a direct mapping to a base instruction, simply use
the `cube` macro (see
[the `cubecl` book](https://github.com/tracel-ai/cubecl/tree/f5b63076a01a5c03ea9ed20799d3eeaf776b45da/cubecl-book)).

## Adding the Op to burn-import

Generating the ONNX test files or tests is already covered
[in the ONNX to burn guide](onnx-to-burn-conversion-tool.md#adding-new-operators); this is more
about the specific changes you need to make when adding new operators after you have generated the
tests.

Changes will need to be made to both `onnx-ir` and `burn-import`. The code within `onnx-ir` defines
how to parse the nodes in an onnx file and produces the intermediate representation. The code within
`burn-import` is divided into two sections: `src/onnx` and `src/burn`. The code under the former
maps that intermediate representation to one used for code generation and the latter defines how to
generate code for the operator you've implemented earlier in this guide.

So when you are loading a model, the operator is first parsed to an intermediate representation
defined by `burn-import` and then mapped to a Burn operation defined under `src/burn/node`; the
mapping from onnx to burn is aptly defined in `src/onnx/to_burn`

Let's review the changes made for powf starting from `src/burn` and moving to `src/onnx`:

1. Determine the type of operator and add your operator to the appropriate node (operation) type, in
   this case
   [BinaryNode under `crates/burn-import/src/burn/node/binary.rs`](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-import/src/burn/node/binary.rs#L160)
   along with its
   [`to_str` definition](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-import/src/burn/node/binary.rs#L15)
2. Add an arm to the match statement inside the `into_burn` function in
   [crates/burn-import/src/onnx/to_burn.rs](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-import/src/onnx/to_burn.rs#L272)
   for the ONNX `NodeType` (which corresponds to an op in the ONNX spec), and make an
   [`{op}_conversion` function](https://github.com/tracel-ai/burn/blob/0ee2021567b3725907df5fd1a905ce60b1aca096/crates/burn-import/src/onnx/to_burn.rs#L717)
   that maps the ONNX node to the binary type
3. Specify how dimensions for the output should be derived in
   [crates/onnx-ir/src/dim_inference.rs](https://github.com/tracel-ai/burn/blob/d4ae82b21ac3dd1def01bd380ab7ea4d3293eccb/crates/onnx-ir/src/dim_inference.rs#L17)

And you're done! Congrats, you just fully added a new operation to burn, and we are all one step
closer to the answer to [Are we learning yet?](https://www.arewelearningyet.com/) being "Yes, and
it's freaking fast!". Buy yourself a coffee.

[^supertrait]:
    for more on supertraits see
    [the advanced trait section of the rust book](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-supertraits-to-require-one-traits-functionality-within-another-trait)

[^autodiff]:
    wiki link for
    [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)

[^absolute_units]:
    for more information on unit structs see
    [the defining and instantiating structs section of the rust book](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#unit-like-structs-without-any-fields)
