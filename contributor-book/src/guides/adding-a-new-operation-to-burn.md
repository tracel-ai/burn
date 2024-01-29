# Adding a new operation to burn

Let's discuss how one might go about adding new operators to Burn, using the example of the recently added [pow operator](https://github.com/tracel-ai/burn/pull/1133/files) (as of 01/24/2024). In that PR, the following things took place (albeit not in this order)

## Adding the Op to burn-tensor

`burn-tensor` is the crate that defines all tensor operations that need to be implemented by the various backends. The core of this lies in  `burn-tensor/src/tensor/api/numeric.rs`, which is home to the numeric trait and its implementation for the different tensor types. The numeric trait is the home of all tensor operations that are numeric in nature and that are shared by `Int` and `Float` Tensor types. More information on the relationship between Tensor modules can be found under the section for [Tensor Architecture](../project-architecture/Tensor.md#tensorops).

here is where pow was added to `burn-tensor/src/tensor/api/numeric.rs`:

1. for the [`Tensor<Backend,Dimension,Kind>` struct](https://github.com/tracel-ai/burn/blob/3b7d9feede702cd616c273fa9eba9fbf14f66964/burn-tensor/src/tensor/api/numeric.rs#L553)
2. for the [numeric trait](https://github.com/tracel-ai/burn/blob/3b7d9feede702cd616c273fa9eba9fbf14f66964/burn-tensor/src/tensor/api/numeric.rs#L1618)
3. for the impl of numeric for [float](https://github.com/tracel-ai/burn/blob/3b7d9feede702cd616c273fa9eba9fbf14f66964/burn-tensor/src/tensor/api/numeric.rs#L2186) and [int](https://github.com/tracel-ai/burn/blob/3b7d9feede702cd616c273fa9eba9fbf14f66964/burn-tensor/src/tensor/api/numeric.rs#L1903)

Tensor is a struct that has a single member: `primitive`(defined [here](https://github.com/tracel-ai/burn/blob/main/burn-tensor/src/tensor/api/base.rs)), That is defined by it's [`Kind`](https://github.com/tracel-ai/burn/blob/3b7d9feede702cd616c273fa9eba9fbf14f66964/burn-tensor/src/tensor/api/kind.rs#L15): one of  `Bool`, `Float`, or `Int` (those linked in 3). These call the ops for that data type defined in the [`Backend`](https://github.com/tracel-ai/burn/blob/3b7d9feede702cd616c273fa9eba9fbf14f66964/burn-tensor/src/tensor/backend/base.rs#L52) supertrait[^1]. This is the trait that is then implemented by the different `burn-` back-ends (such as `burn-ndarray` and `burn-wgpu`) which implement the functions if no default is provided.

In this case, we only don't need to worry about `Bool` Tensors. Ops for `Float` is implemented under [burn-tensor/src/tensor/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/3b7d9feede702cd616c273fa9eba9fbf14f66964/burn-tensor/src/tensor/ops/tensor.rs#L873), and for `Int` under [`burn-tensor/src/tensor/ops/int_tensor.rs`](https://github.com/tracel-ai/burn/blob/e1d873abe2c2fa0bb316719c4761eaf796291166/burn-tensor/src/tensor/ops/int_tensor.rs#L486). The current convention is for int and bool ops, to be prefixed with `int` or `bool`, respectively. As we need to use floating point powers to ensure precision, the `powi` functions are given a default implementation that converts the right hand side of the operator to a float.

The `Int` Tensor function use the ones defined for Float with 2 extra cast (LHS to a `Float` tensor, Output to an `Int`). Given that the rest of the code will only look at the float implementations.

### Adding Test

Additional Test should be added to `burn-tensor` under [`burn-tensor/src/tests/ops/{op_name}.rs`](https://github.com/tracel-ai/burn/burn-tensor/src/tests/ops/powf.rs), inserting the module name into `burn-tensor/src/tests/ops/mod.rs`. Then add it to the `testgen_all` macro under `burn-tensor/src/tests/mod.rs`. This test is automatically added to the backends so it isn't necessary to add them there, save for those that require specific testing such as`burn-autodiff`

## Adding the Op to the burn-autodiff

Since this is probably the hardest and the least straightforward, we'll cover this backend separately. Burn-autodiff enables other backends to use autodifferentiation[^2]. Ops for float types are implemented in [burn-autodiff/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/e1d873abe2c2fa0bb316719c4761eaf796291166/burn-autodiff/src/ops/tensor.rs#L1523) and need to:

1. define a unit struct [^3] that implements a backward (pass) function
2. Within the backward function, as this is an elementwise binary operation it implements the binary function (from backward.rs under the same directory), the last 2 arguments are two closures that define the left and right partial derivatives.
3. Then defines what happens when a specific operation is tracked or untracked, where untracked just calls the function in the normal way, and tracked executes the backward function defined above

steps 1 and 3 are boilerplate, so much so that you can probably just copy the contents of another op of the same type (binary, unary) and change the name of the struct, and ensure that either both sides have the data they need (if they need to have a copy of the opposite sided tensor, clone it's contents).

now for step 2. Since a significant number of the people reading this probably haven't touched calculus either ever, or since however long ago you took the appropriate course, I'll assume that you, the observer, have some prior knowledge of calculus but would benefit from a review of the concepts. If this is not the case, I apologize, you can probably skim this section.

In the case of pow, since this is a binary operation, the left and right functions are the partial derivatives with respect to the left and right sided tensors.

Let's define the operator as a function \\(f(x,y)=x^{y}\\) , where \\(x\\) is the left hand tensor and \\(y\\) is the right handed tensor. The two closures are defining the partial derivatives of \\(f\\) with respect to \\(x\\),\\(y\\). The eli5 is treat the other variable as a constant

$$\frac{\delta }{\delta x} (x^{y})= y \cdot x^{y-1}$$
is the left handed closure, and

$$\frac{\delta }{\delta y} (x^{y}) = x^{y} \cdot ln(x)$$

is the right. If you aren't sure how to calculate these by hand, I recommend using [symbolab](https://www.symbolab.com/solver/partial-derivative-calculator/%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20x%7D%5Cleft(x%5E%7By%7D%5Cright)?or=input), plug in your operator in terms of \\(x\\) and \\(y\\), and just swap out the variable \\(x\\)|\\(y\\) in the partial derivative to get the other side.

### Testing autodiff

Test for autodiff go under [burn-autodiff/src/tests/foo.rs](https://github.com/tracel-ai/burn/blob/4ca3e31601228952bb1c1492bc9cd2adf15b5cf1/burn-autodiff/src/tests/pow.rs#L31) (replacing foo for whatever makes sense for your op), and for tensor operations both the left and right side need to be verified. The easiest way to do this, is to

1. use small tensors with simple values
2. pop open a terminal and launch `ipython` import `numpy` (or just use [google colab](https://colab.google/) if you don't have the packages installed and don't want to install them),  and do the calculations by hand.
3. comparing the actual to expected output for lhs, rhs and regular operation

generally, it seems preferable to use `actual_output_tensor.to_data().assert_approx_eq(&expected_tensor_data,3)` to `assert_eq!(...` due to occasional hiccups with floating point calculations.

## Adding the Op to other backends

most of these are fairly straightforward implementations. for reference here's pow's float implementation for torch, ndarray and candle backends:

1. Torch implementation in [burn-tch/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/main/burn-tch/src/ops/tensor.rs#L461) and the Op used in [burn-tch/src/ops/base.rs](https://github.com/tracel-ai/burn/blob/4ca3e31601228952bb1c1492bc9cd2adf15b5cf1/burn-tch/src/ops/base.rs#L443)
2. NdArray in [burn-ndarray/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/main/burn-ndarray/src/ops/tensor.rs#L443)
3. Candle in [burn-candle/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/4ca3e31601228952bb1c1492bc9cd2adf15b5cf1/burn-candle/src/ops/tensor.rs#L481)

This is where any calculation happens currently. Playing a guessing game with method names and seeing what completions are suggested will take you far. If you are having trouble figuring out how to do it from the docs for that backend, [try searching github for relevant function calls](https://docs.github.com/en/search-github/github-code-search/understanding-github-code-search-syntax).

## Adding the Op to fusion and wgpu backends

Adding an operator to these backends is fairly straightforward, though due to what these backends are for, involves a bit more indirection. Fusion, like autodiff, is not a target backend as much as a backend that enables certain functionality for other backends, in this case kernel fusion (which is currently only supported for `burn-wgpu`), so adding the operator won't involve doing any calculation, you'll just be describing how the generated code should look. Most of this can be copy/pasted/adjusted from other functions.

here's how powf was added to burn fusion:

1. added powf to the float ops under [`burn-fusion/src/ops/float.rs`](https://github.com/tracel-ai/burn/blob/0368409eb3a7beaeda598c0c8ce1dc0c2c8c07cc/burn-fusion/src/ops/float.rs#L1758)
2. added powf to the `FloatOperationDescription` enum under [burn-fusion/src/stream/operation.rs](https://github.com/tracel-ai/burn/blob/0368409eb3a7beaeda598c0c8ce1dc0c2c8c07cc/burn-fusion/src/stream/operation.rs#L385)
3. added powf to the implementations of `FloatOperationDescription` enum under [burn-fusion/src/stream/context.rs](https://github.com/tracel-ai/burn/blob/0368409eb3a7beaeda598c0c8ce1dc0c2c8c07cc/burn-fusion/src/stream/context.rs#L726)

 Adding pow to wgpu was actually pretty easy due to the design. Element-wise tensor ops are just vectorized scalar ops, and given that raising a tensor to a scalar power prior to the tensor version, I just reused the code for scalar powf.

here is where code was added

1. to the implementation of [`TensorOps` under `burn-wgpu/src/ops/float_ops.rs`](https://github.com/tracel-ai/burn/blob/0368409eb3a7beaeda598c0c8ce1dc0c2c8c07cc/burn-wgpu/src/ops/float_ops.rs#L513)
2. the function being called was added to [burn-wgpu/src/ops/numeric.rs](https://github.com/tracel-ai/burn/blob/0368409eb3a7beaeda598c0c8ce1dc0c2c8c07cc/burn-wgpu/src/ops/numeric.rs#L199)
3. the call to the fmt function use to generate wgsl code in [`burn-wgpu/src/codegen/kernel.rs`](https://github.com/tracel-ai/burn/blob/main/burn-wgpu/src/codegen/kernel.rs#L208)
4. A custom function generator was added to [`burn-wgpu/src/codegen/function.rs`](https://github.com/tracel-ai/burn/blob/main/burn-wgpu/src/codegen/function.rs#L99)

Much of the logic  for powf had already been defined, so not much needed to be added. The reason for the custom powf function is generated, rather than directly using the underlying function as is the case with other operations, is due to issues with case handling of the wgsl pow function, like 0 to the 0 power being 1, and any negative number to an even power being positive. We reused as much as the existing logic as possible, such as the operation output description generation in [`burn-wgpu/src/fusion/elemwise/builder.rs`](https://github.com/tracel-ai/burn/blob/main/burn-wgpu/src/fusion/elemwise/optimization.rs) and then branched at the last point based off the var type of the rhs. I don't know if there are other places within `burn-wgpu` where additions would have been necessary otherwise.

## Adding the Op to burn-import

I won't comment on generating the test onnx files or the test, as that is already covered [in the ONNX to burn guide](onnx-to-burn-conversion-tool.md#adding-new-operators), this is more about the specific changes you need to make when adding new operators after you have generated the tests.

The crate is divided into two sections `src/burn` and `src/onnx`. The code under the former corresponds to the operation you've implemented earlier in this guide, and the latter to the operations defined in the onnx specification. So when you are loading a model, the operator is first parsed to an intermediate representation defined by `src/onnx`, and then mapped to a Burn operations  defined under `src/burn/node`.

Let's review the changes made for pow starting from `src/burn` and moving to `src/onnx`:

1. determine the type of operator and add your operator to the appropriate node (operation) type, in this case [BinaryNode under burn-import/src/burn/node/binary.rs](https://github.com/tracel-ai/burn/blob/0368409eb3a7beaeda598c0c8ce1dc0c2c8c07cc/burn-import/src/burn/node/binary.rs#L160) along with its [`to_str` definition](https://github.com/tracel-ai/burn/blob/0368409eb3a7beaeda598c0c8ce1dc0c2c8c07cc/burn-import/src/burn/node/binary.rs#L9)
2. add an arm to the match statement inside the `into_burn` function in [burn-import/src/onnx/to_burn.rs](https://github.com/tracel-ai/burn/blob/0368409eb3a7beaeda598c0c8ce1dc0c2c8c07cc/burn-import/src/onnx/to_burn.rs#L269) for the onnx `NodeType`(corresponds to an op in the Onnx spec), and make a [`foo_conversion` function](https://github.com/tracel-ai/burn/blob/0368409eb3a7beaeda598c0c8ce1dc0c2c8c07cc/burn-import/src/onnx/to_burn.rs#L667) that maps the onnx node to the binary type
3. specify how dimensions for the output should be derived in [burn-import/src/onnx/dim_inference.rs](https://github.com/tracel-ai/burn/blob/0368409eb3a7beaeda598c0c8ce1dc0c2c8c07cc/burn-import/src/onnx/dim_inference.rs#L103)

And you're done! Congrats, you just fully added a new op to burn, and we are all one step closer to the answer to [are we learning yet?](https://www.arewelearningyet.com/) being "Yes, and it's freaking fast!". Buy yourself a coffee

[^1]: for more on supertraits see [the advanced trait section of the rust book](https://doc.rust-lang.org/book/ch19-03-advanced-traits.html#using-supertraits-to-require-one-traits-functionality-within-another-trait)

[^2]: wiki link for  [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)

[^3]: for more information on unit structs see [the defining and instantiating structs section of the rust book](https://doc.rust-lang.org/book/ch05-01-defining-structs.html#unit-like-structs-without-any-fields)
