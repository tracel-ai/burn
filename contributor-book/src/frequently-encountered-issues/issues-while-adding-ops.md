# Issues encountered while adding ops

Below are some of the issues that were encountered while adding ops to the project. If you encounter an issue while adding an op that isn't listed here, and it's not obvious how to fix it, please add it to this list.

## Off by .000001 errors

```sh
---- fusion::base::tests::maxmin::tests::test_mean_dim_2d stdout ---- thread 'fusion::base::tests::maxmin::tests::test_mean_dim_2d' panicked at burn-wgpu/src/fusion/base.rs:185:5: assertion `left == right` failed left: Data { value: [1.0, 4.0], shape: Shape { dims: [2, 1] } } right: Data { value: [0.99999994, 3.9999998], shape: Shape { dims: [2, 1] } } ----

tests::maxmin::tests::test_mean_dim_2d stdout ---- thread 'tests::maxmin::tests::test_mean_dim_2d' panicked at burn-wgpu/src/lib.rs:49:5: assertion `left == right` failed left: Data { value: [1.0, 4.0], shape: Shape { dims: [2, 1] } } right: Data { value: [0.99999994, 3.9999998], shape: Shape { dims: [2, 1] } }
```

If you encounter this, swap out the `assert_eq!` in the failing test for `tensor1.assert_approx_eq` with `3` as the second argument.

## Mismatched types and missing functions

```sh
error[E0308]: mismatched types --> {burn_dir}/target/debug/build/onnx-tests-fed12aaf3671687f/out/model/pow.rs:48:45 | 48 | let pow1_out1 = input1.clone().powf(input1); | ---- ^^^^^^ expected `f32`, found `Tensor<B, 4>` | | | arguments to this method are incorrect | = note: expected type `f32` found struct `Tensor<B, 4>` 

note: method defined here --> {burn_dir}/burn-tensor/src/tensor/api/float.rs:65:12 | 65 | pub fn powf(self, value: f32) -> Self { | ^^^^ 

error[E0599]: no method named `powf_scalar` found for struct `Tensor` in the current scope --> {burn_dir}/target/debug/build/onnx-tests-fed12aaf3671687f/out/model/pow.rs:50:35 | 50 | let pow2_out1 = pow1_out1.powf_scalar(cast1_out1); | ^^^^^^^^^^^ method not found in `Tensor<B, 4>` 

error[E0599]: no method named `powi` found for struct `Tensor` in the current scope --> {burn_dir}/target/debug/build/onnx-tests-fed12aaf3671687f/out/model/pow_int.rs:49:40 | 49 | let pow1_out1 = input1.clone().powi(input1); | ^^^^ method not found in `Tensor<B, 4, Int>` Some errors have detailed explanations: E0308, E0599. 
For more information about an error, try `rustc --explain E0308`. error: could not compile `onnx-tests` (test "onnx_tests") due to 3 previous errors
```

So if you are getting this, you probably didn't impl your operator for the actual Tensor struct. This issue was encountered when adding the Pow operator. The operation was added to the `FloatTensorOps` and `IntTensorOp` traits, but not for the numeric trait (under `burn-tensor/src/tensor/api/numeric.rs`). This, coupled with `powf` existing prior to the PR though only for scalar values (which had been renamed, just not in the right place), led to this confusing issue where it looked like the function was found, but the type was wrong. If that's the case, make sure that it's implemented for the appropriate type, in this case `Float` under [burn-tensor/src/tensor/api/numeric.rs](https://github.com/tracel-ai/burn/blob/4ca3e31601228952bb1c1492bc9cd2adf15b5cf1/burn-tensor/src/tensor/api/numeric.rs#L2186), and calling the `TensorOp.foo_op` defined under [burn-tensor/src/ops/tensor.rs](https://github.com/tracel-ai/burn/blob/4ca3e31601228952bb1c1492bc9cd2adf15b5cf1/burn-tensor/src/tensor/ops/tensor.rs#L873)
