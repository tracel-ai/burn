# testing

## Test for TensorOps

The following examples use matrix multiplication operation.

Test for Tensor operations (as in given this input, expect it match or approximate this output) are defined only in [`burn-tensor/src/test/ops`](https://github.com/tracel-ai/burn/blob/b9bd42959b0d3e755a25e383cb5b38beb25559b8/burn-tensor/src/tests/ops/matmul.rs#L1) and not in the backends, with the exception of `burn-autodiff`. These test are added to the `testgen_all` macro rule in [`burn-tensor/src/test/mod.rs`](https://github.com/tracel-ai/burn/blob/b9bd42959b0d3e755a25e383cb5b38beb25559b8/burn-tensor/src/tests/mod.rs#L59). This is then propagated to the existing backends without any additional work.

### Test for Autodiff

The following examples use the power operation.

Tests for autodiff go under [burn-autodiff/src/tests/{op_name}.rs](https://github.com/tracel-ai/burn/blob/4ca3e31601228952bb1c1492bc9cd2adf15b5cf1/burn-autodiff/src/tests/pow.rs#L31) (replace `foo` for whatever makes sense for your op), and for tensor operations both the left and right sides need to be verified. The easiest way to do this, is to:

1. use small tensors with simple values
2. pop open a terminal, launch `ipython` and import `numpy` then do the calculations by hand. You can also use [google colab](https://colab.google/) if you prefer so that you don't have to install the packages on your system.
3. compare the actual output to the expected output for lhs, rhs and regular operation

generally, it seems preferable to use `actual_output_tensor.into_data().assert_approx_eq(&expected_tensor_data,3)` to `assert_eq!(...` due to occasional hiccups with floating point calculations.
