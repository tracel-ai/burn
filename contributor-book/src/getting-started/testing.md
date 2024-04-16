# Testing

## Test for Tensor Operations

Test for tensor operations (generally of the form: given this input, expect it match or approximate
this output) are defined only in
[`crates/burn-tensor/src/test/ops`](https://github.com/tracel-ai/burn/tree/81a67b6a0992b9b5c33cda8b9784570143b67319/crates/burn-tensor/src/tests/ops)
and not in the backends (with the exception of `burn-autodiff`). The tensor operation tests are
added to the `testgen_all` macro rule in
[`crates/burn-tensor/src/tests/mod.rs`](https://github.com/tracel-ai/burn/blob/81a67b6a0992b9b5c33cda8b9784570143b67319/crates/burn-tensor/src/tests/mod.rs).
This is then propagated to the existing backends without any additional work.

### Test for Autodiff

Tests for autodiff go under
[burn-autodiff/src/tests](https://github.com/tracel-ai/burn/tree/81a67b6a0992b9b5c33cda8b9784570143b67319/crates/burn-autodiff/src/tests)
and should verify backward pass correctness. For binary tensor operations, both the left and right
sides need to be verified.

Here's an easy way to define tests for a new operation's backward pass:

1. Use small tensors with simple values.
2. Pop open a terminal, launch `ipython` and import `numpy` then do the calculations by hand. You
   can also use [Google Colab](https://colab.google/) so you don't have to install the packages on
   your system.
3. Compare the actual outputs to the expected output for left-hand side, right-hand side.

For float tensors, it is advised to use
`actual_output_tensor.into_data().assert_approx_eq(&expected_tensor_data,3)` instead of
`assert_eq!(...` due to occasional hiccups with floating point calculations.
