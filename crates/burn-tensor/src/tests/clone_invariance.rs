#[burn_tensor_testgen::testgen(clone_invariance)]
/// This module tests whether basic tensor operations remain invariant when performed on clones,
/// meaning that cloning input tensors won't affect the results.
///
/// Those are relevant tests because backends may employ unsafe optimizations to reuse tensor data
/// and use different kernels in such cases. We ensure that the results are consistent regardless
/// of the approach and that the input tensors are not modified when cloned.
mod tests {
    use super::*;
    use burn_tensor::activation::{
        gelu, log_sigmoid, log_softmax, mish, relu, sigmoid, silu, softmax, softplus, tanh,
    };
    use burn_tensor::{Distribution, Tensor, TensorData};

    pub trait CloneInvarianceTest<const D: usize> {
        type Args;

        fn args(&self) -> Self::Args;

        fn run(&self, args: &Self::Args, inplace: bool) -> TensorData;

        fn check(&self) {
            let args = self.args();
            let out = self.run(&args, false);
            let out_inplace = self.run(&args, true);

            out.assert_approx_eq(&out_inplace, 4);
        }
    }

    macro_rules! clone_invariance_test {
        (unary: $name:ident, ops_float: $ops:expr) => {
            #[test]
            #[allow(non_snake_case)]
            fn $name() {
                struct $name;

                impl CloneInvarianceTest<2> for $name {
                    type Args = TensorData;

                    fn args(&self) -> Self::Args {
                        TestTensor::<2>::random(
                            [32, 32],
                            Distribution::Default,
                            &Default::default(),
                        )
                        .into_data()
                        .convert::<f32>()
                    }

                    fn run(&self, args: &Self::Args, inplace: bool) -> TensorData {
                        let lhs = TestTensor::from_data(args.clone(), &Default::default());

                        if inplace {
                            $ops(lhs).into_data().convert::<f32>()
                        } else {
                            let out = $ops(lhs.clone()).into_data().convert::<f32>();
                            lhs.into_data().assert_approx_eq(args, 4);
                            out
                        }
                    }
                }

                CloneInvarianceTest::<2>::check(&$name);
            }
        };

        (binary: $name:ident, ops_float: $ops:expr) => {
            #[test]
            #[allow(non_snake_case)]
            fn $name() {
                struct $name;

                impl CloneInvarianceTest<2> for $name {
                    type Args = (TensorData, TensorData);

                    fn args(&self) -> Self::Args {
                        let device = Default::default();
                        (
                            TestTensor::<2>::ones([32, 32], &device)
                                .into_data()
                                .convert::<f32>(),
                            // Avoid div by zero.
                            TestTensor::<2>::ones([32, 32], &device)
                                .into_data()
                                .convert::<f32>(),
                        )
                    }

                    fn run(&self, (lhs_arg, rhs_arg): &Self::Args, inplace: bool) -> TensorData {
                        let device = Default::default();
                        let lhs = TestTensor::from_data(lhs_arg.clone(), &device);
                        let rhs = TestTensor::from_data(rhs_arg.clone(), &device);

                        if inplace {
                            $ops(lhs, rhs).into_data().convert::<f32>()
                        } else {
                            let out = $ops(lhs.clone(), rhs.clone()).into_data().convert::<f32>();

                            lhs.into_data().assert_approx_eq(lhs_arg, 4);
                            rhs.into_data().assert_approx_eq(rhs_arg, 4);

                            out
                        }
                    }
                }

                CloneInvarianceTest::<2>::check(&$name);
            }
        };

        (unary: $name:ident, ops_int: $ops:expr) => {
            #[test]
            #[allow(non_snake_case)]
            fn $name() {
                struct $name;

                impl CloneInvarianceTest<2> for $name {
                    type Args = TensorData;

                    fn args(&self) -> Self::Args {
                        TestTensor::<2>::random(
                            [32, 32],
                            Distribution::Uniform(0.0, 50.0),
                            &Default::default(),
                        )
                        .into_data()
                        .convert::<i32>()
                    }

                    fn run(&self, args: &Self::Args, inplace: bool) -> TensorData {
                        let lhs = TestTensorInt::from_data(args.clone(), &Default::default());

                        if inplace {
                            $ops(lhs).into_data().convert::<f32>()
                        } else {
                            let out = $ops(lhs.clone()).into_data().convert::<f32>();
                            lhs.into_data().convert::<i32>().assert_approx_eq(args, 4);
                            out
                        }
                    }
                }

                CloneInvarianceTest::<2>::check(&$name);
            }
        };

        (binary: $name:ident, ops_int: $ops:expr) => {
            #[test]
            #[allow(non_snake_case)]
            fn $name() {
                struct $name;

                impl CloneInvarianceTest<2> for $name {
                    type Args = (TensorData, TensorData);

                    fn args(&self) -> Self::Args {
                        let device = Default::default();
                        (
                            TestTensor::<2>::random(
                                [32, 32],
                                Distribution::Uniform(0., 50.),
                                &device,
                            )
                            .into_data()
                            .convert::<i32>(),
                            // Avoid div by zero.
                            TestTensor::<2>::random(
                                [32, 32],
                                Distribution::Uniform(1., 51.),
                                &device,
                            )
                            .into_data()
                            .convert::<i32>(),
                        )
                    }

                    fn run(&self, (lhs_arg, rhs_arg): &Self::Args, inplace: bool) -> TensorData {
                        let device = Default::default();
                        let lhs = TestTensorInt::from_data(lhs_arg.clone(), &device);
                        let rhs = TestTensorInt::from_data(rhs_arg.clone(), &device);

                        if inplace {
                            $ops(lhs, rhs).into_data().convert::<f32>()
                        } else {
                            let out = $ops(lhs.clone(), rhs.clone()).into_data().convert::<f32>();

                            lhs.into_data()
                                .convert::<i32>()
                                .assert_approx_eq(lhs_arg, 4);
                            rhs.into_data()
                                .convert::<i32>()
                                .assert_approx_eq(rhs_arg, 4);

                            out
                        }
                    }
                }

                CloneInvarianceTest::<2>::check(&$name);
            }
        };
    }

    mod float {
        use super::*;

        // Unary ops
        clone_invariance_test!(
            unary: AddScalar,
            ops_float: |tensor: TestTensor<2>| tensor.add_scalar(2.0)
        );
        clone_invariance_test!(
            unary: SubScalar,
            ops_float: |tensor: TestTensor<2>| tensor.sub_scalar(2.0)
        );
        clone_invariance_test!(
            unary: DivScalar,
            ops_float: |tensor: TestTensor<2>| tensor.div_scalar(2.0)
        );
        clone_invariance_test!(
            unary: MulScalar,
            ops_float: |tensor: TestTensor<2>| tensor.mul_scalar(2.0)
        );
        clone_invariance_test!(
            unary: PowScalar,
                                        ops_float: |tensor: TestTensor<2>| tensor.powf_scalar(2.0)
        );
        clone_invariance_test!(
            unary: Sqrt,
            ops_float: |tensor: TestTensor<2>| tensor.sqrt()
        );
        clone_invariance_test!(
            unary: Exp,
            ops_float: |tensor: TestTensor<2>| tensor.exp()
        );
        clone_invariance_test!(
            unary: Neg,
            ops_float: |tensor: TestTensor<2>| tensor.neg()
        );
        clone_invariance_test!(
            unary: MeanDim,
            ops_float: |tensor: TestTensor<2>| tensor.mean_dim(1)
        );
        clone_invariance_test!(
            unary: SumDim,
            ops_float: |tensor: TestTensor<2>| tensor.sum_dim(1)
        );
        clone_invariance_test!(
            unary: Sum,
            ops_float: |tensor: TestTensor<2>| tensor.sum().unsqueeze::<2>()
        );
        clone_invariance_test!(
            unary: Mean,
            ops_float: |tensor: TestTensor<2>| tensor.mean().unsqueeze::<2>()
        );
        clone_invariance_test!(
            unary: Clamp,
            ops_float: |tensor: TestTensor<2>| tensor.clamp(-2., 2.)
        );
        clone_invariance_test!(
            unary: ClampMin,
            ops_float: |tensor: TestTensor<2>| tensor.clamp_min(-2.)
        );
        clone_invariance_test!(
            unary: ClampMax,
            ops_float: |tensor: TestTensor<2>| tensor.clamp_max(2.)
        );
        clone_invariance_test!(
            unary: Abs,
            ops_float: |tensor: TestTensor<2>| tensor.abs()
        );
        clone_invariance_test!(
            unary: Cos,
            ops_float: |tensor: TestTensor<2>| tensor.cos()
        );
        clone_invariance_test!(
            unary: Sin,
            ops_float: |tensor: TestTensor<2>| tensor.sin()
        );
        clone_invariance_test!(
            unary: Log,
            ops_float: |tensor: TestTensor<2>| tensor.log()
        );
        clone_invariance_test!(
            unary: Log1P,
            ops_float: |tensor: TestTensor<2>| tensor.log1p()
        );
        clone_invariance_test!(
            unary: SwapDims,
            ops_float: |tensor: TestTensor<2>| tensor.swap_dims(0, 1)
        );
        clone_invariance_test!(
            unary: Transpose,
            ops_float: |tensor: TestTensor<2>| tensor.transpose()
        );
        clone_invariance_test!(
            unary: Slice,
            ops_float: |tensor: TestTensor<2>| tensor.slice([0..12, 12..24])
        );
        clone_invariance_test!(
            unary: Erf,
            ops_float: |tensor: TestTensor<2>| tensor.erf()
        );
        clone_invariance_test!(
            unary: EqualElem,
            ops_float: |tensor: TestTensor<2>| tensor.equal_elem(0.5)
        );
        clone_invariance_test!(
            unary: NotEqualElem,
            ops_float: |tensor: TestTensor<2>| tensor.not_equal_elem(0.5)
        );
        clone_invariance_test!(
            unary: GreaterElem,
            ops_float: |tensor: TestTensor<2>| tensor.greater_elem(0.5)
        );
        clone_invariance_test!(
            unary: GreaterEqualElem,
            ops_float: |tensor: TestTensor<2>| tensor.greater_equal_elem(0.5)
        );
        clone_invariance_test!(
            unary: LowerElem,
            ops_float: |tensor: TestTensor<2>| tensor.lower_elem(0.5)
        );
        clone_invariance_test!(
            unary: LowerEqualElem,
            ops_float: |tensor: TestTensor<2>| tensor.lower_equal_elem(0.5)
        );
        clone_invariance_test!(
            unary: Argmax,
            ops_float: |tensor: TestTensor<2>| tensor.argmax(0)
        );
        clone_invariance_test!(
            unary: Argmin,
            ops_float: |tensor: TestTensor<2>| tensor.argmin(0)
        );
        clone_invariance_test!(
            unary: Max,
            ops_float: |tensor: TestTensor<2>| tensor.max().unsqueeze::<2>()
        );
        clone_invariance_test!(
            unary: Min,
            ops_float: |tensor: TestTensor<2>| tensor.min().unsqueeze::<2>()
        );
        clone_invariance_test!(
            unary: MaxDim,
            ops_float: |tensor: TestTensor<2>| tensor.max_dim(1)
        );
        clone_invariance_test!(
            unary: MaxDimWithIndices,
            ops_float: |tensor: TestTensor<2>| tensor.max_dim_with_indices(1).0
        );
        clone_invariance_test!(
            unary: MinDimWithIndices,
            ops_float: |tensor: TestTensor<2>| tensor.min_dim_with_indices(1).0
        );
        clone_invariance_test!(
            unary: MinDim,
            ops_float: |tensor: TestTensor<2>| tensor.min_dim(1)
        );
        clone_invariance_test!(
            unary: Repeat,
            ops_float: |tensor: TestTensor<2>| {
                tensor.reshape([1, 32, 32]).repeat_dim(0, 4).reshape([4 * 32, 32])
            }
        );
        clone_invariance_test!(
            unary: Reshape,
            ops_float: |tensor: TestTensor<2>| {
                let shape = tensor.shape();
                let new_shape = [shape.num_elements(), 1];
                tensor.reshape(new_shape)
            }
        );
        clone_invariance_test!(
            unary: Gatter,
            ops_float: |tensor: TestTensor<2>| {
                let shape = tensor.shape();
                let indices = TestTensorInt::ones(shape, &Default::default());
                tensor.gather(0, indices)
            }
        );
        clone_invariance_test!(
            unary: Select,
            ops_float: |tensor: TestTensor<2>| {
                let indices = TestTensorInt::from_ints([1, 2, 0, 5], &Default::default());
                tensor.select(0, indices)
            }
        );
        clone_invariance_test!(
            unary: MaskFill,
            ops_float: |tensor: TestTensor<2>| {
                let mask = tensor.clone().greater_elem(0.5);
                tensor.mask_fill(mask, 77.0)
            }
        );

        // Activation
        clone_invariance_test!(
            unary: Softmax,
            ops_float: |tensor: TestTensor<2>| softmax(tensor, 1)
        );
        clone_invariance_test!(
            unary: LogSoftmax,
            ops_float: |tensor: TestTensor<2>| log_softmax(tensor, 1)
        );
        clone_invariance_test!(
            unary: Sigmoid,
            ops_float: |tensor: TestTensor<2>| sigmoid(tensor)
        );
        clone_invariance_test!(
            unary: LogSigmoid,
            ops_float: |tensor: TestTensor<2>| log_sigmoid(tensor)
        );
        clone_invariance_test!(
            unary: Relu,
            ops_float: |tensor: TestTensor<2>| relu(tensor)
        );
        clone_invariance_test!(
            unary: Gelu,
            ops_float: |tensor: TestTensor<2>| gelu(tensor)
        );
        clone_invariance_test!(
            unary: Mish,
            ops_float: |tensor: TestTensor<2>| mish(tensor)
        );
        clone_invariance_test!(
            unary: Silu,
            ops_float: |tensor: TestTensor<2>| silu(tensor)
        );
        clone_invariance_test!(
            unary: Softplus,
            ops_float: |tensor: TestTensor<2>| softplus(tensor, 1.0)
        );
        clone_invariance_test!(
            unary: Tanh,
            ops_float: |tensor: TestTensor<2>| tanh(tensor)
        );

        // Binary ops
        clone_invariance_test!(
            binary: Add,
            ops_float: |lhs: TestTensor<2>, rhs: TestTensor<2>| lhs.add(rhs)
        );
        clone_invariance_test!(
            binary: Sub,
            ops_float: |lhs: TestTensor<2>, rhs: TestTensor<2>| lhs.sub(rhs)
        );
        clone_invariance_test!(
            binary: Div,
            ops_float: |lhs: TestTensor<2>, rhs: TestTensor<2>| lhs.div(rhs)
        );
        clone_invariance_test!(
            binary: Mul,
            ops_float: |lhs: TestTensor<2>, rhs: TestTensor<2>| lhs.mul(rhs)
        );
        clone_invariance_test!(
            binary: Matmul,
            ops_float: |lhs: TestTensor<2>, rhs: TestTensor<2>| lhs.matmul(rhs)
        );
        clone_invariance_test!(
            binary: Equal,
            ops_float: |lhs: TestTensor<2>, rhs: TestTensor<2>| lhs.equal(rhs)
        );
        clone_invariance_test!(
            binary: Greater,
            ops_float: |lhs: TestTensor<2>, rhs: TestTensor<2>| lhs.greater(rhs)
        );
        clone_invariance_test!(
            binary: GreaterEqual,
            ops_float: |lhs: TestTensor<2>, rhs: TestTensor<2>| lhs.greater_equal(rhs)
        );
        clone_invariance_test!(
            binary: Lower,
            ops_float: |lhs: TestTensor<2>, rhs: TestTensor<2>| lhs.lower(rhs)
        );
        clone_invariance_test!(
            binary: LowerEqual,
            ops_float: |lhs: TestTensor<2>, rhs: TestTensor<2>| lhs.lower_equal(rhs)
        );
        clone_invariance_test!(
            binary: Cat,
            ops_float: |lhs: TestTensor<2>, rhs: TestTensor<2>| {
                let lhs = lhs.reshape([1usize, 32, 32]);
                let rhs = rhs.reshape([1usize, 32, 32]);

                TestTensor::cat(vec![lhs, rhs], 0).reshape([64, 32])
            }
        );
        clone_invariance_test!(
            binary: Scatter,
            ops_float: |tensor: TestTensor<2>, values: TestTensor<2>| {
                let shape = tensor.shape();
                let indices = TestTensorInt::ones(shape, &Default::default());
                tensor.scatter(0, indices, values)
            }
        );
        clone_invariance_test!(
            binary: SliceAssign,
            ops_float: |tensor: TestTensor<2>, values: TestTensor<2>| {
                tensor.slice_assign([0..12, 12..24], values.slice([12..24, 0..12]))
            }
        );
        clone_invariance_test!(
            binary: MaskWhere,
            ops_float: |tensor: TestTensor<2>, values: TestTensor<2>| {
                let mask = tensor.clone().greater_elem(0.5);
                tensor.mask_where(mask, values)
            }
        );
        clone_invariance_test!(
            binary: SelectAssign,
            ops_float: |tensor: TestTensor<2>, values: TestTensor<2>| {
                let indices = TestTensorInt::from_ints([1, 2, 0, 5], &Default::default());
                let values = values.select(0, indices.clone());
                tensor.select_assign(0, indices, values)
            }
        );
    }

    mod int {
        use super::*;

        // Unary ops
        clone_invariance_test!(
            unary: AddScalar,
            ops_int: |tensor: TestTensorInt<2>| tensor.add_scalar(2.0)
        );
        clone_invariance_test!(
            unary: SubScalar,
            ops_int: |tensor: TestTensorInt<2>| tensor.sub_scalar(2.0)
        );
        clone_invariance_test!(
            unary: DivScalar,
            ops_int: |tensor: TestTensorInt<2>| tensor.div_scalar(2.0)
        );
        clone_invariance_test!(
            unary: MulScalar,
            ops_int: |tensor: TestTensorInt<2>| tensor.mul_scalar(2.0)
        );
        clone_invariance_test!(
            unary: Neg,
            ops_int: |tensor: TestTensorInt<2>| tensor.neg()
        );
        clone_invariance_test!(
            unary: MeanDim,
            ops_int: |tensor: TestTensorInt<2>| tensor.mean_dim(1)
        );
        clone_invariance_test!(
            unary: SumDim,
            ops_int: |tensor: TestTensorInt<2>| tensor.sum_dim(1)
        );
        clone_invariance_test!(
            unary: Sum,
            ops_int: |tensor: TestTensorInt<2>| tensor.sum().unsqueeze::<2>()
        );
        clone_invariance_test!(
            unary: Mean,
            ops_int: |tensor: TestTensorInt<2>| tensor.mean().unsqueeze::<2>()
        );
        clone_invariance_test!(
            unary: Clamp,
            ops_int: |tensor: TestTensorInt<2>| tensor.clamp(-2., 2.)
        );
        clone_invariance_test!(
            unary: ClampMin,
            ops_int: |tensor: TestTensorInt<2>| tensor.clamp_min(-2.)
        );
        clone_invariance_test!(
            unary: ClampMax,
            ops_int: |tensor: TestTensorInt<2>| tensor.clamp_max(2.)
        );
        clone_invariance_test!(
            unary: Abs,
            ops_int: |tensor: TestTensorInt<2>| tensor.abs()
        );
        clone_invariance_test!(
            unary: SwapDims,
            ops_int: |tensor: TestTensorInt<2>| tensor.swap_dims(0, 1)
        );
        clone_invariance_test!(
            unary: Transpose,
            ops_int: |tensor: TestTensorInt<2>| tensor.transpose()
        );
        clone_invariance_test!(
            unary: Slice,
            ops_int: |tensor: TestTensorInt<2>| tensor.slice([0..12, 12..24])
        );
        clone_invariance_test!(
            unary: EqualElem,
            ops_int: |tensor: TestTensorInt<2>| tensor.equal_elem(25)
        );
        clone_invariance_test!(
            unary: NotEqualElem,
            ops_int: |tensor: TestTensorInt<2>| tensor.not_equal_elem(25)
        );
        clone_invariance_test!(
            unary: GreaterElem,
            ops_int: |tensor: TestTensorInt<2>| tensor.greater_elem(25)
        );
        clone_invariance_test!(
            unary: GreaterEqualElem,
            ops_int: |tensor: TestTensorInt<2>| tensor.greater_equal_elem(25)
        );
        clone_invariance_test!(
            unary: LowerElem,
            ops_int: |tensor: TestTensorInt<2>| tensor.lower_elem(25)
        );
        clone_invariance_test!(
            unary: LowerEqualElem,
            ops_int: |tensor: TestTensorInt<2>| tensor.lower_equal_elem(25)
        );
        clone_invariance_test!(
            unary: Argmax,
            ops_int: |tensor: TestTensorInt<2>| tensor.argmax(0)
        );
        clone_invariance_test!(
            unary: Argmin,
            ops_int: |tensor: TestTensorInt<2>| tensor.argmin(0)
        );
        clone_invariance_test!(
            unary: Max,
            ops_int: |tensor: TestTensorInt<2>| tensor.max().unsqueeze::<2>()
        );
        clone_invariance_test!(
            unary: Min,
            ops_int: |tensor: TestTensorInt<2>| tensor.min().unsqueeze::<2>()
        );
        clone_invariance_test!(
            unary: MaxDim,
            ops_int: |tensor: TestTensorInt<2>| tensor.max_dim(1)
        );
        clone_invariance_test!(
            unary: MaxDimWithIndices,
            ops_int: |tensor: TestTensorInt<2>| tensor.max_dim_with_indices(1).0
        );
        clone_invariance_test!(
            unary: MinDimWithIndices,
            ops_int: |tensor: TestTensorInt<2>| tensor.min_dim_with_indices(1).0
        );
        clone_invariance_test!(
            unary: MinDim,
            ops_int: |tensor: TestTensorInt<2>| tensor.min_dim(1)
        );
        clone_invariance_test!(
            unary: Repeat,
            ops_int: |tensor: TestTensorInt<2>| {
                tensor.reshape([1, 32, 32]).repeat_dim(0, 4).reshape([4 * 32, 32])
            }
        );
        clone_invariance_test!(
            unary: Reshape,
            ops_int: |tensor: TestTensorInt<2>| {
                let shape = tensor.shape();
                let new_shape = [shape.num_elements(), 1];
                tensor.reshape(new_shape)
            }
        );
        clone_invariance_test!(
            unary: Gatter,
            ops_int: |tensor: TestTensorInt<2>| {
                let shape = tensor.shape();
                let indices = TestTensorInt::ones(shape, &Default::default());
                tensor.gather(0, indices)
            }
        );
        clone_invariance_test!(
            unary: Select,
            ops_int: |tensor: TestTensorInt<2>| {
                let indices = TestTensorInt::from_ints([1, 2, 0, 5], &Default::default());
                tensor.select(0, indices)
            }
        );
        clone_invariance_test!(
            unary: MaskFill,
            ops_int: |tensor: TestTensorInt<2>| {
                let mask = tensor.clone().greater_elem(0.5);
                tensor.mask_fill(mask, 77.0)
            }
        );

        // Binary ops
        clone_invariance_test!(
            binary: Add,
            ops_int: |lhs: TestTensorInt<2>, rhs: TestTensorInt<2>| lhs.add(rhs)
        );
        clone_invariance_test!(
            binary: Sub,
            ops_int: |lhs: TestTensorInt<2>, rhs: TestTensorInt<2>| lhs.sub(rhs)
        );
        clone_invariance_test!(
            binary: Div,
            ops_int: |lhs: TestTensorInt<2>, rhs: TestTensorInt<2>| lhs.div(rhs)
        );
        clone_invariance_test!(
            binary: Mul,
            ops_int: |lhs: TestTensorInt<2>, rhs: TestTensorInt<2>| lhs.mul(rhs)
        );
        clone_invariance_test!(
            binary: Equal,
            ops_int: |lhs: TestTensorInt<2>, rhs: TestTensorInt<2>| lhs.equal(rhs)
        );
        clone_invariance_test!(
            binary: NotEqual,
            ops_int: |lhs: TestTensorInt<2>, rhs: TestTensorInt<2>| lhs.not_equal(rhs)
        );
        clone_invariance_test!(
            binary: Greater,
            ops_int: |lhs: TestTensorInt<2>, rhs: TestTensorInt<2>| lhs.greater(rhs)
        );
        clone_invariance_test!(
            binary: GreaterEqual,
            ops_int: |lhs: TestTensorInt<2>, rhs: TestTensorInt<2>| lhs.greater_equal(rhs)
        );
        clone_invariance_test!(
            binary: Lower,
            ops_int: |lhs: TestTensorInt<2>, rhs: TestTensorInt<2>| lhs.lower(rhs)
        );
        clone_invariance_test!(
            binary: LowerEqual,
            ops_int: |lhs: TestTensorInt<2>, rhs: TestTensorInt<2>| lhs.lower_equal(rhs)
        );
        clone_invariance_test!(
            binary: Cat,
            ops_int: |lhs: TestTensorInt<2>, rhs: TestTensorInt<2>| {
                let lhs = lhs.reshape([1usize, 32, 32]);
                let rhs = rhs.reshape([1usize, 32, 32]);

                TestTensorInt::cat(vec![lhs, rhs], 0).reshape([64, 32])
            }
        );
        clone_invariance_test!(
            binary: Scatter,
            ops_int: |tensor: TestTensorInt<2>, values: TestTensorInt<2>| {
                let shape = tensor.shape();
                let indices = TestTensorInt::ones(shape, &Default::default());
                tensor.scatter(0, indices, values)
            }
        );
        clone_invariance_test!(
            binary: SliceAssign,
            ops_int: |tensor: TestTensorInt<2>, values: TestTensorInt<2>| {
                tensor.slice_assign([0..12, 12..24], values.slice([12..24, 0..12]))
            }
        );
        clone_invariance_test!(
            binary: MaskWhere,
            ops_int: |tensor: TestTensorInt<2>, values: TestTensorInt<2>| {
                let mask = tensor.clone().greater_elem(0.5);
                tensor.mask_where(mask, values)
            }
        );
        clone_invariance_test!(
            binary: SelectAssign,
            ops_int: |tensor: TestTensorInt<2>, values: TestTensorInt<2>| {
                let indices = TestTensorInt::from_ints([1, 2, 0, 5], &Default::default());
                let values = values.select(0, indices.clone());
                tensor.select_assign(0, indices, values)
            }
        );
    }
}
