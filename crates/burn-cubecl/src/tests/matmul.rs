#[burn_tensor_testgen::testgen(matmul)]
mod tests {
    use super::*;
    use burn_cubecl::kernel::matmul::{matmul, MatmulStrategy};
    use burn_tensor::{Shape, Tensor, TensorPrimitive};

    mod simple {
        use super::*;

        #[test]
        pub fn straightforward() {
            test_with_params::<2, 2>(1, 2, 1, 1, 1);
        }

        #[test]
        pub fn shapes_smaller_than_blocks() {
            test_with_params::<16, 16>(8, 8, 8, 1, 1);
        }

        #[test]
        pub fn n_smaller_than_m() {
            test_with_params::<2, 2>(8, 8, 3, 1, 1);
        }

        #[test]
        pub fn m_smaller_than_n() {
            test_with_params::<2, 2>(3, 8, 8, 1, 1);
        }

        #[test]
        pub fn k_smaller_than_m_n() {
            test_with_params::<2, 2>(8, 3, 8, 1, 1);
        }

        #[test]
        pub fn k_larger_than_m_n() {
            test_with_params::<2, 2>(8, 48, 8, 1, 1);
        }

        #[test]
        pub fn multibatch_1_dim() {
            test_with_params::<2, 2>(8, 8, 8, 3, 1);
        }

        #[test]
        pub fn multibatch_2_dims() {
            test_with_params::<2, 2>(8, 8, 8, 3, 4);
        }

        #[test]
        pub fn blocks_divide_shapes_unevenly() {
            test_with_params::<3, 3>(7, 7, 7, 1, 1);
        }

        #[test]
        fn swapped_batches_no_padding() {
            let strategy = MatmulStrategy::Simple {
                grid_x: 2,
                grid_y: 2,
            };
            let swap = [0, 1];
            let shape_lhs = [3, 2, 4, 4];
            let shape_rhs = [3, 2, 4, 4];
            same_as_reference_swapped_dims(strategy, swap, swap, shape_lhs, shape_rhs);
        }

        #[test]
        fn swapped_row_col_no_padding() {
            let strategy = MatmulStrategy::Simple {
                grid_x: 2,
                grid_y: 2,
            };
            let swap_lhs = [0, 0];
            let swap_rhs = [2, 3];
            let shape_lhs = [3, 2, 4, 4];
            let shape_rhs = [3, 2, 4, 4];
            same_as_reference_swapped_dims(strategy, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
        }

        #[test]
        fn swapped_row_with_batch_no_padding() {
            let strategy = MatmulStrategy::Simple {
                grid_x: 2,
                grid_y: 2,
            };
            let swap_lhs = [0, 3];
            let swap_rhs = [0, 2];
            let shape_lhs = [4, 4, 4, 4];
            let shape_rhs = [4, 4, 4, 4];
            same_as_reference_swapped_dims(strategy, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
        }

        fn test_with_params<const WORKGROUP_SIZE_X: usize, const WORKGROUP_SIZE_Y: usize>(
            m: usize,
            k: usize,
            n: usize,
            batch_1: usize,
            batch_2: usize,
        ) {
            let shape_lhs = [batch_1, batch_2, m, k];
            let shape_rhs = [batch_1, batch_2, k, n];
            same_as_reference(
                MatmulStrategy::Simple {
                    grid_x: WORKGROUP_SIZE_X,
                    grid_y: WORKGROUP_SIZE_Y,
                },
                shape_lhs,
                shape_rhs,
            );
        }
    }

    mod padding {
        use super::*;
        use burn_cubecl::kernel::matmul::padding::{crop, pad_round};
        use burn_tensor::backend::Backend;

        fn padding_already_round_should_have_same_shape() {
            let row = 10;
            let row_divisor = 5;
            let col = 12;
            let col_divisor = 3;
            let tensor = TestTensor::random(
                [row, col],
                burn_tensor::Distribution::Default,
                &Default::default(),
            );
            let expected_shape = [row, col].into();

            let padded =
                pad_round(tensor.into_primitive().tensor(), row_divisor, col_divisor).into_tensor();

            assert!(padded.shape == expected_shape);
        }

        #[test]
        fn padding_already_round_should_have_same_values() {
            let row = 10;
            let row_divisor = 5;
            let col = 12;
            let col_divisor = 3;
            let tensor = TestTensor::random(
                [row, col],
                burn_tensor::Distribution::Default,
                &Default::default(),
            );

            let padded = pad_round(
                tensor.clone().into_primitive().tensor(),
                row_divisor,
                col_divisor,
            );

            let padded = TestTensor::from_primitive(TensorPrimitive::Float((padded.into_tensor())));
            padded.into_data().assert_approx_eq(&tensor.into_data(), 3);
        }

        #[test]
        fn padding_not_round_should_have_rounded_shape() {
            let row = 10;
            let row_divisor = 6;
            let col = 12;
            let col_divisor = 5;
            let tensor = TestTensor::random(
                [row, col],
                burn_tensor::Distribution::Default,
                &Default::default(),
            );
            let expected_shape = [12, 15].into();

            let padded =
                pad_round(tensor.into_primitive().tensor(), row_divisor, col_divisor).into_tensor();

            assert!(padded.shape == expected_shape);
        }

        #[test]
        fn padding_not_round_should_have_same_values() {
            let row = 10;
            let row_divisor = 6;
            let col = 12;
            let col_divisor = 5;
            let tensor = TestTensor::random(
                [row, col],
                burn_tensor::Distribution::Default,
                &Default::default(),
            );

            let padded = pad_round(
                tensor.clone().into_primitive().tensor(),
                row_divisor,
                col_divisor,
            )
            .into_tensor();

            let padded = TestTensor::from_primitive(TensorPrimitive::Float(padded)).to_data();
            let padded = padded
                .as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap();
            let tensor = tensor.into_data();
            let tensor = tensor
                .as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap();
            for i in 0..row {
                for j in 0..col {
                    assert!(padded[i * 15 + j] == tensor[i * col + j]);
                }
            }
        }

        #[test]
        fn padding_not_round_should_have_zero_padding() {
            let row = 10;
            let row_divisor = 6;
            let col = 12;
            let col_divisor = 5;
            let tensor = TestTensor::random(
                [row, col],
                burn_tensor::Distribution::Default,
                &Default::default(),
            );

            let padded =
                pad_round(tensor.into_primitive().tensor(), row_divisor, col_divisor).into_tensor();
            let padded = TestTensor::from_primitive(TensorPrimitive::Float(padded)).to_data();
            let padded = padded
                .as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap();

            // check right of matrix
            for i in 0..row {
                for j in col..15 {
                    assert!(padded[i * 15 + j] == 0.0);
                }
            }
            // check below matrix, including bottom right
            for i in row..12 {
                for j in 0..15 {
                    assert!(padded[i * 15 + j] == 0.0);
                }
            }
        }

        #[test]
        fn padding_works_with_batch() {
            let row = 10;
            let row_divisor = 4;
            let col = 12;
            let col_divisor = 5;
            let tensor = TestTensor::random(
                [2, 3, row, col],
                burn_tensor::Distribution::Default,
                &Default::default(),
            );
            let expected_shape = [2, 3, 12, 15].into();

            let padded =
                pad_round(tensor.into_primitive().tensor(), row_divisor, col_divisor).into_tensor();

            assert!(padded.shape == expected_shape);
        }

        #[test]
        fn padding_with_row_divisor_larger_than_row() {
            let row = 10;
            let row_divisor = 32;
            let col = 4;
            let col_divisor = 3;
            let tensor = TestTensor::random(
                [row, col],
                burn_tensor::Distribution::Default,
                &Default::default(),
            );
            let expected_shape = [row_divisor, 2 * col_divisor].into();

            let padded =
                pad_round(tensor.into_primitive().tensor(), row_divisor, col_divisor).into_tensor();

            assert!(padded.shape == expected_shape);
        }

        #[test]
        fn padding_with_row_divisor_equal_to_row_but_col_must_be_padded() {
            let row = 32;
            let row_divisor = 32;
            let col = 4;
            let col_divisor = 64;
            let tensor = TestTensor::random(
                [row, col],
                burn_tensor::Distribution::Default,
                &Default::default(),
            );
            let expected_shape = [32, 64].into();

            let padded =
                pad_round(tensor.into_primitive().tensor(), row_divisor, col_divisor).into_tensor();

            assert!(padded.shape == expected_shape);
        }

        #[test]
        fn crop_same_shape_should_be_unchanged_shape() {
            let row = 10;
            let col = 12;
            let device = Default::default();
            let tensor =
                TestTensor::random([row, col], burn_tensor::Distribution::Default, &device);
            let expected_shape = [row, col].into();

            let unpadded = crop(
                tensor.clone().into_primitive().tensor(),
                TestTensor::empty([row, col], &device)
                    .into_primitive()
                    .tensor(),
            );

            assert!(unpadded.shape == expected_shape);
        }

        #[test]
        fn crop_same_shape_should_have_unchanged_values() {
            let row = 10;
            let col = 12;
            let device = Default::default();
            let tensor =
                TestTensor::random([row, col], burn_tensor::Distribution::Default, &device);

            let unpadded = crop(
                tensor.clone().into_primitive().tensor(),
                TestTensor::empty([row, col], &device)
                    .into_primitive()
                    .tensor(),
            );

            let unpadded = TestTensor::from_primitive(TensorPrimitive::Float(unpadded)).to_data();
            let unpadded = unpadded
                .as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap();
            let tensor = tensor.into_data();
            let tensor = tensor
                .as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap();
            for i in 0..row {
                for j in 0..col {
                    assert!(unpadded[i * col + j] == tensor[i * col + j]);
                }
            }
        }

        #[test]
        fn crop_should_decrease_shape() {
            let row = 10;
            let keep_rows = 8;
            let col = 12;
            let keep_cols = 10;
            let device = Default::default();
            let tensor =
                TestTensor::random([row, col], burn_tensor::Distribution::Default, &device);
            let expected_shape = [keep_rows, keep_cols].into();

            let unpadded = crop(
                tensor.clone().into_primitive().tensor(),
                TestTensor::empty([keep_rows, keep_cols], &device)
                    .into_primitive()
                    .tensor(),
            );

            assert!(unpadded.shape == expected_shape);
        }

        #[test]
        fn crop_should_keep_same_values() {
            let row = 4;
            let keep_rows = 3;
            let col = 4;
            let keep_cols = 3;
            let device = Default::default();
            let tensor =
                TestTensor::random([row, col], burn_tensor::Distribution::Default, &device);

            let unpadded = crop(
                tensor.clone().into_primitive().tensor(),
                TestTensor::empty([keep_rows, keep_cols], &device)
                    .into_primitive()
                    .tensor(),
            );

            let unpadded = TestTensor::from_primitive(TensorPrimitive::Float(unpadded)).to_data();
            let unpadded = unpadded
                .as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap();
            let tensor = tensor.into_data();
            let tensor = tensor
                .as_slice::<<TestBackend as Backend>::FloatElem>()
                .unwrap();

            for i in 0..keep_rows {
                for j in 0..keep_cols {
                    assert!(unpadded[i * keep_cols + j] == tensor[i * col + j]);
                }
            }
        }
    }

    fn same_as_reference<const D: usize, S>(strategy: MatmulStrategy, shape_lhs: S, shape_rhs: S)
    where
        S: Into<Shape<D>>,
    {
        let ref_tensor_device = Default::default();
        let x = ReferenceTensor::random(
            shape_lhs,
            burn_tensor::Distribution::Uniform(-1.0, 1.0),
            &ref_tensor_device,
        );
        let y = ReferenceTensor::random(
            shape_rhs,
            burn_tensor::Distribution::Uniform(-1.0, 1.0),
            &ref_tensor_device,
        );

        let test_tensor_device = Default::default();
        let x_jit = TestTensor::from_data(x.to_data(), &test_tensor_device);
        let y_jit = TestTensor::from_data(y.to_data(), &test_tensor_device);

        let z_reference = x.matmul(y);
        let z = Tensor::<TestBackend, D>::from_primitive(TensorPrimitive::Float(matmul(
            x_jit.into_primitive().tensor(),
            y_jit.into_primitive().tensor(),
            strategy,
        )));

        z_reference.into_data().assert_approx_eq(&z.into_data(), 3);
    }

    fn same_as_reference_swapped_dims<const D: usize, S>(
        strategy: MatmulStrategy,
        swap_lhs: [usize; 2],
        swap_rhs: [usize; 2],
        shape_lhs: S,
        shape_rhs: S,
    ) where
        S: Into<Shape<D>>,
    {
        let x = ReferenceTensor::random(
            shape_lhs,
            burn_tensor::Distribution::Uniform(-1.0, 1.0),
            &Default::default(),
        );
        let y = ReferenceTensor::random(
            shape_rhs,
            burn_tensor::Distribution::Uniform(-1.0, 1.0),
            &Default::default(),
        );

        let x_jit = TestTensor::from_data(x.to_data(), &Default::default())
            .swap_dims(swap_lhs[0], swap_lhs[1]);
        let y_jit = TestTensor::from_data(y.to_data(), &Default::default())
            .swap_dims(swap_rhs[0], swap_rhs[1]);

        let z_reference = x
            .swap_dims(swap_lhs[0], swap_lhs[1])
            .matmul(y.swap_dims(swap_rhs[0], swap_rhs[1]));

        let z = Tensor::<TestBackend, D>::from_primitive(TensorPrimitive::Float(matmul(
            x_jit.into_primitive().tensor(),
            y_jit.into_primitive().tensor(),
            strategy,
        )));

        z_reference.into_data().assert_approx_eq(&z.into_data(), 3);
    }
}
