#[burn_tensor_testgen::testgen(gather_scatter)]
mod tests {
    use super::*;
    use burn_tensor::{IndexingUpdateOp, Tensor, TensorData};

    #[test]
    fn should_gather_1d_dim0() {
        let device = Default::default();
        let tensor = TestTensor::<1>::from_floats([0.0, 1.0, 2.0], &device);
        let indices = TestTensorInt::from_ints([1, 1, 0, 1, 2], &device);

        let output = tensor.gather(0, indices);

        output
            .into_data()
            .assert_eq(&TensorData::from([1.0, 1.0, 0.0, 1.0, 2.0]), false);
    }

    #[test]
    fn should_gather_1d_dim0_int() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::from_ints([5, 6, 7], &device);
        let indices = TestTensorInt::from_ints([1, 1, 0, 1, 2], &device);

        let output = tensor.gather(0, indices);

        output
            .into_data()
            .assert_eq(&TensorData::from([6, 6, 5, 6, 7]), false);
    }

    #[test]
    fn should_gather_2d_dim0() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
        let indices = TestTensorInt::from_ints([[0, 1, 0], [1, 0, 1]], &device);

        let output = tensor.gather(0, indices);

        output
            .into_data()
            .assert_eq(&TensorData::from([[0.0, 4.0, 2.0], [3.0, 1.0, 5.0]]), false);
    }

    #[test]
    fn should_gather_2d_dim1() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
        let indices = TestTensorInt::from_ints([[2, 1, 0, 0], [2, 0, 1, 2]], &device);

        let output = tensor.gather(1, indices);

        output.into_data().assert_eq(
            &TensorData::from([[2.0, 1.0, 0.0, 0.0], [5.0, 3.0, 4.0, 5.0]]),
            false,
        );
    }

    #[test]
    fn should_gather_3d_dim1() {
        let device = Default::default();
        let tensor = TestTensor::<3>::from_floats(
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            ],
            &device,
        );
        let indices =
            TestTensorInt::from_ints([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 1]]], &device);

        let output = tensor.gather(1, indices);
        let expected = TensorData::from([
            [[3.0, 1.0, 2.0], [0.0, 4.0, 2.0]],
            [[6.0, 7.0, 11.0], [6.0, 10.0, 11.0]],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_gather_2d_only_1dim() {
        let device = Default::default();
        let tensor = TestTensor::from_floats([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], &device);
        let indices = TestTensorInt::<2>::from_ints([[1, 2]], &device).reshape([2, 1]);

        let output = tensor.gather(1, indices);

        output
            .into_data()
            .assert_eq(&TensorData::from([[1.0], [5.0]]), false);
    }

    #[test]
    fn should_gather_indices_broadcasted() {
        let device = Default::default();

        let batch_size = 3;
        let fft_size = 4;
        let shape = [batch_size, fft_size, 2];
        let x = TestTensorInt::arange(
            0..shape.iter().product::<usize>() as i64,
            &Default::default(),
        )
        .reshape(shape);
        let idx = TestTensorInt::<1>::from_ints([0, 2, 1, 3], &device);

        let expected = TestTensorInt::<3>::from([
            [[0, 1], [4, 5], [2, 3], [6, 7]],
            [[8, 9], [12, 13], [10, 11], [14, 15]],
            [[16, 17], [20, 21], [18, 19], [22, 23]],
        ])
        .into_data();

        // Case 1: gather dim 2
        let perm = idx
            .clone()
            .reshape([1, 1, fft_size])
            .repeat_dim(0, batch_size)
            .repeat_dim(1, 2);

        let input = x.clone().permute([0, 2, 1]);
        let out = input.gather(2, perm).permute([0, 2, 1]);

        out.into_data().assert_eq(&expected, true);

        // Case 2: gather directly on dim 1
        let perm = idx.reshape([1, fft_size, 1]).repeat_dim(0, batch_size);
        let out2 = x.gather(1, perm.repeat_dim(2, 2));

        out2.into_data().assert_eq(&expected, true);
    }

    #[test]
    fn should_scatter_add_1d() {
        let device = Default::default();
        let tensor = TestTensor::<1>::from_floats([0.0, 0.0, 0.0], &device);
        let values = TestTensor::from_floats([5.0, 4.0, 3.0], &device);
        let indices = TestTensorInt::from_ints([1, 0, 2], &device);

        let output = tensor.scatter(0, indices, values, IndexingUpdateOp::Add);

        output
            .into_data()
            .assert_eq(&TensorData::from([4.0, 5.0, 3.0]), false);
    }

    #[test]
    fn should_scatter_add_1d_int() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::from_ints([0, 0, 0], &device);
        let values = TestTensorInt::from_ints([5, 4, 3], &device);
        let indices = TestTensorInt::from_ints([1, 0, 2], &device);

        let output = tensor.scatter(0, indices, values, IndexingUpdateOp::Add);

        output
            .into_data()
            .assert_eq(&TensorData::from([4, 5, 3]), false);
    }

    #[test]
    fn should_scatter_add_2d_dim0() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_floats([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], &device);
        let values = TestTensor::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let indices = TestTensorInt::from_ints([[1, 0, 1], [1, 1, 0]], &device);

        let output = tensor.scatter(0, indices, values, IndexingUpdateOp::Add);

        output
            .into_data()
            .assert_eq(&TensorData::from([[0.0, 2.0, 6.0], [5.0, 5.0, 3.0]]), false);
    }

    #[test]
    fn should_scatter_add_2d_dim1() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_floats([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], &device);
        let values = TestTensor::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let indices = TestTensorInt::from_ints([[1, 0, 2], [1, 2, 0]], &device);

        let output = tensor.scatter(1, indices, values, IndexingUpdateOp::Add);

        output
            .into_data()
            .assert_eq(&TensorData::from([[2.0, 1.0, 3.0], [6.0, 4.0, 5.0]]), false);
    }

    #[test]
    fn should_scatter_add_3d_dim1() {
        let device = Default::default();
        let tensor = TestTensor::<3>::from_floats(
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
            ],
            &device,
        );
        let values = TestTensor::from_floats(
            [
                [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
                [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
            ],
            &device,
        );
        let indices =
            TestTensorInt::from_ints([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 1]]], &device);

        let output = tensor.scatter(1, indices, values, IndexingUpdateOp::Add);
        let expected = TensorData::from([
            [[15.0, 14.0, 33.0], [15.0, 20.0, 5.0]],
            [[45.0, 26.0, 8.0], [9.0, 32.0, 54.0]],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_scatter_add_2d_dim1_diff_shape() {
        let device = Default::default();
        let tensor = TestTensor::<2>::from_floats([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], &device);
        let values = TestTensor::from_floats([[1.0], [4.0]], &device);
        let indices = TestTensorInt::from_ints([[1], [2]], &device);

        let output = tensor.scatter(1, indices, values, IndexingUpdateOp::Add);

        output
            .into_data()
            .assert_eq(&TensorData::from([[0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]), false);
    }

    #[test]
    #[should_panic]
    fn scatter_should_panic_on_mismatch_of_shapes() {
        let device = Default::default();
        let tensor = TestTensor::<1>::from_floats([0.0, 0.0, 0.0], &device);
        let values = TestTensor::from_floats([5.0, 4.0], &device);
        let indices = TestTensorInt::from_ints([1, 0, 2], &device);

        tensor.scatter(0, indices, values, IndexingUpdateOp::Add);
    }
}
