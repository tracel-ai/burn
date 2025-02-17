#[burn_tensor_testgen::testgen(repeat)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Bool, Int, Tensor, TensorData};

    #[test]
    fn should_support_repeat_ops_one_dimension() {
        let data = TensorData::from([[0.0f32, 1.0f32, 2.0f32]]);
        let tensor = TestTensor::<2>::from_data(data, &Default::default());

        let output = tensor.repeat(&[4, 1, 1]);
        let expected = TensorData::from([
            [0.0f32, 1.0f32, 2.0f32],
            [0.0f32, 1.0f32, 2.0f32],
            [0.0f32, 1.0f32, 2.0f32],
            [0.0f32, 1.0f32, 2.0f32],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_bool_repeat_ops_one_dimension() {
        let data = TensorData::from([[true, false, false]]);
        let tensor = TestTensorBool::<2>::from_data(data, &Default::default());

        let output = tensor.repeat(&[4, 1, 1]);
        let expected = TensorData::from([
            [true, false, false],
            [true, false, false],
            [true, false, false],
            [true, false, false],
        ]);
        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_int_repeat_ops_one_dimension() {
        let data = TensorData::from([[0i32, 1i32, 2i32]]);
        let tensor = TestTensorInt::<2>::from_data(data, &Default::default());

        let output = tensor.repeat(&[4, 1, 1]);
        let expected = TensorData::from([
            [0i32, 1i32, 2i32],
            [0i32, 1i32, 2i32],
            [0i32, 1i32, 2i32],
            [0i32, 1i32, 2i32],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_float_repeat_repeating_on_many_dimensions() {
        let data = TensorData::from([
            [[1.0f32, 2.0f32], [3.0f32, 4.0f32]],
            [[5.0f32, 6.0f32], [7.0f32, 8.0f32]],
            [[9.0f32, 10.0f32], [11.0f32, 12.0f32]],
            [[13.0f32, 14.0f32], [15.0f32, 16.0f32]],
        ]);
        let tensor = TestTensor::<3>::from_data(data, &Default::default());

        let output = tensor.repeat(&[2, 3, 2]);
        let expected = TensorData::from([
            [
                [1.0f32, 2.0f32, 1.0f32, 2.0f32],
                [3.0f32, 4.0f32, 3.0f32, 4.0f32],
                [1.0f32, 2.0f32, 1.0f32, 2.0f32],
                [3.0f32, 4.0f32, 3.0f32, 4.0f32],
                [1.0f32, 2.0f32, 1.0f32, 2.0f32],
                [3.0f32, 4.0f32, 3.0f32, 4.0f32],
            ],
            [
                [5.0f32, 6.0f32, 5.0f32, 6.0f32],
                [7.0f32, 8.0f32, 7.0f32, 8.0f32],
                [5.0f32, 6.0f32, 5.0f32, 6.0f32],
                [7.0f32, 8.0f32, 7.0f32, 8.0f32],
                [5.0f32, 6.0f32, 5.0f32, 6.0f32],
                [7.0f32, 8.0f32, 7.0f32, 8.0f32],
            ],
            [
                [9.0f32, 10.0f32, 9.0f32, 10.0f32],
                [11.0f32, 12.0f32, 11.0f32, 12.0f32],
                [9.0f32, 10.0f32, 9.0f32, 10.0f32],
                [11.0f32, 12.0f32, 11.0f32, 12.0f32],
                [9.0f32, 10.0f32, 9.0f32, 10.0f32],
                [11.0f32, 12.0f32, 11.0f32, 12.0f32],
            ],
            [
                [13.0f32, 14.0f32, 13.0f32, 14.0f32],
                [15.0f32, 16.0f32, 15.0f32, 16.0f32],
                [13.0f32, 14.0f32, 13.0f32, 14.0f32],
                [15.0f32, 16.0f32, 15.0f32, 16.0f32],
                [13.0f32, 14.0f32, 13.0f32, 14.0f32],
                [15.0f32, 16.0f32, 15.0f32, 16.0f32],
            ],
            [
                [1.0f32, 2.0f32, 1.0f32, 2.0f32],
                [3.0f32, 4.0f32, 3.0f32, 4.0f32],
                [1.0f32, 2.0f32, 1.0f32, 2.0f32],
                [3.0f32, 4.0f32, 3.0f32, 4.0f32],
                [1.0f32, 2.0f32, 1.0f32, 2.0f32],
                [3.0f32, 4.0f32, 3.0f32, 4.0f32],
            ],
            [
                [5.0f32, 6.0f32, 5.0f32, 6.0f32],
                [7.0f32, 8.0f32, 7.0f32, 8.0f32],
                [5.0f32, 6.0f32, 5.0f32, 6.0f32],
                [7.0f32, 8.0f32, 7.0f32, 8.0f32],
                [5.0f32, 6.0f32, 5.0f32, 6.0f32],
                [7.0f32, 8.0f32, 7.0f32, 8.0f32],
            ],
            [
                [9.0f32, 10.0f32, 9.0f32, 10.0f32],
                [11.0f32, 12.0f32, 11.0f32, 12.0f32],
                [9.0f32, 10.0f32, 9.0f32, 10.0f32],
                [11.0f32, 12.0f32, 11.0f32, 12.0f32],
                [9.0f32, 10.0f32, 9.0f32, 10.0f32],
                [11.0f32, 12.0f32, 11.0f32, 12.0f32],
            ],
            [
                [13.0f32, 14.0f32, 13.0f32, 14.0f32],
                [15.0f32, 16.0f32, 15.0f32, 16.0f32],
                [13.0f32, 14.0f32, 13.0f32, 14.0f32],
                [15.0f32, 16.0f32, 15.0f32, 16.0f32],
                [13.0f32, 14.0f32, 13.0f32, 14.0f32],
                [15.0f32, 16.0f32, 15.0f32, 16.0f32],
            ],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_int_repeat_on_many_dims() {
        let data = TensorData::from([
            [[1i32, 2i32], [3i32, 4i32]],
            [[5i32, 6i32], [7i32, 8i32]],
            [[9i32, 10i32], [11i32, 12i32]],
            [[13i32, 14i32], [15i32, 16i32]],
        ]);
        let tensor = TestTensorInt::<3>::from_data(data, &Default::default());

        let output = tensor.repeat(&[2, 3, 2]);

        let expected = TensorData::from([
            [
                [1i32, 2i32, 1i32, 2i32],
                [3i32, 4i32, 3i32, 4i32],
                [1i32, 2i32, 1i32, 2i32],
                [3i32, 4i32, 3i32, 4i32],
                [1i32, 2i32, 1i32, 2i32],
                [3i32, 4i32, 3i32, 4i32],
            ],
            [
                [5i32, 6i32, 5i32, 6i32],
                [7i32, 8i32, 7i32, 8i32],
                [5i32, 6i32, 5i32, 6i32],
                [7i32, 8i32, 7i32, 8i32],
                [5i32, 6i32, 5i32, 6i32],
                [7i32, 8i32, 7i32, 8i32],
            ],
            [
                [9i32, 10i32, 9i32, 10i32],
                [11i32, 12i32, 11i32, 12i32],
                [9i32, 10i32, 9i32, 10i32],
                [11i32, 12i32, 11i32, 12i32],
                [9i32, 10i32, 9i32, 10i32],
                [11i32, 12i32, 11i32, 12i32],
            ],
            [
                [13i32, 14i32, 13i32, 14i32],
                [15i32, 16i32, 15i32, 16i32],
                [13i32, 14i32, 13i32, 14i32],
                [15i32, 16i32, 15i32, 16i32],
                [13i32, 14i32, 13i32, 14i32],
                [15i32, 16i32, 15i32, 16i32],
            ],
            [
                [1i32, 2i32, 1i32, 2i32],
                [3i32, 4i32, 3i32, 4i32],
                [1i32, 2i32, 1i32, 2i32],
                [3i32, 4i32, 3i32, 4i32],
                [1i32, 2i32, 1i32, 2i32],
                [3i32, 4i32, 3i32, 4i32],
            ],
            [
                [5i32, 6i32, 5i32, 6i32],
                [7i32, 8i32, 7i32, 8i32],
                [5i32, 6i32, 5i32, 6i32],
                [7i32, 8i32, 7i32, 8i32],
                [5i32, 6i32, 5i32, 6i32],
                [7i32, 8i32, 7i32, 8i32],
            ],
            [
                [9i32, 10i32, 9i32, 10i32],
                [11i32, 12i32, 11i32, 12i32],
                [9i32, 10i32, 9i32, 10i32],
                [11i32, 12i32, 11i32, 12i32],
                [9i32, 10i32, 9i32, 10i32],
                [11i32, 12i32, 11i32, 12i32],
            ],
            [
                [13i32, 14i32, 13i32, 14i32],
                [15i32, 16i32, 15i32, 16i32],
                [13i32, 14i32, 13i32, 14i32],
                [15i32, 16i32, 15i32, 16i32],
                [13i32, 14i32, 13i32, 14i32],
                [15i32, 16i32, 15i32, 16i32],
            ],
        ]);

        output.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_support_bool_repeat_on_many_dimension() {
        let data = TensorData::from([
            [[false, true], [true, false]],
            [[true, true], [false, false]],
        ]);
        let tensor = TestTensorBool::<3>::from_data(data, &Default::default());

        let output = tensor.repeat(&[2, 3, 2]);
        let expected = TensorData::from([
            [
                [false, true, false, true],
                [true, false, true, false],
                [false, true, false, true],
                [true, false, true, false],
                [false, true, false, true],
                [true, false, true, false],
            ],
            [
                [true, true, true, true],
                [false, false, false, false],
                [true, true, true, true],
                [false, false, false, false],
                [true, true, true, true],
                [false, false, false, false],
            ],
            [
                [false, true, false, true],
                [true, false, true, false],
                [false, true, false, true],
                [true, false, true, false],
                [false, true, false, true],
                [true, false, true, false],
            ],
            [
                [true, true, true, true],
                [false, false, false, false],
                [true, true, true, true],
                [false, false, false, false],
                [true, true, true, true],
                [false, false, false, false],
            ],
        ]);

        output.into_data().assert_eq(&expected, false);
    }
}
