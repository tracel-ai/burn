use super::*;
use burn_tensor::TensorData;

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
