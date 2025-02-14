#[burn_tensor_testgen::testgen(flip)]
mod tests {
    use super::*;
    use burn_tensor::{Device, Int, Shape, Tensor, TensorData};

    #[test]
    fn flip_int() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

        let flipped = tensor.clone().flip([0, 2]);
        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).flip((0, 2))
        let expected = TensorData::from([
            [[15, 14, 13, 12], [19, 18, 17, 16], [23, 22, 21, 20]],
            [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]],
        ]);

        flipped.into_data().assert_eq(&expected, false);

        // Test with no flip
        let flipped = tensor.clone().flip([]);
        assert_eq!(tensor.into_data(), flipped.into_data());
    }

    #[test]
    fn flip_float() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .float();

        let flipped = tensor.clone().flip([0, 2]);
        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).flip((0, 2)).float()
        let expected = TensorData::from([
            [
                [15., 14., 13., 12.],
                [19., 18., 17., 16.],
                [23., 22., 21., 20.],
            ],
            [[3., 2., 1., 0.], [7., 6., 5., 4.], [11., 10., 9., 8.]],
        ]);

        flipped.into_data().assert_eq(&expected, false);

        // Test with no flip
        let flipped = tensor.clone().flip([]);
        tensor.into_data().assert_eq(&flipped.into_data(), false);
    }

    #[test]
    fn flip_bool() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .greater_elem(10);

        let flipped = tensor.clone().flip([0, 2]);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).flip((0, 2)).gt(10)
        let data_expected = TensorData::from([
            [
                [true, true, true, true],
                [true, true, true, true],
                [true, true, true, true],
            ],
            [
                [false, false, false, false],
                [false, false, false, false],
                [true, false, false, false],
            ],
        ]);

        flipped.into_data().assert_eq(&data_expected, false);

        // Test with no flip
        let flipped = tensor.clone().flip([]);
        tensor.into_data().assert_eq(&flipped.into_data(), false);
    }

    #[test]
    #[should_panic]
    fn flip_duplicated_axes() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with a duplicated axis
        let _ = tensor.clone().flip([0, 0, 1]);
    }

    #[test]
    #[should_panic]
    fn flip_out_of_bound_axis() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with an out of bound axis
        let _ = tensor.clone().flip([3, 0, 1]);
    }
}
