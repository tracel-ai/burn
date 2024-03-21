#[burn_tensor_testgen::testgen(flip)]
mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Data, Device, Int, Shape, Tensor};

    #[test]
    fn normal_int() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        let flipped = tensor.clone().flip([0, 2]);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).flip((0, 2))
        let data_expected = Data::from([
            [[15, 14, 13, 12], [19, 18, 17, 16], [23, 22, 21, 20]],
            [[3, 2, 1, 0], [7, 6, 5, 4], [11, 10, 9, 8]],
        ]);

        assert_eq!(data_expected, flipped.into_data());

        // Test with no flip
        let flipped = tensor.clone().flip([]);
        assert_eq!(tensor.into_data(), flipped.into_data());
    }

    #[test]
    fn normal_float() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .float();

        let flipped = tensor.clone().flip([0, 2]);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).flip((0, 2)).float()
        let data_expected = Data::from([
            [
                [15., 14., 13., 12.],
                [19., 18., 17., 16.],
                [23., 22., 21., 20.],
            ],
            [[3., 2., 1., 0.], [7., 6., 5., 4.], [11., 10., 9., 8.]],
        ]);

        assert_eq!(data_expected, flipped.into_data());

        // Test with no flip
        let flipped = tensor.clone().flip([]);
        assert_eq!(tensor.into_data(), flipped.into_data());
    }

    #[test]
    fn normal_bool() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .greater_elem(10);

        let flipped = tensor.clone().flip([0, 2]);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).flip((0, 2)).gt(10)
        let data_expected = Data::from([
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

        assert_eq!(data_expected, flipped.into_data());

        // Test with no flip
        let flipped = tensor.clone().flip([]);
        assert_eq!(tensor.into_data(), flipped.into_data());
    }

    #[test]
    #[should_panic]
    fn edge_duplicated_axes() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with a duplicated axis
        let _ = tensor.clone().flip([0, 0, 1]);
    }

    #[test]
    #[should_panic]
    fn edge_out_of_bound_axis() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with an out of bound axis
        let _ = tensor.clone().flip([3, 0, 1]);
    }
}
