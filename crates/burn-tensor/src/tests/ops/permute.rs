#[burn_tensor_testgen::testgen(permute)]
mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Data, Device, Int, Shape, Tensor};

    #[test]
    fn normal_int() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        let permuted = tensor.clone().permute([2, 1, 0]);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).permute(2, 1, 0)
        let data_expected = Data::from([
            [[0, 12], [4, 16], [8, 20]],
            [[1, 13], [5, 17], [9, 21]],
            [[2, 14], [6, 18], [10, 22]],
            [[3, 15], [7, 19], [11, 23]],
        ]);

        assert_eq!(data_expected, permuted.into_data());

        // Test with negative axis
        let permuted = tensor.clone().permute([-1, 1, 0]);
        assert_eq!(data_expected, permuted.into_data());

        // Test with the same axis
        let permuted = tensor.clone().permute([0, 1, 2]);
        assert_eq!(tensor.into_data(), permuted.into_data());
    }

    #[test]
    fn normal_float() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .float();

        let permuted = tensor.clone().permute([2, 1, 0]);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).permute(2, 1, 0).float()
        let data_expected = Data::from([
            [[0., 12.], [4., 16.], [8., 20.]],
            [[1., 13.], [5., 17.], [9., 21.]],
            [[2., 14.], [6., 18.], [10., 22.]],
            [[3., 15.], [7., 19.], [11., 23.]],
        ]);

        assert_eq!(data_expected, permuted.into_data());

        // Test with negative axis
        let permuted = tensor.clone().permute([-1, 1, 0]);
        assert_eq!(data_expected, permuted.into_data());

        // Test with the same axis
        let permuted = tensor.clone().permute([0, 1, 2]);
        assert_eq!(tensor.into_data(), permuted.into_data());
    }

    #[test]
    fn normal_bool() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .greater_elem(10);

        let permuted = tensor.clone().permute([2, 1, 0]);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).permute(2, 1, 0).gt(10)
        let data_expected = Data::from([
            [[false, true], [false, true], [false, true]],
            [[false, true], [false, true], [false, true]],
            [[false, true], [false, true], [false, true]],
            [[false, true], [false, true], [true, true]],
        ]);

        assert_eq!(data_expected, permuted.into_data());

        // Test with negative axis
        let permuted = tensor.clone().permute([-1, 1, 0]);
        assert_eq!(data_expected, permuted.into_data());

        // Test with the same axis
        let permuted = tensor.clone().permute([0, 1, 2]);
        assert_eq!(tensor.into_data(), permuted.into_data());
    }

    #[test]
    #[should_panic]
    fn edge_repeated_axes() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with a repeated axis
        let _ = tensor.clone().permute([0, 0, 1]);
    }

    #[test]
    #[should_panic]
    fn edge_out_of_bound_axis() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with a repeated axis
        let _ = tensor.clone().permute([3, 0, 1]);
    }
}
