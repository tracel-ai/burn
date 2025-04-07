#[burn_tensor_testgen::testgen(permute)]
mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Device, Int, Shape, Tensor, TensorData};

    #[test]
    fn permute_int() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

        let permuted = tensor.clone().permute([2, 1, 0]);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).permute(2, 1, 0)
        let expected = TensorData::from([
            [[0, 12], [4, 16], [8, 20]],
            [[1, 13], [5, 17], [9, 21]],
            [[2, 14], [6, 18], [10, 22]],
            [[3, 15], [7, 19], [11, 23]],
        ]);

        permuted.into_data().assert_eq(&expected, false);

        // Test with negative axis
        let permuted = tensor.clone().permute([-1, 1, 0]);
        permuted.into_data().assert_eq(&expected, false);

        // Test with the same axis
        let permuted = tensor.clone().permute([0, 1, 2]);
        permuted.into_data().assert_eq(&tensor.into_data(), true);
    }

    #[test]
    fn permute_float() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .float();

        let permuted = tensor.clone().permute([2, 1, 0]);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).permute(2, 1, 0).float()
        let expected = TensorData::from([
            [[0., 12.], [4., 16.], [8., 20.]],
            [[1., 13.], [5., 17.], [9., 21.]],
            [[2., 14.], [6., 18.], [10., 22.]],
            [[3., 15.], [7., 19.], [11., 23.]],
        ]);

        permuted.into_data().assert_eq(&expected, false);

        // Test with negative axis
        let permuted = tensor.clone().permute([-1, 1, 0]);
        permuted.into_data().assert_eq(&expected, false);

        // Test with the same axis
        let permuted = tensor.clone().permute([0, 1, 2]);
        permuted.into_data().assert_eq(&tensor.into_data(), true);
    }

    #[test]
    fn permute_bool() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .greater_elem(10);

        let permuted = tensor.clone().permute([2, 1, 0]);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).permute(2, 1, 0).gt(10)
        let expected = TensorData::from([
            [[false, true], [false, true], [false, true]],
            [[false, true], [false, true], [false, true]],
            [[false, true], [false, true], [false, true]],
            [[false, true], [false, true], [true, true]],
        ]);

        permuted.into_data().assert_eq(&expected, false);

        // Test with negative axis
        let permuted = tensor.clone().permute([-1, 1, 0]);
        permuted.into_data().assert_eq(&expected, false);

        // Test with the same axis
        let permuted = tensor.clone().permute([0, 1, 2]);
        permuted.into_data().assert_eq(&tensor.into_data(), false);
    }

    #[test]
    #[should_panic]
    fn edge_repeated_axes() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with a repeated axis
        let _ = tensor.clone().permute([0, 0, 1]);
    }

    #[test]
    #[should_panic]
    fn edge_out_of_bound_axis() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with a repeated axis
        let _ = tensor.clone().permute([3, 0, 1]);
    }
}
