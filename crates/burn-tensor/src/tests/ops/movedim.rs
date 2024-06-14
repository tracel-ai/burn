#[burn_tensor_testgen::testgen(movedim)]
mod tests {
    use super::*;
    use burn_tensor::backend::Backend;
    use burn_tensor::{Data, Device, Int, Shape, Tensor};

    #[test]
    fn normal_int() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        let permuted = tensor.clone().movedim(0, 2);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim(0, 2)
        let data_expected = Data::from([
            [[0, 12], [1, 13], [2, 14], [3, 15]],
            [[4, 16], [5, 17], [6, 18], [7, 19]],
            [[8, 20], [9, 21], [10, 22], [11, 23]],
        ]);

        assert_eq!(data_expected, permuted.into_data());

        // Test with negative axis
        let permuted = tensor.clone().movedim(0, -1);
        assert_eq!(data_expected, permuted.into_data());

        // Test with the same axis
        let permuted = tensor.clone().movedim(0, 0);
        assert_eq!(tensor.into_data(), permuted.into_data());
    }

    #[test]
    fn normal_float() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .float();

        let permuted = tensor.clone().movedim(0, 2);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim(0, 2).float()
        let data_expected = Data::from([
            [[0., 12.], [1., 13.], [2., 14.], [3., 15.]],
            [[4., 16.], [5., 17.], [6., 18.], [7., 19.]],
            [[8., 20.], [9., 21.], [10., 22.], [11., 23.]],
        ]);

        assert_eq!(data_expected, permuted.into_data());

        // Test with negative axis
        let permuted = tensor.clone().movedim(0, -1);
        assert_eq!(data_expected, permuted.into_data());

        // Test with the same axis
        let permuted = tensor.clone().movedim(0, 0);
        assert_eq!(tensor.into_data(), permuted.into_data());
    }

    #[test]
    fn normal_bool() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .greater_elem(10);

        let permuted = tensor.clone().movedim(0, 2);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim(0, 2).gt(10)
        let data_expected = Data::from([
            [[false, true], [false, true], [false, true], [false, true]],
            [[false, true], [false, true], [false, true], [false, true]],
            [[false, true], [false, true], [false, true], [true, true]],
        ]);

        assert_eq!(data_expected, permuted.into_data());

        // Test with negative axis
        let permuted = tensor.clone().movedim(0, -1);
        assert_eq!(data_expected, permuted.into_data());

        // Test with the same axis
        let permuted = tensor.clone().movedim(0, 0);
        assert_eq!(tensor.into_data(), permuted.into_data());
    }

    #[test]
    fn vec_input_int() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        let permuted = tensor.clone().movedim(vec![0, 1], vec![1, 0]);

        // from pytorch
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim([0, 1], [1, 0])
        let data_expected = Data::from([
            [[0, 1, 2, 3], [12, 13, 14, 15]],
            [[4, 5, 6, 7], [16, 17, 18, 19]],
            [[8, 9, 10, 11], [20, 21, 22, 23]],
        ]);

        assert_eq!(data_expected, permuted.into_data());

        // Test with negative axes
        let permuted = tensor.clone().movedim(vec![-3, -2], vec![-2, -3]);
        assert_eq!(data_expected, permuted.into_data());

        // Test with the same axes
        let permuted = tensor.clone().movedim(vec![0, 1], vec![0, 1]);
        assert_eq!(tensor.into_data(), permuted.into_data());
    }

    #[test]
    fn vec_input_float() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .float();

        let permuted = tensor.clone().movedim(vec![0, 1], vec![1, 0]);

        // from pytorch
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim([0, 1], [1, 0]).float()
        let data_expected = Data::from([
            [[0., 1., 2., 3.], [12., 13., 14., 15.]],
            [[4., 5., 6., 7.], [16., 17., 18., 19.]],
            [[8., 9., 10., 11.], [20., 21., 22., 23.]],
        ]);

        assert_eq!(data_expected, permuted.into_data());

        // Test with negative axes
        let permuted = tensor.clone().movedim(vec![-3, -2], vec![-2, -3]);
        assert_eq!(data_expected, permuted.into_data());

        // Test with the same axes
        let permuted = tensor.clone().movedim(vec![0, 1], vec![0, 1]);
        assert_eq!(tensor.into_data(), permuted.into_data());
    }

    #[test]
    fn vec_input_bool() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .greater_elem(10);

        let permuted = tensor.clone().movedim(vec![0, 1], vec![1, 0]);

        // from pytorch
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim([0, 1], [1, 0]).gt(10)
        let data_expected = Data::from([
            [[false, false, false, false], [true, true, true, true]],
            [[false, false, false, false], [true, true, true, true]],
            [[false, false, false, true], [true, true, true, true]],
        ]);

        assert_eq!(data_expected, permuted.into_data());

        // Test with negative axes
        let permuted = tensor.clone().movedim(vec![-3, -2], vec![-2, -3]);
        assert_eq!(data_expected, permuted.into_data());

        // Test with the same axes
        let permuted = tensor.clone().movedim(vec![0, 1], vec![0, 1]);
        assert_eq!(tensor.into_data(), permuted.into_data());
    }

    #[test]
    fn different_input_types() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .float();

        let permuted = tensor.clone().movedim(0_usize, 2_i32);

        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim(0, 2).float()
        let data_expected = Data::from([
            [[0., 12.], [1., 13.], [2., 14.], [3., 15.]],
            [[4., 16.], [5., 17.], [6., 18.], [7., 19.]],
            [[8., 20.], [9., 21.], [10., 22.], [11., 23.]],
        ]);

        assert_eq!(data_expected, permuted.into_data());

        // Test with negative axis
        let permuted = tensor.clone().movedim(0_usize, -1);
        assert_eq!(data_expected, permuted.into_data());

        // Test with the same axis
        let permuted = tensor.clone().movedim(0_i32, 0_usize);
        assert_eq!(tensor.into_data(), permuted.into_data());
    }

    #[test]
    #[should_panic]
    fn edge_different_sizes() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with a repeated axis
        let _ = tensor.clone().movedim(vec![0, 1], vec![0]);
    }

    #[test]
    #[should_panic]
    fn edge_out_of_bound_axis() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with an out of bound axis
        let _ = tensor.clone().movedim(0, 100);
    }

    #[test]
    #[should_panic]
    fn edge_vec_is_not_a_set() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with a repeated axis
        let _ = tensor.clone().movedim(vec![0, 1, 1, 1, 1], vec![0, 0, 1]);
    }

    #[test]
    #[should_panic]
    fn edge_out_of_bound_axis_vec() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with an out of bound axis
        let _ = tensor.clone().movedim(vec![0, 100], vec![0, 1]);
    }
}
