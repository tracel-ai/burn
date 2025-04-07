#[burn_tensor_testgen::testgen(movedim)]
mod tests {
    use super::*;
    use burn_tensor::{Device, Int, Shape, Tensor, TensorData};

    #[test]
    fn movedim_int() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

        let permuted = tensor.clone().movedim(0, 2);
        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim(0, 2)
        let expected = TensorData::from([
            [[0, 12], [1, 13], [2, 14], [3, 15]],
            [[4, 16], [5, 17], [6, 18], [7, 19]],
            [[8, 20], [9, 21], [10, 22], [11, 23]],
        ]);

        permuted.into_data().assert_eq(&expected, false);

        // Test with negative axis
        let permuted = tensor.clone().movedim(0, -1);
        permuted.into_data().assert_eq(&expected, false);

        // Test with the same axis
        let permuted = tensor.clone().movedim(0, 0);
        permuted.into_data().assert_eq(&tensor.into_data(), true);
    }

    #[test]
    fn movedim_float() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .float();

        let permuted = tensor.clone().movedim(0, 2);
        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim(0, 2).float()
        let expected = TensorData::from([
            [[0., 12.], [1., 13.], [2., 14.], [3., 15.]],
            [[4., 16.], [5., 17.], [6., 18.], [7., 19.]],
            [[8., 20.], [9., 21.], [10., 22.], [11., 23.]],
        ]);

        permuted.into_data().assert_eq(&expected, false);

        // Test with negative axis
        let permuted = tensor.clone().movedim(0, -1);
        permuted.into_data().assert_eq(&expected, false);

        // Test with the same axis
        let permuted = tensor.clone().movedim(0, 0);
        permuted.into_data().assert_eq(&tensor.into_data(), true);
    }

    #[test]
    fn movedim_bool() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .greater_elem(10);

        let permuted = tensor.clone().movedim(0, 2);
        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim(0, 2).gt(10)
        let expected = TensorData::from([
            [[false, true], [false, true], [false, true], [false, true]],
            [[false, true], [false, true], [false, true], [false, true]],
            [[false, true], [false, true], [false, true], [true, true]],
        ]);

        permuted.into_data().assert_eq(&expected, false);

        // Test with negative axis
        let permuted = tensor.clone().movedim(0, -1);
        permuted.into_data().assert_eq(&expected, false);

        // Test with the same axis
        let permuted = tensor.clone().movedim(0, 0);
        permuted.into_data().assert_eq(&tensor.into_data(), false);
    }

    #[test]
    fn vec_input_int() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

        let permuted = tensor.clone().movedim(vec![0, 1], vec![1, 0]);
        // from pytorch
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim([0, 1], [1, 0])
        let expected = TensorData::from([
            [[0, 1, 2, 3], [12, 13, 14, 15]],
            [[4, 5, 6, 7], [16, 17, 18, 19]],
            [[8, 9, 10, 11], [20, 21, 22, 23]],
        ]);

        permuted.into_data().assert_eq(&expected, false);

        // Test with negative axes
        let permuted = tensor.clone().movedim(vec![-3, -2], vec![-2, -3]);
        permuted.into_data().assert_eq(&expected, false);

        // Test with the same axes
        let permuted = tensor.clone().movedim(vec![0, 1], vec![0, 1]);
        permuted.into_data().assert_eq(&tensor.into_data(), true);
    }

    #[test]
    fn vec_input_float() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .float();

        let permuted = tensor.clone().movedim(vec![0, 1], vec![1, 0]);
        // from pytorch
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim([0, 1], [1, 0]).float()
        let expected = TensorData::from([
            [[0., 1., 2., 3.], [12., 13., 14., 15.]],
            [[4., 5., 6., 7.], [16., 17., 18., 19.]],
            [[8., 9., 10., 11.], [20., 21., 22., 23.]],
        ]);

        permuted.into_data().assert_eq(&expected, false);

        // Test with negative axes
        let permuted = tensor.clone().movedim(vec![-3, -2], vec![-2, -3]);
        permuted.into_data().assert_eq(&expected, false);

        // Test with the same axes
        let permuted = tensor.clone().movedim(vec![0, 1], vec![0, 1]);
        permuted.into_data().assert_eq(&tensor.into_data(), true);
    }

    #[test]
    fn vec_input_bool() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .greater_elem(10);

        let permuted = tensor.clone().movedim(vec![0, 1], vec![1, 0]);
        // from pytorch
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim([0, 1], [1, 0]).gt(10)
        let expected = TensorData::from([
            [[false, false, false, false], [true, true, true, true]],
            [[false, false, false, false], [true, true, true, true]],
            [[false, false, false, true], [true, true, true, true]],
        ]);

        permuted.into_data().assert_eq(&expected, false);

        // Test with negative axes
        let permuted = tensor.clone().movedim(vec![-3, -2], vec![-2, -3]);
        permuted.into_data().assert_eq(&expected, false);

        // Test with the same axes
        let permuted = tensor.clone().movedim(vec![0, 1], vec![0, 1]);
        permuted.into_data().assert_eq(&tensor.into_data(), false);
    }

    #[test]
    fn different_input_types() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device)
            .reshape([2, 3, 4])
            .float();

        let permuted = tensor.clone().movedim(0_usize, 2_i32);
        // from pytorch:
        // import torch; torch.arange(0, 24).reshape(2, 3, 4).movedim(0, 2).float()
        let expected = TensorData::from([
            [[0., 12.], [1., 13.], [2., 14.], [3., 15.]],
            [[4., 16.], [5., 17.], [6., 18.], [7., 19.]],
            [[8., 20.], [9., 21.], [10., 22.], [11., 23.]],
        ]);

        permuted.into_data().assert_eq(&expected, false);

        // Test with negative axis
        let permuted = tensor.clone().movedim(0_usize, -1);
        permuted.into_data().assert_eq(&expected, false);

        // Test with the same axis
        let permuted = tensor.clone().movedim(0_i32, 0_usize);
        permuted.into_data().assert_eq(&tensor.into_data(), true);
    }

    #[test]
    #[should_panic]
    fn edge_different_sizes() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with a repeated axis
        let _ = tensor.clone().movedim(vec![0, 1], vec![0]);
    }

    #[test]
    #[should_panic]
    fn edge_out_of_bound_axis() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with an out of bound axis
        let _ = tensor.clone().movedim(0, 100);
    }

    #[test]
    #[should_panic]
    fn edge_vec_is_not_a_set() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with a repeated axis
        let _ = tensor.clone().movedim(vec![0, 1, 1, 1, 1], vec![0, 0, 1]);
    }

    #[test]
    #[should_panic]
    fn edge_out_of_bound_axis_vec() {
        let device = Default::default();
        let tensor = TestTensorInt::<1>::arange(0..24, &device).reshape([2, 3, 4]);

        // Test with an out of bound axis
        let _ = tensor.clone().movedim(vec![0, 100], vec![0, 1]);
    }
}
