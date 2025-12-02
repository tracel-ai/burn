#[burn_tensor_testgen::testgen(module_max_pool1d)]
mod tests {
    use super::*;
    use burn_tensor::module::{max_pool1d, max_pool1d_with_indices};
    use burn_tensor::{Tensor, TensorData, backend::Backend};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_max_pool1d_simple() {
        let kernel_size = 3;
        let padding = 0;
        let stride = 1;
        let dilation = 1;

        let x = TestTensor::from([[
            [0.9861, 0.5474, 0.4477, 0.0732, 0.3548, 0.8221],
            [0.8148, 0.5474, 0.9490, 0.7890, 0.5537, 0.5689],
        ]]);
        let y = TestTensor::<3>::from([[
            [0.9861, 0.5474, 0.4477, 0.8221],
            [0.949, 0.949, 0.949, 0.789],
        ]]);

        let output = max_pool1d(x, kernel_size, stride, padding, dilation, false);

        y.to_data()
            .assert_approx_eq::<FT>(&output.into_data(), Tolerance::default());
    }

    #[test]
    fn test_max_pool1d_different_padding_stride_kernel() {
        let kernel_size = 3;
        let padding = 1;
        let stride = 2;
        let dilation = 1;

        let x = TestTensor::from([[[0.6309, 0.6112, 0.6998, 0.4708]]]);
        let y = TestTensor::<3>::from([[[0.6309, 0.6998]]]);

        let output = max_pool1d(x, kernel_size, stride, padding, dilation, false);

        y.to_data()
            .assert_approx_eq::<FT>(&output.into_data(), Tolerance::default());
    }

    #[test]
    fn test_max_pool1d_with_neg() {
        let kernel_size = 3;
        let padding = 1;
        let stride = 1;
        let dilation = 1;

        let x = TestTensor::from([[[-0.6309, -0.6112, -0.6998, -0.4708]]]);
        let y = TestTensor::<3>::from([[[-0.6112, -0.6112, -0.4708, -0.4708]]]);

        let output = max_pool1d(x, kernel_size, stride, padding, dilation, false);

        y.to_data()
            .assert_approx_eq::<FT>(&output.into_data(), Tolerance::default());
    }

    #[test]
    fn test_max_pool1d_with_dilation() {
        let kernel_size = 2;
        let padding = 1;
        let stride = 1;
        let dilation = 2;

        let x = TestTensor::from([[
            [0.9861, 0.5474, 0.4477, 0.0732, 0.3548, 0.8221],
            [0.8148, 0.5474, 0.9490, 0.7890, 0.5537, 0.5689],
        ]]);
        let y = TestTensor::<3>::from([[
            [0.5474, 0.9861, 0.5474, 0.4477, 0.8221, 0.3548],
            [0.5474, 0.9490, 0.7890, 0.9490, 0.7890, 0.5537],
        ]]);

        let output = max_pool1d(x, kernel_size, stride, padding, dilation, false);

        y.to_data()
            .assert_approx_eq::<FT>(&output.into_data(), Tolerance::default());
    }

    #[test]
    fn test_max_pool1d_with_indices() {
        let kernel_size = 2;
        let padding = 0;
        let stride = 1;
        let dilation = 1;

        let x = TestTensor::from([[[0.2479, 0.6386, 0.3166, 0.5742]]]);
        let indices = TensorData::from([[[1, 1, 3]]]);
        let y = TestTensor::<3>::from([[[0.6386, 0.6386, 0.5742]]]);

        let (output, output_indices) =
            max_pool1d_with_indices(x, kernel_size, stride, padding, dilation, false);

        y.to_data()
            .assert_approx_eq::<FT>(&output.into_data(), Tolerance::default());
        output_indices.into_data().assert_eq(&indices, false);
    }

    #[test]
    fn test_max_pool1d_complex() {
        let kernel_size = 4;
        let padding = 2;
        let stride = 1;
        let dilation = 1;

        let x = TestTensor::from([[[0.5388, 0.0676, 0.7122, 0.8316, 0.0653]]]);
        let indices = TensorData::from([[[0, 2, 3, 3, 3, 3]]]);
        let y = TestTensor::<3>::from([[[0.5388, 0.7122, 0.8316, 0.8316, 0.8316, 0.8316]]]);

        let (output, output_indices) =
            max_pool1d_with_indices(x, kernel_size, stride, padding, dilation, false);

        y.to_data()
            .assert_approx_eq::<FT>(&output.into_data(), Tolerance::default());
        output_indices.into_data().assert_eq(&indices, false);
    }

    #[test]
    fn test_max_pool1d_ceil_mode() {
        // Test ceil_mode=true produces larger output when input doesn't divide evenly by stride
        // Input: 1x1x6, kernel: 3, stride: 2, padding: 0
        // Floor mode: output = (6-3)/2+1 = 2 elements
        // Ceil mode: output = ceil((6-3)/2)+1 = ceil(1.5)+1 = 3 elements
        let kernel_size = 3;
        let padding = 0;
        let stride = 2;
        let dilation = 1;

        let x = TestTensor::from([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]);

        // With ceil_mode=false (floor): output is 2 elements
        // Window 0: positions [0:3] -> max(1,2,3) = 3
        // Window 1: positions [2:5] -> max(3,4,5) = 5
        let y_floor = TestTensor::<3>::from([[[3.0, 5.0]]]);

        let output_floor = max_pool1d(x.clone(), kernel_size, stride, padding, dilation, false);

        y_floor
            .to_data()
            .assert_approx_eq::<FT>(&output_floor.into_data(), Tolerance::default());

        // With ceil_mode=true: output is 3 elements
        // Window 0: positions [0:3] -> max(1,2,3) = 3
        // Window 1: positions [2:5] -> max(3,4,5) = 5
        // Window 2: positions [4:7] -> max(5,6) = 6 (partial window)
        let y_ceil = TestTensor::<3>::from([[[3.0, 5.0, 6.0]]]);

        let output_ceil = max_pool1d(x, kernel_size, stride, padding, dilation, true);

        y_ceil
            .to_data()
            .assert_approx_eq::<FT>(&output_ceil.into_data(), Tolerance::default());
    }
}
