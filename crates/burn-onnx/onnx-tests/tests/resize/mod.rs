use crate::include_models;
include_models!(
    resize_1d_linear_scale,
    resize_1d_nearest_scale,
    resize_2d_bicubic_scale,
    resize_2d_bilinear_scale,
    resize_2d_nearest_scale,
    resize_with_sizes,
    resize_with_shape,
    resize_with_sizes_tensor,
    resize_with_scales_tensor
);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use burn::tensor::{Tensor, TensorData, ops::FloatElem};
    use float_cmp::ApproxEq;

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn resize_with_sizes() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_with_sizes::Model<TestBackend> = resize_with_sizes::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
            ]]],
            &device,
        );

        // The sizes are [1, 1, 2, 3]
        let output = model.forward(input);
        let expected = TensorData::from([[[[0.0f32, 1.5, 3.0], [12.0, 13.5, 15.0]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn resize_with_shape() {
        // Initialize the model without weights
        let device = Default::default();
        let model: resize_with_shape::Model<TestBackend> = resize_with_shape::Model::new(&device);

        // Create input tensor [1, 3, 4, 4]
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0, 15.0],
                ],
                [
                    [16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0],
                    [24.0, 25.0, 26.0, 27.0],
                    [28.0, 29.0, 30.0, 31.0],
                ],
                [
                    [32.0, 33.0, 34.0, 35.0],
                    [36.0, 37.0, 38.0, 39.0],
                    [40.0, 41.0, 42.0, 43.0],
                    [44.0, 45.0, 46.0, 47.0],
                ],
            ]],
            &device,
        );

        // The model should resize from [1, 3, 4, 4] to [1, 3, 8, 8] using bilinear interpolation
        let output = model.forward(input);

        // Check output dimensions
        assert_eq!(output.dims(), [1, 3, 8, 8]);

        // Check that interpolation is working correctly by verifying corner values
        // The corners should match the original corners
        let output_data = output.to_data();
        let values: Vec<f32> = output_data.to_vec().unwrap();

        // Check first channel corners
        assert_eq!(values[0], 0.0); // Top-left of first channel
        assert_eq!(values[7], 3.0); // Top-right of first channel
        assert_eq!(values[56], 12.0); // Bottom-left of first channel
        assert_eq!(values[63], 15.0); // Bottom-right of first channel

        // Check that the output has the right number of elements
        assert_eq!(values.len(), 1 * 3 * 8 * 8);
    }

    #[test]
    fn resize_with_sizes_tensor() {
        // Initialize the model without weights
        let device = Default::default();
        let model: resize_with_sizes_tensor::Model<TestBackend> =
            resize_with_sizes_tensor::Model::new(&device);

        // Create input tensor [1, 3, 4, 4]
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                [
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0],
                    [29.0, 30.0, 31.0, 32.0],
                ],
                [
                    [33.0, 34.0, 35.0, 36.0],
                    [37.0, 38.0, 39.0, 40.0],
                    [41.0, 42.0, 43.0, 44.0],
                    [45.0, 46.0, 47.0, 48.0],
                ],
            ]],
            &device,
        );

        // Create sizes tensor [1, 3, 2, 2] - resize to 2x2
        let sizes =
            Tensor::<TestBackend, 1, burn::tensor::Int>::from_ints([1i64, 3, 2, 2], &device);

        // The model should resize from [1, 3, 4, 4] to [1, 3, 2, 2] using nearest neighbor
        let output = model.forward(input, sizes);

        // Check output dimensions
        assert_eq!(output.dims(), [1, 3, 2, 2]);

        // With nearest neighbor and downsampling by 2, we should get corners
        let output_data = output.to_data();
        let values: Vec<f32> = output_data.to_vec().unwrap();

        // For nearest neighbor with 2x downsampling, we expect to sample at positions (0,0), (0,2), (2,0), (2,2)
        // First channel should have values [1, 3, 9, 11]
        assert_eq!(values[0], 1.0); // [0, 0, 0, 0]
        assert_eq!(values[1], 3.0); // [0, 0, 0, 2]
        assert_eq!(values[2], 9.0); // [0, 0, 2, 0]
        assert_eq!(values[3], 11.0); // [0, 0, 2, 2]

        // Check that the output has the right number of elements
        assert_eq!(values.len(), 1 * 3 * 2 * 2);
    }

    #[test]
    fn resize_with_scales_1d_linear() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_1d_linear_scale::Model<TestBackend> =
            resize_1d_linear_scale::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 3>::from_floats(
            [[[1.5410, -0.2934, -2.1788, 0.5684, -1.0845, -1.3986]]],
            &device,
        );

        // The scales are 1.5
        let output = model.forward(input);

        Tensor::<TestBackend, 3>::from([[[
            1.5410, 0.3945, -0.7648, -1.9431, -0.8052, 0.3618, -0.6713, -1.2023, -1.3986,
        ]]])
        .to_data()
        .assert_approx_eq::<FT>(&output.into_data(), burn::tensor::Tolerance::default());
    }

    #[test]
    fn resize_with_scales_2d_bilinear() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_2d_bilinear_scale::Model<TestBackend> =
            resize_2d_bilinear_scale::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [-1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.6920],
                [-0.3160, -2.1152, 0.3223, -1.2633, 0.3500, 0.3081],
                [0.1198, 1.2377, 1.1168, -0.2473, -1.3527, -1.6959],
                [0.5667, 0.7935, 0.4397, 0.1124, 0.6408, 0.4412],
                [-0.2159, -0.7425, 0.5627, 0.2596, 0.5229, 2.3022],
                [-1.4689, -1.5867, 1.2032, 0.0845, -1.2001, -0.0048],
            ]]],
            &device,
        );

        // The scales are 1.5, 1.5
        let output = model.forward(input);

        let output_sum = output.sum().into_scalar();
        let expected_sum = -3.401_126_6; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn resize_with_scales_2d_nearest() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_2d_nearest_scale::Model<TestBackend> =
            resize_2d_nearest_scale::Model::<TestBackend>::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [-1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.6920],
                [-0.3160, -2.1152, 0.3223, -1.2633, 0.3500, 0.3081],
                [0.1198, 1.2377, 1.1168, -0.2473, -1.3527, -1.6959],
                [0.5667, 0.7935, 0.4397, 0.1124, 0.6408, 0.4412],
                [-0.2159, -0.7425, 0.5627, 0.2596, 0.5229, 2.3022],
                [-1.4689, -1.5867, 1.2032, 0.0845, -1.2001, -0.0048],
            ]]],
            &device,
        );

        // The scales are 1.5, 1.5
        let output = model.forward(input);

        assert_eq!(output.dims(), [1, 1, 9, 9]);

        let output_sum = output.sum().into_scalar();
        let expected_sum = -0.812_227_7; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn resize_with_scales_1d_nearest() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_1d_nearest_scale::Model<TestBackend> =
            resize_1d_nearest_scale::Model::<TestBackend>::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 3>::from_floats(
            [[[1.5410, -0.2934, -2.1788, 0.5684, -1.0845, -1.3986]]],
            &device,
        );

        // The scales are 1.5, 1.5
        let output = model.forward(input);

        assert_eq!(output.dims(), [1, 1, 9]);

        let output_sum = output.sum().into_scalar();
        let expected_sum = -4.568_224; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }

    #[test]
    fn resize_with_scales_2d_bicubic() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: resize_2d_bicubic_scale::Model<TestBackend> =
            resize_2d_bicubic_scale::Model::<TestBackend>::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [-1.1258, -1.1524, -0.2506, -0.4339, 0.8487, 0.6920],
                [-0.3160, -2.1152, 0.3223, -1.2633, 0.3500, 0.3081],
                [0.1198, 1.2377, 1.1168, -0.2473, -1.3527, -1.6959],
                [0.5667, 0.7935, 0.4397, 0.1124, 0.6408, 0.4412],
                [-0.2159, -0.7425, 0.5627, 0.2596, 0.5229, 2.3022],
                [-1.4689, -1.5867, 1.2032, 0.0845, -1.2001, -0.0048],
            ]]],
            &device,
        );

        // The scales are 1.5, 1.5
        let output = model.forward(input);

        assert_eq!(output.dims(), [1, 1, 9, 9]);

        let output_sum = output.sum().into_scalar();

        let expected_sum = -3.515_921; // from pytorch

        assert!(expected_sum.approx_eq(output_sum, (1.0e-3, 2)));
    }

    #[test]
    fn resize_with_scales_tensor() {
        // Initialize the model without weights
        let device = Default::default();
        let model: resize_with_scales_tensor::Model<TestBackend> =
            resize_with_scales_tensor::Model::new(&device);

        // Create input tensor [1, 3, 4, 4]
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                [
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0],
                    [29.0, 30.0, 31.0, 32.0],
                ],
                [
                    [33.0, 34.0, 35.0, 36.0],
                    [37.0, 38.0, 39.0, 40.0],
                    [41.0, 42.0, 43.0, 44.0],
                    [45.0, 46.0, 47.0, 48.0],
                ],
            ]],
            &device,
        );

        // Create scales tensor to double spatial dimensions: [1, 3, 4, 4] -> [1, 3, 8, 8]
        // Format: [scale_n, scale_c, scale_h, scale_w]
        let scales = Tensor::<TestBackend, 1>::from_floats([1.0f32, 1.0, 2.0, 2.0], &device);

        // The model should resize from [1, 3, 4, 4] to [1, 3, 8, 8] using nearest neighbor
        let output = model.forward(input, scales);

        // Check output dimensions
        assert_eq!(output.dims(), [1, 3, 8, 8]);

        // With nearest neighbor and 2x upsampling, output sum should be 4x input sum
        // Input sum per channel: 136, 392, 648 => total = 1176
        // Output sum should be 4704 (each value repeated 4 times)
        let output_sum = output.sum().into_scalar();
        let expected_sum = 4704.0f32;

        assert!(expected_sum.approx_eq(output_sum, (1.0e-4, 2)));
    }
}
