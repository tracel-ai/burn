// Import the shared macro
use crate::include_models;
include_models!(
    maxpool1d,
    maxpool1d_ceil_mode,
    maxpool2d,
    maxpool2d_ceil_mode
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn maxpool1d() {
        let device = Default::default();

        let model: maxpool1d::Model<TestBackend> = maxpool1d::Model::new(&device);
        let input = Tensor::<TestBackend, 3>::from_floats(
            [[
                [1.927, 1.487, 0.901, -2.106, 0.678],
                [-1.235, -0.043, -1.605, -0.752, -0.687],
                [-0.493, 0.241, -1.111, 0.092, -2.317],
                [-0.217, -1.385, -0.396, 0.803, -0.622],
                [-0.592, -0.063, -0.829, 0.331, -1.558],
            ]],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([[
            [1.927f32, 1.927, 0.901],
            [-0.043, -0.043, -0.687],
            [0.241, 0.241, 0.092],
            [-0.217, 0.803, 0.803],
            [-0.063, 0.331, 0.331],
        ]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn maxpool2d() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: maxpool2d::Model<TestBackend> = maxpool2d::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.927, 1.487, 0.901, -2.106, 0.678],
                [-1.235, -0.043, -1.605, -0.752, -0.687],
                [-0.493, 0.241, -1.111, 0.092, -2.317],
                [-0.217, -1.385, -0.396, 0.803, -0.622],
                [-0.592, -0.063, -0.829, 0.331, -1.558],
            ]]],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([[[
            [0.901f32, 1.927, 1.487, 0.901],
            [0.901, 1.927, 1.487, 0.901],
            [-0.396, 0.803, 0.241, -0.396],
        ]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn maxpool1d_ceil_mode() {
        // Test ceil_mode=True for MaxPool1d
        // Input: 1x1x6 (values 1-6), kernel: 3, stride: 2, padding: 0
        // With ceil_mode=True: output = ceil((6-3)/2)+1 = 3 elements
        let device = Default::default();
        let model: maxpool1d_ceil_mode::Model<TestBackend> =
            maxpool1d_ceil_mode::Model::new(&device);

        let input =
            Tensor::<TestBackend, 3>::from_floats([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], &device);
        let output = model.forward(input);

        // Window 0: max(1,2,3) = 3
        // Window 1: max(3,4,5) = 5
        // Window 2: max(5,6) = 6 (partial window at edge)
        let expected = TensorData::from([[[3.0f32, 5.0, 6.0]]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn maxpool2d_ceil_mode() {
        // Test ceil_mode=True for MaxPool2d
        // Input: 1x1x6x6 (values 1-36), kernel: 3x3, stride: 2x2, padding: 0
        // With ceil_mode=True: output = 3x3
        let device = Default::default();
        let model: maxpool2d_ceil_mode::Model<TestBackend> =
            maxpool2d_ceil_mode::Model::new(&device);

        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                [25.0, 26.0, 27.0, 28.0, 29.0, 30.0],
                [31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
            ]]],
            &device,
        );
        let output = model.forward(input);

        // With ceil_mode=True, we get 3x3 output instead of 2x2
        // (0,0): max of rows 0-2, cols 0-2 = 15
        // (0,1): max of rows 0-2, cols 2-4 = 17
        // (0,2): max of rows 0-2, cols 4-5 = 18
        // (1,0): max of rows 2-4, cols 0-2 = 27
        // (1,1): max of rows 2-4, cols 2-4 = 29
        // (1,2): max of rows 2-4, cols 4-5 = 30
        // (2,0): max of rows 4-5, cols 0-2 = 33
        // (2,1): max of rows 4-5, cols 2-4 = 35
        // (2,2): max of rows 4-5, cols 4-5 = 36
        let expected = TensorData::from([[[
            [15.0f32, 17.0, 18.0],
            [27.0, 29.0, 30.0],
            [33.0, 35.0, 36.0],
        ]]]);
        output.to_data().assert_eq(&expected, true);
    }
}
