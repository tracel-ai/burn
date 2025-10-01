// Import the shared macro
use crate::include_models;
include_models!(avg_pool1d, avg_pool2d);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData, Tolerance, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn avg_pool1d() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: avg_pool1d::Model<TestBackend> = avg_pool1d::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 3>::from_floats(
            [[
                [-1.526, -0.750, -0.654, -1.609, -0.100],
                [-0.609, -0.980, -1.609, -0.712, 1.171],
                [1.767, -0.095, 0.139, -1.579, -0.321],
                [-0.299, 1.879, 0.336, 0.275, 1.716],
                [-0.056, 0.911, -1.392, 2.689, -0.111],
            ]],
            &device,
        );
        let (output1, output2, output3) = model.forward(input.clone(), input.clone(), input);
        let expected1 = TensorData::from([[[-1.135f32], [-0.978], [0.058], [0.548], [0.538]]]);
        let expected2 = TensorData::from([[
            [-0.569f32, -1.135, -0.591],
            [-0.397, -0.978, -0.288],
            [0.418, 0.058, -0.440],
            [0.395, 0.548, 0.582],
            [0.214, 0.538, 0.296],
        ]]);
        let expected3 = TensorData::from([[
            [-1.138f32, -1.135, -0.788],
            [-0.794, -0.978, -0.383],
            [0.836, 0.058, -0.587],
            [0.790, 0.548, 0.776],
            [0.427, 0.538, 0.395],
        ]]);

        let expected_shape1 = Shape::from([1, 5, 1]);
        let expected_shape2 = Shape::from([1, 5, 3]);
        let expected_shape3 = Shape::from([1, 5, 3]);

        assert_eq!(output1.shape(), expected_shape1);
        assert_eq!(output2.shape(), expected_shape2);
        assert_eq!(output3.shape(), expected_shape3);

        let tolerance = Tolerance::default();
        output1
            .to_data()
            .assert_approx_eq::<FT>(&expected1, tolerance);
        output2
            .to_data()
            .assert_approx_eq::<FT>(&expected2, tolerance);
        output3
            .to_data()
            .assert_approx_eq::<FT>(&expected3, tolerance);
    }

    #[test]
    fn avg_pool2d() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: avg_pool2d::Model<TestBackend> = avg_pool2d::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats(
            [[[
                [-0.077, 0.360, -0.782, 0.072, 0.665],
                [-0.287, 1.621, -1.597, -0.052, 0.611],
                [0.760, -0.034, -0.345, 0.494, -0.078],
                [-1.805, -0.476, 0.205, 0.338, 1.353],
                [0.374, 0.013, 0.774, -0.109, -0.271],
            ]]],
            &device,
        );
        let (output1, output2, output3) = model.forward(input.clone(), input.clone(), input);
        let expected1 = TensorData::from([[[[0.008f32, -0.131, -0.208, 0.425]]]]);
        let expected2 = TensorData::from([[[
            [-0.045f32, 0.202, -0.050, -0.295, 0.162, 0.160],
            [-0.176, 0.008, -0.131, -0.208, 0.425, 0.319],
            [-0.084, -0.146, 0.017, 0.170, 0.216, 0.125],
        ]]]);
        let expected3 = TensorData::from([[[
            [-0.182f32, 0.404, -0.100, -0.590, 0.324, 0.638],
            [-0.352, 0.008, -0.131, -0.208, 0.425, 0.638],
            [-0.224, -0.195, 0.023, 0.226, 0.288, 0.335],
        ]]]);

        let expected_shape1 = Shape::from([1, 1, 1, 4]);
        let expected_shape2 = Shape::from([1, 1, 3, 6]);
        let expected_shape3 = Shape::from([1, 1, 3, 6]);

        assert_eq!(output1.shape(), expected_shape1);
        assert_eq!(output2.shape(), expected_shape2);
        assert_eq!(output3.shape(), expected_shape3);

        let tolerance = Tolerance::rel_abs(0.01, 0.001);
        output1
            .to_data()
            .assert_approx_eq::<FT>(&expected1, tolerance);
        output2
            .to_data()
            .assert_approx_eq::<FT>(&expected2, tolerance);
        output3
            .to_data()
            .assert_approx_eq::<FT>(&expected3, tolerance);
    }
}
