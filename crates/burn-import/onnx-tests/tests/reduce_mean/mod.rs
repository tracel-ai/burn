// Import the shared macro
use crate::include_models;
include_models!(
    reduce_mean,
    reduce_mean_no_keepdims,
    reduce_mean_all_dims,
    reduce_mean_multi_axes
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn reduce_mean() {
        let device = Default::default();
        let model: reduce_mean::Model<Backend> = reduce_mean::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 4>::from_floats([[[[1.0, 4.0, 9.0, 25.0]]]], &device);
        let (output_scalar, output_tensor, output_value) = model.forward(input.clone());
        let expected_scalar = TensorData::from([9.75f32]);
        let expected = TensorData::from([[[[9.75f32]]]]);

        output_scalar.to_data().assert_eq(&expected_scalar, true);
        output_tensor.to_data().assert_eq(&input.to_data(), true);
        output_value.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn reduce_mean_no_keepdims() {
        let device = Default::default();
        let model: reduce_mean_no_keepdims::Model<Backend> =
            reduce_mean_no_keepdims::Model::new(&device);

        // Test with shape [2, 3, 4] -> reduce on dim 1 -> [2, 4]
        let input = Tensor::<Backend, 3>::from_floats(
            [
                [
                    [-1.1258, -1.1524, -0.2506, -0.4339],
                    [0.8487, 0.6920, -0.3160, -2.1152],
                    [0.4681, -0.1577, 1.4437, 0.2660],
                ],
                [
                    [0.1665, 0.8744, -0.1435, -0.1116],
                    [0.9318, 1.2590, 2.0050, 0.0537],
                    [0.6181, -0.4128, -0.8411, -2.3160],
                ],
            ],
            &device,
        );

        let output = model.forward(input);

        // Expected output shape should be [2, 4] after reducing dimension 1
        let expected = TensorData::from([
            [0.0637, -0.2060, 0.2924, -0.7610],
            [0.5721, 0.5735, 0.3401, -0.7913],
        ]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn reduce_mean_all_dims() {
        let device = Default::default();
        let model: reduce_mean_all_dims::Model<Backend> = reduce_mean_all_dims::Model::new(&device);

        // Test with shape [2, 3, 4] -> reduce all dims -> scalar
        let input = Tensor::<Backend, 3>::from_floats(
            [
                [
                    [-1.1258, -1.1524, -0.2506, -0.4339],
                    [0.8487, 0.6920, -0.3160, -2.1152],
                    [0.4681, -0.1577, 1.4437, 0.2660],
                ],
                [
                    [0.1665, 0.8744, -0.1435, -0.1116],
                    [0.9318, 1.2590, 2.0050, 0.0537],
                    [0.6181, -0.4128, -0.8411, -2.3160],
                ],
            ],
            &device,
        );

        let output = model.forward(input);

        // Expected output is the mean of all 24 elements
        let expected = TensorData::from([0.010432422f32]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }

    #[test]
    fn reduce_mean_multi_axes() {
        let device = Default::default();
        let model: reduce_mean_multi_axes::Model<Backend> =
            reduce_mean_multi_axes::Model::new(&device);

        // Test with shape [2, 3, 4] -> reduce on dims [0, 2] -> [3]
        let input = Tensor::<Backend, 3>::from_floats(
            [
                [
                    [-1.1258, -1.1524, -0.2506, -0.4339],
                    [0.8487, 0.6920, -0.3160, -2.1152],
                    [0.4681, -0.1577, 1.4437, 0.2660],
                ],
                [
                    [0.1665, 0.8744, -0.1435, -0.1116],
                    [0.9318, 1.2590, 2.0050, 0.0537],
                    [0.6181, -0.4128, -0.8411, -2.3160],
                ],
            ],
            &device,
        );

        let output = model.forward(input);

        // Expected output shape should be [3] after reducing dimensions [0, 2]
        let expected = TensorData::from([-0.2721f32, 0.4199, -0.1165]);

        output
            .to_data()
            .assert_approx_eq::<f32>(&expected, burn::tensor::Tolerance::default());
    }
}
