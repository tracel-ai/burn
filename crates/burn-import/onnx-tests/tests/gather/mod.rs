// Import the shared macro
use crate::include_models;
include_models!(
    gather_1d_idx,
    gather_2d_idx,
    gather_elements,
    gather_scalar,
    gather_scalar_out,
    gather_shape
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn gather_1d_idx() {
        let model: gather_1d_idx::Model<Backend> = gather_1d_idx::Model::default();

        let device = Default::default();

        let input = Tensor::<Backend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let index = Tensor::<Backend, 1, Int>::from_ints([0, 2], &device);
        let expected = TensorData::from([[1f32, 3.], [4., 6.]]);
        let output = model.forward(input, index);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_2d_idx() {
        let model: gather_2d_idx::Model<Backend> = gather_2d_idx::Model::default();

        let device = Default::default();

        let input = Tensor::<Backend, 2>::from_data([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]], &device);
        let index = Tensor::<Backend, 2, Int>::from_data([[0, 1], [1, 2]], &device);
        let expected = TensorData::from([[[1f32, 1.2], [2.3, 3.4]], [[2.3, 3.4], [4.5, 5.7]]]);
        let output = model.forward(input, index);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_shape() {
        let model: gather_shape::Model<Backend> = gather_shape::Model::default();

        let device = Default::default();

        let input = Tensor::<Backend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        // shape(input) = [2, 3]
        let index = Tensor::<Backend, 1, Int>::from_ints([0], &device);
        let output = model.forward(input, index);
        let expected = [2usize];

        assert_eq!(output, expected);
    }

    #[test]
    fn gather_scalar() {
        let model: gather_scalar::Model<Backend> = gather_scalar::Model::default();

        let device = Default::default();

        let input = Tensor::<Backend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let index = 0;
        let output = model.forward(input, index);
        let expected = TensorData::from([1f32, 2., 3.]);

        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_scalar_out() {
        let model: gather_scalar_out::Model<Backend> = gather_scalar_out::Model::default();

        let device = Default::default();

        let input = Tensor::<Backend, 1>::from_floats([1., 2., 3.], &device);
        let index = 1;
        let output = model.forward(input, index);

        let expected = TensorData::from([2f32]);
        assert_eq!(output.to_data(), expected);
    }

    #[test]
    fn gather_elements() {
        // Initialize the model with weights (loaded from the exported file)
        let model: gather_elements::Model<Backend> = gather_elements::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 2>::from_floats([[1., 2.], [3., 4.]], &device);
        let index = Tensor::<Backend, 2, Int>::from_ints([[0, 0], [1, 0]], &device);
        let output = model.forward(input, index);
        let expected = TensorData::from([[1f32, 1.], [4., 3.]]);

        assert_eq!(output.to_data(), expected);
    }
}
