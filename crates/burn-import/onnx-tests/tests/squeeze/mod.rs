use crate::include_models;
include_models!(squeeze, squeeze_multiple);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};

    use crate::backend::Backend;

    #[test]
    fn squeeze() {
        let device = Default::default();
        let model = squeeze::Model::<Backend>::new(&device);
        let input_shape = Shape::from([3, 4, 1, 5]);
        let expected_shape = Shape::from([3, 4, 5]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input);
        assert_eq!(expected_shape, output.shape());
    }

    #[test]
    fn squeeze_multiple() {
        let device = Default::default();
        let model = squeeze_multiple::Model::<Backend>::new(&device);
        let input_shape = Shape::from([3, 4, 1, 5, 1]);
        let expected_shape = Shape::from([3, 4, 5]);
        let input = Tensor::ones(input_shape, &device);
        let output = model.forward(input);
        assert_eq!(expected_shape, output.shape());
    }
}
