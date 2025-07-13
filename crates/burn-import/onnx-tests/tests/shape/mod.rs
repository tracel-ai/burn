use crate::include_models;
include_models!(shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    use crate::backend::Backend;

    #[test]
    fn shape() {
        let device = Default::default();
        let model: shape::Model<Backend> = shape::Model::new(&device);

        // Run the model
        let input = Tensor::<Backend, 2>::ones([4, 2], &device);
        let output = model.forward(input);
        let expected = [4, 2];
        assert_eq!(output, expected);
    }
}
