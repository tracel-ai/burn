use crate::include_models;
include_models!(shape);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    use crate::backend::TestBackend;

    #[test]
    fn shape() {
        let device = Default::default();
        let model: shape::Model<TestBackend> = shape::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 2>::ones([4, 2], &device);
        let output = model.forward(input);
        let expected = [4i64, 2i64];
        assert_eq!(output, expected);
    }
}
