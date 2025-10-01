use crate::include_models;
include_models!(random_uniform_like);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn random_uniform_like() {
        let device = Default::default();
        let model = random_uniform_like::Model::<TestBackend>::new(&device);
        let input = TensorData::zeros::<f64, _>(Shape::from([2, 4, 4]));
        let expected_shape = Shape::from([2, 4, 4]);

        let output = model.forward(input.into());

        assert_eq!(expected_shape, output.shape());
    }
}
