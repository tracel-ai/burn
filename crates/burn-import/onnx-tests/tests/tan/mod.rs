use crate::include_models;
include_models!(tan);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn tan() {
        // Initialize the model
        let device = Default::default();
        let model = tan::Model::<TestBackend>::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 4>::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let output = model.forward(input);
        // data from pyTorch
        let expected = TensorData::from([[[[1.5574f32, -2.1850, -0.1425, 1.1578]]]]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }
}
