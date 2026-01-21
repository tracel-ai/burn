use crate::include_models;
include_models!(hard_swish);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData, ops::FloatElem};

    use crate::backend::TestBackend;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn hard_swish() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: hard_swish::Model<TestBackend> = hard_swish::Model::new(&device);

        let input = Tensor::<TestBackend, 2>::from_floats(
            [
                [0.33669037, 0.12880941, 0.23446237],
                [0.23033303, -1.12285638, -0.18632829],
            ],
            &device,
        );
        let output = model.forward(input);

        let expected = TensorData::from([
            [0.18723859, 0.06717002, 0.12639327],
            [0.12400874, -0.35129377, -0.08737776],
        ]);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, burn::tensor::Tolerance::default());
    }
}
