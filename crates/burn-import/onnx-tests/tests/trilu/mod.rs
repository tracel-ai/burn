use crate::include_models;
include_models!(trilu_lower, trilu_upper);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn trilu_upper() {
        let device = Default::default();
        let model: trilu_upper::Model<TestBackend> = trilu_upper::Model::new(&device);
        let input = Tensor::<TestBackend, 3>::from_floats(
            [[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]],
            &device,
        );
        let expected = TensorData::from([[
            [1.0_f32, 2.0_f32, 3.0_f32],
            [0.0_f32, 5.0_f32, 6.0_f32],
            [0.0_f32, 0.0_f32, 9.0_f32],
        ]]);

        let output = model.forward(input).to_data();

        output.assert_eq(&expected, true);
    }

    #[test]
    fn trilu_lower() {
        let device = Default::default();
        let model: trilu_lower::Model<TestBackend> = trilu_lower::Model::new(&device);
        let input = Tensor::<TestBackend, 3>::from_floats(
            [[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]],
            &device,
        );
        let expected = TensorData::from([[
            [1.0_f32, 0.0_f32, 0.0_f32],
            [4.0_f32, 5.0_f32, 0.0_f32],
            [7.0_f32, 8.0_f32, 9.0_f32],
        ]]);

        let output = model.forward(input).to_data();

        output.assert_eq(&expected, true);
    }
}
