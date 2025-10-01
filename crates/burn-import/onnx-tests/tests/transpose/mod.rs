use crate::include_models;
include_models!(transpose);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn transpose() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: transpose::Model<TestBackend> = transpose::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 3>::from_floats(
            [
                [[0., 1., 2., 3.], [4., 5., 6., 7.], [8., 9., 10., 11.]],
                [
                    [12., 13., 14., 15.],
                    [16., 17., 18., 19.],
                    [20., 21., 22., 23.],
                ],
            ],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([
            [[0f32, 4., 8.], [12., 16., 20.]],
            [[1., 5., 9.], [13., 17., 21.]],
            [[2., 6., 10.], [14., 18., 22.]],
            [[3., 7., 11.], [15., 19., 23.]],
        ]);

        output.to_data().assert_eq(&expected, true);
    }
}
