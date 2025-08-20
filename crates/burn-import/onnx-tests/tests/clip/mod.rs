// Import the shared macro
use crate::include_models;
include_models!(clip);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn clip() {
        // Initialize the model without weights (because the exported file does not contain them)
        let device = Default::default();
        let model: clip::Model<TestBackend> = clip::Model::new(&device);

        // Run the model
        let input = Tensor::<TestBackend, 1>::from_floats(
            [
                0.88226926,
                0.91500396,
                0.38286376,
                0.95930564,
                0.390_448_2,
                0.60089535,
            ],
            &device,
        );
        let (output1, output2, output3) = model.forward(input);
        let expected1 = TensorData::from([
            0.88226926f32,
            0.91500396,
            0.38286376,
            0.95930564,
            0.390_448_2,
            0.60089535,
        ]);
        let expected2 = TensorData::from([0.7f32, 0.7, 0.5, 0.7, 0.5, 0.60089535]);
        let expected3 = TensorData::from([0.8f32, 0.8, 0.38286376, 0.8, 0.390_448_2, 0.60089535]);

        output1.to_data().assert_eq(&expected1, true);
        output2.to_data().assert_eq(&expected2, true);
        output3.to_data().assert_eq(&expected3, true);
    }
}
