// Import the shared macro
use crate::include_models;
include_models!(argmin);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn argmin() {
        // Initialize the model with weights (loaded from the exported file)
        let model: argmin::Model<Backend> = argmin::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 2>::from_floats(
            [[1.6124, 1.0463, -1.3808], [-0.3852, 0.1301, 0.9780]],
            &device,
        );
        let output = model.forward(input);
        let expected = TensorData::from([[2i64], [0]]);

        output.to_data().assert_eq(&expected, true);
    }
}
