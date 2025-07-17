// Import the shared macro
use crate::include_models;
include_models!(argmax);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn argmax() {
        // Initialize the model with weights (loaded from the exported file)
        let model: argmax::Model<Backend> = argmax::Model::default();

        let device = Default::default();
        // Run the model
        let input = Tensor::<Backend, 2>::from_floats([[1., 2., 3.], [4., 5., 6.]], &device);
        let output = model.forward(input);

        #[cfg(feature = "int-i32")]
        let expected = TensorData::from([[2i32], [2]]);

        #[cfg(not(feature = "int-i32"))]
        let expected = TensorData::from([[2i64], [2]]);

        output.to_data().assert_eq(&expected, true);
    }
}
