// Import the shared macro
use crate::include_models;
include_models!(less, less_scalar);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn less() {
        let device = Default::default();
        let model: less::Model<Backend> = less::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<Backend, 2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);
        #[cfg(feature = "bool-u32")]
        let expected = TensorData::from([[0u32, 1, 0, 0]]);

        #[cfg(not(feature = "bool-u32"))]
        let expected = TensorData::from([[false, true, false, false]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn less_scalar() {
        let device = Default::default();
        let model: less_scalar::Model<Backend> = less_scalar::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);

        #[cfg(feature = "bool-u32")]
        let expected = TensorData::from([[0u32, 0, 0, 1]]);

        #[cfg(not(feature = "bool-u32"))]
        let expected = TensorData::from([[false, false, false, true]]);

        output.to_data().assert_eq(&expected, true);
    }
}
