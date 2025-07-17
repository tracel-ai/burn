// Import the shared macro
use crate::include_models;
include_models!(greater_or_equal, greater_or_equal_scalar);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::Backend;

    #[test]
    fn greater_or_equal() {
        let device = Default::default();
        let model: greater_or_equal::Model<Backend> = greater_or_equal::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 4.0, 9.0, 25.0]], &device);
        let input2 = Tensor::<Backend, 2>::from_floats([[1.0, 5.0, 8.0, -25.0]], &device);

        let output = model.forward(input1, input2);

        #[cfg(feature = "bool-u32")]
        let expected = TensorData::from([[1u32, 0, 1, 1]]);

        #[cfg(not(feature = "bool-u32"))]
        let expected = TensorData::from([[true, false, true, true]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn greater_or_equal_scalar() {
        let device = Default::default();
        let model: greater_or_equal_scalar::Model<Backend> =
            greater_or_equal_scalar::Model::new(&device);

        let input1 = Tensor::<Backend, 2>::from_floats([[1.0, 4.0, 9.0, 0.5]], &device);
        let input2 = 1.0;

        let output = model.forward(input1, input2);

        #[cfg(feature = "bool-u32")]
        let expected = TensorData::from([[1u32, 1, 1, 0]]);

        #[cfg(not(feature = "bool-u32"))]
        let expected = TensorData::from([[true, true, true, false]]);

        output.to_data().assert_eq(&expected, true);
    }
}
