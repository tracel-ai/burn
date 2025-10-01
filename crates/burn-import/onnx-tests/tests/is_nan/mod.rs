// Import the shared macro
use crate::include_models;
include_models!(is_nan, is_nan_scalar);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn is_nan() {
        let device = Default::default();
        let model: is_nan::Model<TestBackend> = is_nan::Model::new(&device);

        let input1 =
            Tensor::<TestBackend, 2>::from_floats([[1.0, f32::NAN, -9.0, f32::NAN]], &device);

        let output = model.forward(input1);
        let expected = TensorData::from([[false, true, false, true]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn is_nan_scalar() {
        let device = Default::default();
        let model: is_nan_scalar::Model<TestBackend> = is_nan_scalar::Model::new(&device);

        let input1 = f32::NAN;

        let output = model.forward(input1);
        let expected = true;

        assert_eq!(output, expected);
    }
}
