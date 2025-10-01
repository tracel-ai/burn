use crate::include_models;
include_models!(pow, pow_int);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    #[test]
    fn pow_int_with_tensor_and_scalar() {
        let device = Default::default();
        let model: pow_int::Model<TestBackend> = pow_int::Model::new(&device);

        let input1 = Tensor::<TestBackend, 4, Int>::from_ints([[[[1, 2, 3, 4]]]], &device);
        let input2 = 2;

        let output = model.forward(input1, input2);
        let expected = TensorData::from([[[[1i64, 16, 729, 65536]]]]);

        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn pow_with_tensor_and_scalar() {
        let device = Default::default();
        let model: pow::Model<TestBackend> = pow::Model::new(&device);

        let input1 = Tensor::from_floats([[[[1., 2., 3., 4.]]]], &device);
        let input2 = 2f64;

        let output = model.forward(input1, input2);

        let expected = TensorData::from([[[[1.0000f32, 1.6000e+01, 7.2900e+02, 6.5536e+04]]]]);

        output.to_data().assert_eq(&expected, true);
    }
}
