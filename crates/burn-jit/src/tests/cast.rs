#[burn_tensor_testgen::testgen(cast)]
mod tests {
    use super::*;
    use burn_tensor::{Int, Tensor, TensorData};

    #[test]
    fn should_cast_int_to_float() {
        const START: usize = 0;
        const END: usize = 100;

        let device = Default::default();
        let tensor = Tensor::<TestBackend, 1, Int>::arange(START as i64..END as i64, &device);

        let data_int = tensor.to_data();
        let data_int = data_int.as_slice::<i32>().unwrap();
        let data_float = tensor.float().into_data();
        let data_float = data_float.as_slice::<f32>().unwrap();

        for i in START..END {
            assert_eq!(data_int[i], i as i32);
            assert_eq!(data_float[i], i as f32);
        }
    }

    #[test]
    fn should_cast_bool_to_int() {
        let device = Default::default();

        let tensor_1 =
            Tensor::<TestBackend, 2>::from_floats([[1., 0., 3.], [0., 0., 900.]], &device);
        let tensor_2: Tensor<TestBackend, 2, Int> = tensor_1.clone().greater_elem(0.0).int();

        tensor_2
            .to_data()
            .assert_eq(&TensorData::from([[1, 0, 1], [0, 0, 1]]), false);
    }

    #[test]
    fn should_cast_bool_to_float() {
        let device = Default::default();

        let tensor_1 =
            Tensor::<TestBackend, 2>::from_floats([[1., 0., 3.], [0., 0., 900.]], &device);
        let tensor_2: Tensor<TestBackend, 2> = tensor_1.clone().greater_elem(0.0).float();

        tensor_2
            .to_data()
            .assert_eq(&TensorData::from([[1., 0., 1.], [0., 0., 1.]]), false);
    }
}
