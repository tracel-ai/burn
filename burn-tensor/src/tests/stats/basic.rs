#[burn_tensor_testgen::testgen(stats)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};

    #[test]
    fn test_var() {
        let data = Data::from([[0.5, 1.8, 0.2, -2.0], [3.0, -4.0, 5.0, 0.0]]);
        let tensor = Tensor::<TestBackend, 2>::from_data(data);

        let data_actual = tensor.var(1).into_data();

        let data_expected = Data::from([[2.4892], [15.3333]]);
        data_expected.assert_approx_eq(&data_actual, 3);
    }

    #[test]
    fn test_display_2d_tensor() {
        let data = Data::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        let tensor: burn_tensor::Tensor<TestBackend, 2, burn_tensor::Int> = Tensor::from_data(data);

        let mut output = format!("{}", tensor);
        let mut expected = "Tensor {\n  data: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],\n  shape:   [3, 3],\n  device:  Cpu,\n  backend: \"ndarray\",\n  dtype:   \"int\",\n}";
        assert_eq!(output, expected);

        let data_float = Data::from([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]);
        let tensor_float: burn_tensor::Tensor<TestBackend, 2, burn_tensor::Float> =
            Tensor::from_data(data_float);

        output = format!("{}", tensor_float);
        expected = "Tensor {\n  data: [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]],\n  shape:   [3, 3],\n  device:  Cpu,\n  backend: \"ndarray\",\n  dtype:   \"float\",\n}";
        assert_eq!(output, expected);

        let data_bool = Data::from([[true, false, true], [false, true, false], [false, true, true]]);
        let tensor_bool: burn_tensor::Tensor<TestBackend, 2, burn_tensor::Bool> =
            Tensor::from_data(data_bool);

        output = format!("{}", tensor_bool);
        expected = "Tensor {\n  data: [[true, false, true], [false, true, false], [false, true, true]],\n  shape:   [3, 3],\n  device:  Cpu,\n  backend: \"ndarray\",\n  dtype:   \"bool\",\n}";
        assert_eq!(output, expected);
    }

    #[test]
    fn test_display_3d_tensor() {
        let data = Data::from([
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
        ]);
        let tensor: burn_tensor::Tensor<TestBackend, 3, burn_tensor::Int> = Tensor::from_data(data);

        let output = format!("{}", tensor);
        let expected = "Tensor {\n  data: [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], \
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]],\n  shape:   [2, 3, 4],\n  device:  Cpu,\n  backend: \"ndarray\",\n  dtype:   \"int\",\n}";
        assert_eq!(output, expected);
    }

    #[test]
    fn test_display_4d_tensor() {
        let data = Data::from([
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]],
        ]);

        let tensor: burn_tensor::Tensor<TestBackend, 4, burn_tensor::Int> = Tensor::from_data(data);

        let output = format!("{}", tensor);
        let expected = "Tensor {\n  data: [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]],\n  shape:   [2, 2, 2, 3],\n  device:  Cpu,\n  backend: \"ndarray\",\n  dtype:   \"int\",\n}";
        assert_eq!(output, expected);
    }
}
