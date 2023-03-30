#[burn_tensor_testgen::testgen(stats)]
mod tests {
    use super::*;
    use burn_tensor::{Data, Tensor};
    use burn_ndarray::NdArrayBackend; // Well this doesn't work

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
        let data = Data::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]); // Is there any sort of narrowing conversion to Int?
        let tensor: burn_tensor::Tensor<burn_ndarray::NdArrayBackend<f32>, 2, burn_tensor::Int> = Tensor::from_data(data);

        let output = format!("{}", tensor);
        let expected = "Tensor {\n  data: [[1, 2, 3], [4, 5, 6], [7, 8, 9]],\n  shape:   [3, 3],\n  device:  Cpu,\n  backend: ndarray,\n  dtype:   int,\n}";
        assert_eq!(output, expected);
    }
}
