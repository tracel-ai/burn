#[burn_tensor_testgen::testgen(q_gather_scatter)]
mod tests {
    use super::*;
    use burn_tensor::quantization::{QuantizationStrategy, SymmetricQuantization};
    use burn_tensor::{Tensor, TensorData};

    #[test]
    fn should_gather_1d_dim0() {
        let device = Default::default();
        // Quantized [0.0, 1.0, 2.0]
        let data = TensorData::quantized(
            vec![0i8, 64, 127],
            [3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.015748031)),
        );
        let tensor = TestTensor::<1>::from_data(data, &device);
        let indices = TestTensorInt::from_ints([1, 1, 0, 1, 2], &device);

        let output = tensor.gather(0, indices);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([1.0, 1.0, 0.0, 1.0, 2.0]), 1);
    }

    #[test]
    fn should_gather_2d_dim0() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        let indices = TestTensorInt::from_ints([[0, 1, 0], [1, 0, 1]], &device);

        let output = tensor.gather(0, indices);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[0.0, 4.0, 2.0], [3.0, 1.0, 5.0]]), 1);
    }

    #[test]
    fn should_gather_2d_dim1() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        let indices = TestTensorInt::from_ints([[2, 1, 0, 0], [2, 0, 1, 2]], &device);

        let output = tensor.gather(1, indices);

        // Precision 1 to approximate de/quantization errors
        output.dequantize().into_data().assert_approx_eq(
            &TensorData::from([[2.0, 1.0, 0.0, 0.0], [5.0, 3.0, 4.0, 5.0]]),
            1,
        );
    }

    #[test]
    fn should_gather_3d_dim1() {
        let device = Default::default();
        // Quantized  [
        //     [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        //     [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        // ]
        let data = TensorData::quantized(
            vec![0i8, 12, 23, 35, 46, 58, 69, 81, 92, 104, 115, 127],
            [2, 2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.08661418)),
        );
        let tensor = TestTensor::<3>::from_data(data, &device);
        let indices =
            TestTensorInt::from_ints([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 1]]], &device);

        let output = tensor.gather(1, indices);
        let expected = TensorData::from([
            [[3.0, 1.0, 2.0], [0.0, 4.0, 2.0]],
            [[6.0, 7.0, 11.0], [6.0, 10.0, 11.0]],
        ]);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&expected, 1);
    }

    #[test]
    fn should_gather_2d_only_1dim() {
        let device = Default::default();
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![0i8, 25, 51, 76, 102, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        let indices = TestTensorInt::<2>::from_ints([[1, 2]], &device).reshape([2, 1]);

        let output = tensor.gather(1, indices);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[1.0], [5.0]]), 1);
    }

    #[test]
    fn should_scatter_1d() {
        let device = Default::default();
        // Quantized [0.0, 0.0, 0.0]
        let data = TensorData::quantized(
            vec![0i8, 0, 0],
            [3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.1)),
        );
        let tensor = TestTensor::<1>::from_data(data, &device);
        // Quantized [5.0, 4.0, 3.0]
        let data = TensorData::quantized(
            vec![127i8, 102, 76],
            [3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.03937008)),
        );
        let values = TestTensor::<1>::from_data(data, &device);
        let indices = TestTensorInt::from_ints([1, 0, 2], &device);

        let output = tensor.scatter(0, indices, values);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([4.0, 5.0, 3.0]), 1);
    }

    #[test]
    fn should_scatter_2d_dim0() {
        let device = Default::default();
        // Quantized [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        let data = TensorData::quantized(
            vec![0i8, 0, 0, 0, 0, 0],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.1)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        // Quantized [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let data = TensorData::quantized(
            vec![21i8, 42, 64, 85, 106, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.047244094)),
        );
        let values = TestTensor::<2>::from_data(data, &device);
        let indices = TestTensorInt::from_ints([[1, 0, 1], [1, 1, 0]], &device);

        let output = tensor.scatter(0, indices, values);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[0.0, 2.0, 6.0], [5.0, 5.0, 3.0]]), 1);
    }

    #[test]
    fn should_scatter_2d_dim1() {
        let device = Default::default();
        // Quantized [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        let data = TensorData::quantized(
            vec![0i8, 0, 0, 0, 0, 0],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.1)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        // Quantized [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let data = TensorData::quantized(
            vec![21i8, 42, 64, 85, 106, 127],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.047244094)),
        );
        let values = TestTensor::<2>::from_data(data, &device);
        let indices = TestTensorInt::from_ints([[1, 0, 2], [1, 2, 0]], &device);

        let output = tensor.scatter(1, indices, values);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[2.0, 1.0, 3.0], [6.0, 4.0, 5.0]]), 1);
    }

    #[test]
    fn should_scatter_3d_dim1() {
        let device = Default::default();
        // Quantized  [
        //     [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
        //     [[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        // ]
        let data = TensorData::quantized(
            vec![0i8, 12, 23, 35, 46, 58, 69, 81, 92, 104, 115, 127],
            [2, 2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.08661418)),
        );
        let tensor = TestTensor::<3>::from_data(data, &device);
        // Quantized  [
        //     [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]],
        //     [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
        // ]
        let data = TensorData::quantized(
            vec![66i8, 72, 77, 83, 88, 94, 99, 105, 110, 116, 121, 127],
            [2, 2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.18110237)),
        );
        let values = TestTensor::<3>::from_data(data, &device);
        let indices =
            TestTensorInt::from_ints([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 1, 1]]], &device);

        let output = tensor.scatter(1, indices, values);
        let expected = TensorData::from([
            [[15.0, 14.0, 33.0], [15.0, 20.0, 5.0]],
            [[45.0, 26.0, 8.0], [9.0, 32.0, 54.0]],
        ]);

        // Set higher tolerance (0.2) due to larger de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq_diff(&expected, 0.2);
    }

    #[test]
    fn should_scatter_2d_dim1_diff_shape() {
        let device = Default::default();
        // Quantized [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        let data = TensorData::quantized(
            vec![0i8, 0, 0, 0, 0, 0],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.1)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        // Quantized [[1.0], [4.0]]
        let data = TensorData::quantized(
            vec![32i8, 127],
            [2, 1],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.031496063)),
        );
        let values = TestTensor::<2>::from_data(data, &device);
        let indices = TestTensorInt::from_ints([[1], [2]], &device);

        let output = tensor.scatter(1, indices, values);

        // Precision 1 to approximate de/quantization errors
        output
            .dequantize()
            .into_data()
            .assert_approx_eq(&TensorData::from([[0.0, 1.0, 0.0], [0.0, 0.0, 4.0]]), 1);
    }

    #[test]
    #[should_panic]
    fn scatter_should_panic_on_mismatch_of_shapes() {
        let device = Default::default();
        // Quantized [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        let data = TensorData::quantized(
            vec![0i8, 0, 0, 0, 0, 0],
            [2, 3],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.1)),
        );
        let tensor = TestTensor::<2>::from_data(data, &device);
        // Quantized [1.0, 4.0]
        let data = TensorData::quantized(
            vec![32i8, 127],
            [2],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.031496063)),
        );
        let values = TestTensor::<2>::from_data(data, &device);
        let indices = TestTensorInt::from_ints([1, 0, 2], &device);

        tensor.scatter(0, indices, values);
    }
}
