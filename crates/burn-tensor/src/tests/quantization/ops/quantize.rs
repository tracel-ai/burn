#[burn_tensor_testgen::testgen(quantize)]

mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use burn_tensor::quantization::{
        QParams, QuantScheme, QuantizationParameters, QuantizationStrategy, QuantizedBytes,
        SymmetricQuantization,
    };
    use burn_tensor::{DType, Tensor, TensorData};
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    fn get_q_params(data: TensorData) -> QParams<Vec<f32>> {
        let num_elements = data.num_elements();
        let scheme = if let DType::QFloat(scheme) = data.dtype {
            scheme
        } else {
            unreachable!()
        };
        let q_bytes = QuantizedBytes {
            bytes: data.into_bytes(),
            scheme,
            num_elements,
        };
        q_bytes.into_vec_i8().1
    }

    #[test]
    fn should_support_quantize_symmetric_int8() {
        let device = Default::default();
        let tensor = TestTensor::<1>::from_floats([-1.8, -1.0, 0.0, 0.5], &device);
        let scheme = QuantScheme::default();
        let qparams = QuantizationParameters {
            scales: Tensor::from_floats([0.014_173_228], &device),
        };

        let x_q = tensor.clone().quantize(&scheme, qparams);

        let x_q_data = x_q.to_data();
        let expected = TensorData::quantized(
            vec![-127i8, -71, 0, 35],
            [4],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                0.014_173_228,
            )),
        );

        // Values equality
        x_q_data.assert_eq(&expected, false);

        // Quantization parameters check
        let qparams = get_q_params(x_q_data);
        let expected = get_q_params(expected);
        assert_eq!(qparams.scales.len(), 1);
        assert_eq!(qparams.scales, expected.scales);

        // Dequantize
        let x = x_q.dequantize();

        // Precision 2 for dequantization errors
        x.into_data().assert_approx_eq::<FT>(
            &tensor.into_data(),
            Tolerance::absolute(1e-1).set_relative(1e-2),
        );
    }

    #[test]
    fn should_support_quantize_dynamic_int8() {
        let device = Default::default();
        // NOTE: we use fully representable values since different backend implementations could differ slightly
        // due to rounding discrepancies
        let tensor = TestTensor::<1>::from_floats([5., 0., 4., -12.7], &device);
        let scheme = QuantScheme::default();

        let x_q = tensor.quantize_dynamic(&scheme);

        let expected = TensorData::quantized(
            vec![50i8, 0, 40, -127],
            [4],
            QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(0.1)),
        );

        x_q.into_data().assert_eq(&expected, false);
    }
}
