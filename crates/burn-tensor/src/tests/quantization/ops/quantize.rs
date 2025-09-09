#[burn_tensor_testgen::testgen(quantize)]
mod tests {
    use super::*;
    use alloc::{vec, vec::Vec};
    use burn_tensor::quantization::{
        QParams, QTensorPrimitive, QuantStore, QuantValue, QuantizationParameters,
        QuantizationStrategy, QuantizedBytes, SymmetricQuantization,
    };
    use burn_tensor::{DType, Tensor, TensorData};
    use burn_tensor::{
        Tolerance,
        ops::{FloatElem, QuantizedTensor},
    };
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
        let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);
        let qparams = QuantizationParameters {
            scales: Tensor::from_floats([0.014_173_228], &device),
        };

        let x_q = tensor.clone().quantize(&scheme, qparams);

        let x_q_data = x_q.to_data();
        let expected = TensorData::quantized(
            vec![-127i8, -71, 0, 35],
            [4],
            QuantizationStrategy::PerTensorSymmetric(SymmetricQuantization::init(
                0.014_173_228,
                QuantValue::Q8S,
            )),
            scheme,
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
        let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);

        let x_q = tensor.quantize_dynamic(&scheme);

        let expected = TensorData::quantized(
            vec![50i8, 0, 40, -127],
            [4],
            QuantizationStrategy::PerTensorSymmetric(SymmetricQuantization::init(
                0.1,
                QuantValue::Q8S,
            )),
            scheme,
        );

        x_q.into_data().assert_eq(&expected, false);
    }

    #[test]
    fn should_quantize_dequantize_symmetric_single_with_transform() {
        let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);
        let input = TestTensorInt::<1>::arange(0..32, &Default::default()).float();

        let quant = input.quantize_dynamic(&scheme);
        let result = quant * 10;

        let data = result.into_data();
        let expected = [
            0.0, 9.76378, 19.52756, 29.29134, 39.05512, 48.818897, 61.02362, 70.7874, 80.551186,
            90.31496, 100.07874, 109.84252, 119.60631, 129.37009, 139.13387, 148.89764, 161.10237,
            170.86615, 180.62991, 190.39369, 200.15749, 209.92126, 219.68504, 229.44882, 239.21262,
            248.97638, 261.1811, 270.9449, 280.70865, 290.47244, 300.23624, 310.0,
        ];
        data.assert_approx_eq::<FT>(&TensorData::from(expected), Tolerance::default());
    }

    #[test]
    fn should_quantize_dequantize_symmetric_arange_16x16() {
        let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);

        let input = Tensor::<TestBackend, 1, burn_tensor::Int>::arange(0..256, &Default::default())
            .float()
            .reshape([16, 16]);

        let output = input.quantize_dynamic(&scheme);

        let output = output.dequantize();

        let expected = TensorData::new(
            vec![
                0.0, 0.0, 2.007874, 2.007874, 4.015748, 4.015748, 6.023622, 6.023622, 8.031496,
                8.031496, 10.03937, 10.03937, 12.047244, 12.047244, 14.055119, 14.055119,
                16.062992, 16.062992, 18.070866, 18.070866, 20.07874, 20.07874, 22.086615,
                22.086615, 24.094488, 24.094488, 26.102362, 26.102362, 28.110237, 28.110237,
                30.11811, 30.11811, 32.125984, 32.125984, 34.133858, 34.133858, 36.14173, 36.14173,
                38.149605, 38.149605, 40.15748, 40.15748, 42.165356, 42.165356, 44.17323, 44.17323,
                46.181103, 46.181103, 48.188976, 48.188976, 50.19685, 50.19685, 52.204723,
                52.204723, 54.212597, 54.212597, 56.220474, 56.220474, 58.228348, 58.228348,
                60.23622, 60.23622, 62.244095, 62.244095, 64.25197, 64.25197, 66.25984, 66.25984,
                68.267715, 68.267715, 70.27559, 70.27559, 72.28346, 72.28346, 74.291336, 74.291336,
                76.29921, 76.29921, 78.30708, 78.30708, 80.31496, 80.31496, 82.32284, 82.32284,
                84.33071, 84.33071, 86.338585, 86.338585, 88.34646, 88.34646, 90.35433, 90.35433,
                92.362206, 92.362206, 94.37008, 94.37008, 96.37795, 96.37795, 98.385826, 98.385826,
                100.3937, 100.3937, 102.40157, 102.40157, 104.40945, 104.40945, 106.41732,
                106.41732, 108.42519, 108.42519, 110.43307, 110.43307, 112.44095, 112.44095,
                114.44882, 114.44882, 116.456696, 116.456696, 118.46457, 118.46457, 120.47244,
                120.47244, 122.480316, 122.480316, 124.48819, 124.48819, 126.49606, 126.49606,
                128.50394, 128.50394, 130.51181, 130.51181, 132.51968, 132.51968, 134.52756,
                134.52756, 136.53543, 136.53543, 138.5433, 138.5433, 140.55118, 140.55118,
                142.55905, 142.55905, 144.56693, 144.56693, 146.5748, 146.5748, 148.58267,
                148.58267, 150.59055, 150.59055, 152.59842, 152.59842, 154.6063, 154.6063,
                156.61417, 156.61417, 158.62204, 158.62204, 160.62991, 160.62991, 162.6378,
                162.6378, 164.64568, 164.64568, 166.65355, 166.65355, 168.66142, 168.66142,
                170.6693, 170.6693, 172.67717, 172.67717, 174.68504, 174.68504, 176.69292,
                176.69292, 178.70079, 178.70079, 180.70866, 180.70866, 182.71654, 182.71654,
                184.72441, 184.72441, 186.73228, 186.73228, 188.74016, 188.74016, 190.74803,
                190.74803, 192.7559, 192.7559, 194.76378, 194.76378, 196.77165, 196.77165,
                198.77953, 198.77953, 200.7874, 200.7874, 202.79527, 202.79527, 204.80315,
                204.80315, 206.81102, 206.81102, 208.8189, 208.8189, 210.82677, 210.82677,
                212.83464, 212.83464, 214.84251, 214.84251, 216.85039, 216.85039, 218.85826,
                218.85826, 220.86613, 220.86613, 222.87401, 222.87401, 224.8819, 224.8819,
                226.88977, 226.88977, 228.89764, 228.89764, 230.90552, 230.90552, 232.91339,
                232.91339, 234.92126, 234.92126, 236.92914, 236.92914, 238.93701, 238.93701,
                240.94489, 240.94489, 242.95276, 242.95276, 244.96063, 244.96063, 246.9685,
                246.9685, 248.97638, 248.97638, 250.98425, 250.98425, 252.99213, 252.99213, 255.0,
                255.0,
            ],
            [16, 16],
        );

        output
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }
}
