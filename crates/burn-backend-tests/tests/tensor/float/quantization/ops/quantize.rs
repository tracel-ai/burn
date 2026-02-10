use super::*;
use alloc::{vec, vec::Vec};
use burn_tensor::quantization::{
    QParams, QTensorPrimitive, QuantLevel, QuantScheme, QuantStore, QuantValue,
    QuantizationParameters, QuantizedBytes,
};
use burn_tensor::{DType, TensorData};
use burn_tensor::{Tolerance, ops::QuantizedTensor};

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
        scales: TestTensor::from_floats([0.014_173_228], &device),
    };

    let x_q = tensor.clone().quantize(&scheme, qparams);

    let x_q_data = x_q.to_data();
    let expected = TensorData::quantized(
        vec![-127i8, -71, 0, 35],
        [4],
        scheme.with_store(QuantStore::Native),
        &[0.014_173_228], // scale
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
    x.into_data().assert_approx_eq::<FloatElem>(
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
        scheme.with_store(QuantStore::Native),
        &[0.1], // scale
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
    data.assert_approx_eq::<FloatElem>(&TensorData::from(expected), Tolerance::default());
}

#[test]
fn should_quantize_dequantize_symmetric_arange_16x16() {
    let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);

    let input: TestTensor<2> = TestTensorInt::arange(0..256, &Default::default())
        .float()
        .reshape([16, 16]);

    let output = input.quantize_dynamic(&scheme);

    let output = output.dequantize();

    let expected = TensorData::new(
        vec![
            0.0, 0.0, 2.007874, 2.007874, 4.015748, 4.015748, 6.023622, 6.023622, 8.031496,
            8.031496, 10.03937, 10.03937, 12.047244, 12.047244, 14.055119, 14.055119, 16.062992,
            16.062992, 18.070866, 18.070866, 20.07874, 20.07874, 22.086615, 22.086615, 24.094488,
            24.094488, 26.102362, 26.102362, 28.110237, 28.110237, 30.11811, 30.11811, 32.125984,
            32.125984, 34.133858, 34.133858, 36.14173, 36.14173, 38.149605, 38.149605, 40.15748,
            40.15748, 42.165356, 42.165356, 44.17323, 44.17323, 46.181103, 46.181103, 48.188976,
            48.188976, 50.19685, 50.19685, 52.204723, 52.204723, 54.212597, 54.212597, 56.220474,
            56.220474, 58.228348, 58.228348, 60.23622, 60.23622, 62.244095, 62.244095, 64.25197,
            64.25197, 66.25984, 66.25984, 68.267715, 68.267715, 70.27559, 70.27559, 72.28346,
            72.28346, 74.291336, 74.291336, 76.29921, 76.29921, 78.30708, 78.30708, 80.31496,
            80.31496, 82.32284, 82.32284, 84.33071, 84.33071, 86.338585, 86.338585, 88.34646,
            88.34646, 90.35433, 90.35433, 92.362206, 92.362206, 94.37008, 94.37008, 96.37795,
            96.37795, 98.385826, 98.385826, 100.3937, 100.3937, 102.40157, 102.40157, 104.40945,
            104.40945, 106.41732, 106.41732, 108.42519, 108.42519, 110.43307, 110.43307, 112.44095,
            112.44095, 114.44882, 114.44882, 116.456696, 116.456696, 118.46457, 118.46457,
            120.47244, 120.47244, 122.480316, 122.480316, 124.48819, 124.48819, 126.49606,
            126.49606, 128.50394, 128.50394, 130.51181, 130.51181, 132.51968, 132.51968, 134.52756,
            134.52756, 136.53543, 136.53543, 138.5433, 138.5433, 140.55118, 140.55118, 142.55905,
            142.55905, 144.56693, 144.56693, 146.5748, 146.5748, 148.58267, 148.58267, 150.59055,
            150.59055, 152.59842, 152.59842, 154.6063, 154.6063, 156.61417, 156.61417, 158.62204,
            158.62204, 160.62991, 160.62991, 162.6378, 162.6378, 164.64568, 164.64568, 166.65355,
            166.65355, 168.66142, 168.66142, 170.6693, 170.6693, 172.67717, 172.67717, 174.68504,
            174.68504, 176.69292, 176.69292, 178.70079, 178.70079, 180.70866, 180.70866, 182.71654,
            182.71654, 184.72441, 184.72441, 186.73228, 186.73228, 188.74016, 188.74016, 190.74803,
            190.74803, 192.7559, 192.7559, 194.76378, 194.76378, 196.77165, 196.77165, 198.77953,
            198.77953, 200.7874, 200.7874, 202.79527, 202.79527, 204.80315, 204.80315, 206.81102,
            206.81102, 208.8189, 208.8189, 210.82677, 210.82677, 212.83464, 212.83464, 214.84251,
            214.84251, 216.85039, 216.85039, 218.85826, 218.85826, 220.86613, 220.86613, 222.87401,
            222.87401, 224.8819, 224.8819, 226.88977, 226.88977, 228.89764, 228.89764, 230.90552,
            230.90552, 232.91339, 232.91339, 234.92126, 234.92126, 236.92914, 236.92914, 238.93701,
            238.93701, 240.94489, 240.94489, 242.95276, 242.95276, 244.96063, 244.96063, 246.9685,
            246.9685, 248.97638, 248.97638, 250.98425, 250.98425, 252.99213, 252.99213, 255.0,
            255.0,
        ],
        [16, 16],
    );

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

#[test]
fn should_quantize_dequantize_symmetric_per_block_arange_16x16() {
    let scheme = QuantizedTensor::<TestBackend>::default_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::block([2, 16]));

    let input: TestTensor<2> = TestTensorInt::arange(0..256, &Default::default())
        .float()
        .reshape([16, 16]);

    let output = input.quantize_dynamic(&scheme);

    let output = output.dequantize();

    let expected = TensorData::new(
        vec![
            0.0, 0.97637796, 1.9527559, 2.929134, 3.9055119, 4.88189, 6.102362, 7.07874, 8.055119,
            9.031496, 10.0078745, 10.984252, 11.96063, 12.937008, 13.913386, 14.889764, 16.110237,
            17.086615, 18.062992, 19.03937, 20.015749, 20.992126, 21.968504, 22.944881, 23.92126,
            24.897638, 26.11811, 27.094488, 28.070866, 29.047245, 30.023623, 31.0, 32.244095,
            33.23622, 34.228348, 35.220474, 36.212597, 37.204723, 38.19685, 39.188976, 40.181103,
            41.17323, 42.165356, 43.157482, 44.149605, 45.14173, 46.133858, 47.125984, 48.11811,
            49.110237, 50.102364, 51.09449, 52.086613, 53.07874, 54.070866, 55.062992, 56.05512,
            57.047245, 58.03937, 59.031498, 60.02362, 61.015747, 62.007874, 63.0, 64.33071,
            65.07874, 65.826775, 67.32284, 68.07087, 68.8189, 70.314964, 71.062996, 71.81102,
            73.30708, 74.055115, 74.80315, 76.29921, 77.04724, 77.79527, 79.291336, 80.03937,
            80.7874, 82.28346, 83.031494, 83.779526, 85.27559, 86.02362, 86.77165, 88.267715,
            89.01575, 89.76378, 91.25984, 92.00787, 92.755905, 94.25197, 95.0, 96.0, 97.0, 98.0,
            99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
            111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0,
            123.0, 124.0, 125.0, 126.0, 127.0, 127.70079, 128.95276, 130.20473, 131.4567, 131.4567,
            132.70866, 133.96063, 135.2126, 136.46457, 136.46457, 137.71654, 138.9685, 140.22047,
            141.47244, 141.47244, 142.72441, 143.97638, 145.22835, 146.48032, 146.48032, 147.73228,
            148.98425, 150.23622, 151.48819, 151.48819, 152.74016, 153.99213, 155.2441, 156.49606,
            156.49606, 157.74803, 159.0, 159.41733, 160.92126, 162.4252, 162.4252, 163.92914,
            165.43307, 165.43307, 166.93701, 168.44095, 168.44095, 169.94489, 171.44882, 171.44882,
            172.95276, 174.4567, 174.4567, 175.96063, 177.46457, 177.46457, 178.9685, 180.47244,
            180.47244, 181.97638, 183.48032, 183.48032, 184.98425, 186.48819, 186.48819, 187.99213,
            189.49606, 189.49606, 191.0, 191.3937, 193.14961, 193.14961, 194.90552, 196.66142,
            196.66142, 198.41733, 198.41733, 200.17323, 200.17323, 201.92914, 203.68504, 203.68504,
            205.44095, 205.44095, 207.19685, 207.19685, 208.95276, 210.70866, 210.70866, 212.46457,
            212.46457, 214.22047, 214.22047, 215.97638, 217.73228, 217.73228, 219.48819, 219.48819,
            221.2441, 221.2441, 223.0, 224.8819, 224.8819, 226.88977, 226.88977, 228.89764,
            228.89764, 230.90552, 230.90552, 232.91339, 232.91339, 234.92126, 234.92126, 236.92914,
            236.92914, 238.93701, 238.93701, 240.94489, 240.94489, 242.95276, 242.95276, 244.96063,
            244.96063, 246.9685, 246.9685, 248.97638, 248.97638, 250.98425, 250.98425, 252.99213,
            252.99213, 255.0, 255.0,
        ],
        [16, 16],
    );

    output
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::default());
}

fn should_quantize_transposed(tensor: Tensor<TestBackend, 2>, scheme: QuantScheme) {
    let tensor_t = tensor.clone().transpose();

    let output = tensor_t.quantize_dynamic(&scheme).dequantize().transpose();

    tensor
        .into_data()
        .assert_approx_eq::<FloatElem>(&output.into_data(), Tolerance::permissive());
}

#[test]
fn should_quantize_symmetric_int8_transposed_8x32() {
    let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);

    let tensor = TestTensorInt::arange(0..256, &Default::default())
        .float()
        .reshape([8, 32]);
    should_quantize_transposed(tensor, scheme);
}

#[test]
fn should_quantize_symmetric_int8_transposed_48x64() {
    let scheme = QuantizedTensor::<TestBackend>::default_scheme().with_value(QuantValue::Q8S);

    let tensor = TestTensorInt::arange(0..3072, &Default::default())
        .float()
        .reshape([48, 64]);
    should_quantize_transposed(tensor, scheme);
}

#[test]
fn should_quantize_symmetric_per_block_int8_transposed_32x64() {
    let scheme = QuantizedTensor::<TestBackend>::default_scheme()
        .with_value(QuantValue::Q8S)
        .with_level(QuantLevel::block([32]));

    let tensor = TestTensorInt::arange(0..2048, &Default::default())
        .float()
        .div_scalar(2048.)
        .reshape([32, 64]);
    should_quantize_transposed(tensor, scheme);
}
