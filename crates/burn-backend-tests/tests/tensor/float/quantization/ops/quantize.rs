use super::*;
use alloc::{vec, vec::Vec};
use burn_tensor::Tolerance;
use burn_tensor::quantization::{
    BlockExtent, BlockScale, BlockSize, DecodedScales, QuantScheme, QuantStore, QuantValue,
    QuantizationParameters, QuantizedBytes, ScaleDtype,
};
use burn_tensor::{DType, Device, Element, TensorData};

fn get_q_params(data: TensorData) -> DecodedScales {
    let scheme = if let DType::QFloat(scheme) = data.dtype() {
        scheme
    } else {
        unreachable!()
    };
    let q_bytes = QuantizedBytes {
        shape: data.shape().clone(),
        bytes: data.into_bytes(),
        scheme,
    };
    q_bytes.into_vec_i8().1
}

#[test]
fn should_support_quantize_symmetric_int8() {
    // Strict equality was based on full precision
    if !matches!(FloatElem::dtype(), DType::F32) {
        return;
    }
    let device = Default::default();
    let tensor = TestTensor::<1>::from_data([-1.8, -1.0, 0.0, 0.5], &device);
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    let qparams = QuantizationParameters {
        scales: TestTensor::from_data([0.014_173_228], &device),
        coarser: None,
    };

    let x_q = tensor.clone().quantize(&scheme, qparams);

    let x_q_data = x_q.to_data();
    let expected = TensorData::quantized(
        vec![-127i8, -71, 0, 35],
        [4],
        scheme.with_store(QuantStore::Native),
        &[0.014_173_228], // scale,
        None,
    );

    // Values equality
    x_q_data.assert_eq(&expected, false);

    // Quantization parameters check
    let qparams = get_q_params(x_q_data);
    let expected = get_q_params(expected);
    assert_eq!(qparams.block.len(), 1);
    // TODO: check scales
    assert_eq!(qparams, expected);

    // Dequantize
    let x = x_q.dequantize();

    x.into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), Tolerance::rel_abs(1e-1, 1e-2));
}

#[test]
fn should_support_quantize_dynamic_int8() {
    let device = Default::default();
    // NOTE: we use fully representable values since different backend implementations could differ slightly
    // due to rounding discrepancies
    let tensor = TestTensor::<1>::from_data([5., 0., 4., -12.7], &device);
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);

    let x_q = tensor.quantize_dynamic(&scheme);

    let expected = TensorData::quantized(
        vec![50i8, 0, 40, -127],
        [4],
        scheme.with_store(QuantStore::Native),
        &[0.1], // scale,
        None,
    );

    x_q.into_data().assert_eq(&expected, false);
}

#[test]
fn should_quantize_dequantize_symmetric_single_with_transform() {
    let device = Default::default();
    let input = TestTensorInt::<1>::arange(0..32, &device).float();
    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);

    let quant = input.quantize_dynamic(&scheme);
    let result = quant * 10;

    let data = result.into_data();
    let expected = [
        0.0, 9.76378, 19.52756, 29.29134, 39.05512, 48.818897, 61.02362, 70.7874, 80.551186,
        90.31496, 100.07874, 109.84252, 119.60631, 129.37009, 139.13387, 148.89764, 161.10237,
        170.86615, 180.62991, 190.39369, 200.15749, 209.92126, 219.68504, 229.44882, 239.21262,
        248.97638, 261.1811, 270.9449, 280.70865, 290.47244, 300.23624, 310.0,
    ];
    data.assert_approx_eq::<FloatElem>(&TensorData::from(expected), Tolerance::permissive());
}

#[test]
fn should_quantize_dequantize_symmetric_arange_16x16() {
    let device = Default::default();

    let input: TestTensor<2> = TestTensorInt::arange(0..256, &device)
        .float()
        .div_scalar(256.)
        .reshape([16, 16]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    let output = input.clone().quantize_dynamic(&scheme);
    let output = output.dequantize();

    output.into_data().assert_approx_eq::<FloatElem>(
        &input.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

#[test]
fn should_quantize_dequantize_symmetric_per_block_arange_16x16() {
    let device = Default::default();

    let input: TestTensor<2> = TestTensorInt::arange(0..256, &device)
        .float()
        .div_scalar(256.)
        .reshape([16, 16]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_levels([BlockScale::per_block([2, 16], ScaleDtype::F32)]);

    let output = input.clone().quantize_dynamic(&scheme);
    let output = output.dequantize();

    output.into_data().assert_approx_eq::<FloatElem>(
        &input.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

/// A block that does not span the trailing dimension is a rectangle, not a run of the flat
/// storage. Each `[2, 4]` block here holds one magnitude, so its scale is that magnitude over
/// 127; chunking the flat storage in eights would instead pair the two blocks of a row and give
/// both the larger scale. Four wide, so a packed store still keeps one word inside one block.
#[test]
fn should_quantize_blocks_that_do_not_span_the_trailing_dim() {
    let device = Default::default();

    let input = TestTensor::<2>::from_data(
        [
            [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0],
            [100.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0],
            [100.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0],
        ],
        &device,
    );

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_levels([BlockScale::per_block([2, 4], ScaleDtype::F32)]);

    let quantized = input.clone().quantize_dynamic(&scheme);

    let scales = get_q_params(quantized.to_data()).block;
    let expected = [1.0f32, 10.0, 100.0, 1000.0].map(|magnitude| magnitude / 127.0);
    TensorData::new(scales, [2, 2]).assert_approx_eq::<f32>(
        &TensorData::new(expected.to_vec(), [2, 2]),
        Tolerance::relative(1e-3),
    );

    // Every block is uniform, so a right grid reconstructs it exactly and a wrong one is off by
    // the ratio between the two magnitudes it merged.
    quantized
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&input.into_data(), Tolerance::relative(1e-2));
}

/// Bit equality rather than a tolerance: both paths reconstruct from the same stored scales, so
/// any difference means a level was rounded, folded or sliced differently on one of them.
///
/// WGSL has no e4m3, so a ue4m3 scale reconstructs as NaN there. The f16 sibling below covers the
/// byte layout on those backends.
#[cfg(not(feature = "wgpu"))]
#[test]
fn should_round_trip_two_level_through_bytes() {
    let device = Default::default();

    let input: TestTensor<2> = TestTensorInt::arange(0..256, &device)
        .float()
        .div_scalar(256.)
        .reshape([16, 16]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_levels([
            BlockScale::per_block([2, 16], ScaleDtype::UE4M3),
            BlockScale::per_tensor(ScaleDtype::F32),
        ]);

    let quantized = input.quantize_dynamic(&scheme);
    let direct = quantized.clone().dequantize().into_data();

    let reloaded = TestTensor::<2>::from_data(quantized.into_data(), &device);
    let round_tripped = reloaded.dequantize().into_data();

    round_tripped.assert_eq(&direct, true);
}

/// The per-tensor scale exists so that accuracy stops depending on how large the weights are.
///
/// WGSL has no e4m3, so a ue4m3 scale reconstructs as NaN there.
#[cfg(not(feature = "wgpu"))]
#[test]
fn two_level_error_should_not_track_weight_magnitude() {
    use burn_tensor::ElementConversion;

    let device = Default::default();

    let big: TestTensor<2> = TestTensorInt::arange(-128..128, &device)
        .float()
        .div_scalar(128.)
        .reshape([16, 16]);
    let small = big.clone().mul_scalar(0.02);

    let base = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    let two_level = base.with_levels([
        BlockScale::per_block([2, 16], ScaleDtype::UE4M3),
        BlockScale::per_tensor(ScaleDtype::F32),
    ]);
    let one_level = base.with_levels([BlockScale::per_block([2, 16], ScaleDtype::UE4M3)]);

    let error = |tensor: TestTensor<2>, scheme: &QuantScheme| -> f32 {
        let reference = tensor.clone();
        let dequantized = tensor.quantize_dynamic(scheme).dequantize();
        let diff: FloatElem = (reference.clone() - dequantized).abs().sum().into_scalar();
        let total: FloatElem = reference.abs().sum().into_scalar();
        diff.elem::<f32>() / total.elem::<f32>()
    };

    let big_error = error(big.clone(), &two_level);
    let small_error = error(small.clone(), &two_level);

    assert!(
        (small_error - big_error).abs() / big_error < 0.2,
        "two-level error should be the same at either magnitude, \
         got {big_error} and {small_error}"
    );

    // A one-level scheme's scales have to cover the magnitude themselves, and underflow here.
    let small_one_level = error(small, &one_level);
    assert!(
        small_one_level > small_error * 2.0,
        "one-level 8-bit scales should degrade on small weights where two-level does not, \
         got one-level {small_one_level} and two-level {small_error}"
    );
}

/// Compare a view applied to a quantized tensor with the same view applied after dequantization.
/// Both paths reconstruct from the same quantized values and scales, so they must be bit-exact.
fn assert_layout_matches_dequantize<const D: usize>(
    layout_quantized: Tensor<D>,
    layout_dequantized: Tensor<D>,
) {
    layout_quantized
        .dequantize()
        .into_data()
        .assert_eq(&layout_dequantized.into_data(), true);
}

fn should_quantize_after_transpose<const D: usize>(tensor: Tensor<D>, scheme: QuantScheme) {
    let tensor_t = tensor.clone().transpose();

    let output = tensor_t.quantize_dynamic(&scheme).dequantize().transpose();

    tensor.into_data().assert_approx_eq::<FloatElem>(
        &output.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

fn should_transpose_after_quantize_match_dequantized_layout<const D: usize>(
    tensor: Tensor<D>,
    scheme: QuantScheme,
) {
    let quantized = tensor.quantize_dynamic(&scheme);

    assert_layout_matches_dequantize(
        quantized.clone().transpose(),
        quantized.dequantize().transpose(),
    );
}

#[test]
fn should_quantize_symmetric_int8_after_transpose_8x32() {
    let device = Default::default();

    let tensor = TestTensorInt::arange(0..256, &device)
        .float()
        .div_scalar(256.)
        .reshape([8, 32]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    should_quantize_after_transpose(tensor, scheme);
}

#[test]
fn should_transpose_symmetric_int8_after_quantize_8x32_match_dequantized_layout() {
    let device = Default::default();

    let values = (0..256)
        .map(|value| value as f32 / 256.)
        .collect::<Vec<_>>();
    let tensor = TestTensor::<2>::from_data(TensorData::new(values, [8, 32]), &device);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    should_transpose_after_quantize_match_dequantized_layout(tensor, scheme);
}

#[test]
fn should_quantize_symmetric_int8_after_transpose_48x64() {
    let device = Default::default();

    let tensor = TestTensorInt::arange(0..3072, &device)
        .float()
        .div_scalar(3072.)
        .reshape([48, 64]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);
    should_quantize_after_transpose(tensor, scheme);
}

#[test]
fn should_quantize_symmetric_per_block_int8_after_transpose_32x64() {
    let device = Default::default();

    let tensor = TestTensorInt::arange(0..2048, &device)
        .float()
        .div_scalar(2048.)
        .reshape([32, 64]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_levels([BlockScale::per_block([32], ScaleDtype::F32)]);
    should_quantize_after_transpose(tensor, scheme);
}

#[test]
fn should_transpose_symmetric_per_block_int8_after_quantize_32x64_match_dequantized_layout() {
    let device = Default::default();

    let values = (0..2048)
        .map(|value| value as f32 / 2048.)
        .collect::<Vec<_>>();
    let tensor = TestTensor::<2>::from_data(TensorData::new(values, [32, 64]), &device);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_levels([BlockScale::per_block([32], ScaleDtype::F32)]);

    should_transpose_after_quantize_match_dequantized_layout(tensor, scheme);
}

#[test]
fn should_swap_dims_symmetric_per_block_int8_after_quantize_match_dequantized_layout() {
    let device = Default::default();

    let values = (0..256)
        .map(|value| value as f32 / 256.)
        .collect::<Vec<_>>();
    let tensor = TestTensor::<3>::from_data(TensorData::new(values, [2, 8, 16]), &device);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_levels([BlockScale::per_block([1, 2, 16], ScaleDtype::F32)]);
    let quantized = tensor.quantize_dynamic(&scheme);

    assert_layout_matches_dequantize(
        quantized.clone().swap_dims(0, 2),
        quantized.dequantize().swap_dims(0, 2),
    );
}

#[test]
fn should_permute_symmetric_per_block_int8_after_quantize_2x8x16_match_dequantized_layout() {
    let device = Default::default();

    let values = (0..256)
        .map(|value| value as f32 / 256.)
        .collect::<Vec<_>>();
    let tensor = TestTensor::<3>::from_data(TensorData::new(values, [2, 8, 16]), &device);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_levels([BlockScale::per_block([1, 2, 16], ScaleDtype::F32)]);
    let quantized = tensor.quantize_dynamic(&scheme);

    assert_layout_matches_dequantize(
        quantized.clone().permute([1, 2, 0]),
        quantized.dequantize().permute([1, 2, 0]),
    );
}

#[test]
fn should_permute_packed_axis_first_after_quantize_match_dequantized_layout() {
    let device = Default::default();

    let values = (0..256)
        .map(|value| value as f32 / 256.)
        .collect::<Vec<_>>();
    let tensor = TestTensor::<3>::from_data(TensorData::new(values, [2, 8, 16]), &device);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_levels([BlockScale::per_block([1, 2, 16], ScaleDtype::F32)]);
    let quantized = tensor.quantize_dynamic(&scheme);

    assert_layout_matches_dequantize(
        quantized.clone().permute([2, 0, 1]),
        quantized.dequantize().permute([2, 0, 1]),
    );
}

#[test]
fn should_quantize_symmetric_int8_after_permuting_batch_dims() {
    let device = Default::default();

    let tensor = TestTensorInt::arange(0..2048, &device)
        .float()
        .div_scalar(2048.)
        .reshape([2, 4, 8, 32]);

    // Permute [0,1,2,3] -> [1,2,0,3]
    // This rearranges batch dims but keeps packed dim in place
    let tensor_permuted = tensor.clone().permute([1, 2, 0, 3]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S);

    let output = tensor_permuted
        .quantize_dynamic(&scheme)
        .dequantize()
        .permute([2, 0, 1, 3]); // reverse permutation

    tensor.into_data().assert_approx_eq::<FloatElem>(
        &output.into_data(),
        Tolerance::absolute(1e-1).set_relative(1e-2),
    );
}

#[test]
fn should_quantize_symmetric_two_level_f16_block_scales() {
    let device = Default::default();

    let input: TestTensor<2> = TestTensorInt::arange(0..256, &device)
        .float()
        .div_scalar(256.)
        .reshape([16, 16]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_levels([
            BlockScale::per_block([2, 16], ScaleDtype::F16),
            BlockScale::per_tensor(ScaleDtype::F32),
        ]);

    let quantized = input.clone().quantize_dynamic(&scheme);
    let direct = quantized.clone().dequantize().into_data();

    let reloaded = TestTensor::<2>::from_data(quantized.into_data(), &device);
    let round_tripped = reloaded.dequantize().into_data();

    round_tripped.assert_eq(&direct, true);
    direct.assert_approx_eq(&input.into_data(), Tolerance::<f32>::rel_abs(1e-2, 1e-2));
}

/// A weight quantized along its first axis — the transposed view of one quantized along its
/// last, which is how a matmul's right-hand side wants its blocks to run — saves and loads
/// bit-identical. Each packed word holds values of one column, and the bytes keep them together.
#[test]
fn should_round_trip_packed_along_the_first_axis_through_bytes() {
    let device = Default::default();

    let input: TestTensor<2> = TestTensorInt::arange(0..512, &device)
        .float()
        .div_scalar(512.)
        .sub_scalar(0.5)
        .reshape([32, 16]);

    let scheme = device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q4S)
        .with_store(QuantStore::PackedU32(0))
        .with_levels([
            BlockScale::per_block([16], ScaleDtype::F16),
            BlockScale::per_tensor(ScaleDtype::F32),
        ]);
    // A backend that cannot hold this packed scheme has no packed axis to keep.
    if !device.supports_dtype(DType::QFloat(scheme)) {
        return;
    }

    let quantized = input
        .clone()
        .swap_dims(0, 1)
        .quantize_dynamic(&scheme)
        .swap_dims(0, 1);
    let DType::QFloat(packed) = quantized.dtype() else {
        unreachable!()
    };
    assert_eq!(packed.store, QuantStore::PackedU32(1));

    let direct = quantized.clone().dequantize().into_data();
    let saved = quantized.into_data();
    let reloaded = TestTensor::<2>::from_data(saved.clone(), &device);

    assert_eq!(reloaded.dtype(), DType::QFloat(packed));
    reloaded
        .clone()
        .dequantize()
        .into_data()
        .assert_eq(&direct, true);
    assert_eq!(reloaded.into_data().as_bytes(), saved.as_bytes());
    direct.assert_approx_eq(&input.into_data(), Tolerance::<f32>::rel_abs(1e-1, 1e-1));
}

fn per_block_32_scheme(device: &Device) -> QuantScheme {
    device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_levels([BlockScale::per_block([32], ScaleDtype::F32)])
}

#[test]
fn should_reshape_per_block_along_block_boundaries() {
    let device = Default::default();
    let tensor = TestTensorInt::arange(0..64, &device)
        .float()
        .div_scalar(64.);
    let expected = tensor.clone().reshape([2, 32]).into_data();

    let output = tensor
        .quantize_dynamic(&per_block_32_scheme(&device))
        .reshape([2, 32]);

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&expected, Tolerance::rel_abs(2e-2, 1e-2));
}

// A single [32] block cannot tile [2, 16]: it would have to span both rows.
#[test]
#[should_panic]
fn should_panic_when_reshape_splits_a_block_across_rows() {
    let device = Default::default();
    let tensor = TestTensorInt::arange(0..32, &device)
        .float()
        .div_scalar(32.);

    // Read back so a lazy backend executes the reshape.
    let _ = tensor
        .quantize_dynamic(&per_block_32_scheme(&device))
        .reshape([2, 16])
        .dequantize()
        .into_data();
}

#[test]
fn should_quantize_each_row_under_its_own_scale() {
    let device = Default::default();
    let row: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let rows: Vec<f32> = [1.0f32, 2.0, 0.5]
        .iter()
        .flat_map(|scale| row.iter().map(move |v| v * scale))
        .collect();
    let tensor = TestTensor::<2>::from_data(TensorData::new(rows, [3, 8]), &device);
    let qparams = QuantizationParameters {
        scales: TestTensor::from_data([1.0, 2.0, 0.5], &device),
        coarser: None,
    };

    let actual = tensor
        .clone()
        .quantize(&QuantScheme::fp8_per_row(), qparams)
        .dequantize();

    // Every value is its row's scale times an integer up to 8, which e4m3 holds exactly.
    actual
        .into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), Tolerance::default());
}

/// Blocks of `[2, 4]` over `[4, 8]` lay a `[2, 2]` grid, whose flat scales list its blocks row by
/// row. The scales are not powers of two, so a value read under another block's scale rounds.
#[test]
fn should_quantize_each_block_of_a_grid_under_its_own_scale() {
    let device = Default::default();
    let scales = [[1.0f32, 3.0], [5.0, 7.0]];
    let values: Vec<f32> = (0..4)
        .flat_map(|row| (0..8).map(move |col| (col + 1) as f32 * scales[row / 2][col / 4]))
        .collect();
    let tensor = TestTensor::<2>::from_data(TensorData::new(values, [4, 8]), &device);
    let qparams = QuantizationParameters {
        scales: TestTensor::from_data(
            TensorData::new(scales.as_flattened().to_vec(), [4]),
            &device,
        ),
        coarser: None,
    };
    let scheme = QuantScheme::default()
        .with_value(QuantValue::E4M3)
        .with_store(QuantStore::Native)
        .with_levels([BlockScale::per_block([2, 4], ScaleDtype::F32)]);

    let actual = tensor.clone().quantize(&scheme, qparams).dequantize();

    // Every value is its block's scale times an integer up to 8, which e4m3 holds exactly.
    actual
        .into_data()
        .assert_approx_eq::<FloatElem>(&tensor.into_data(), Tolerance::default());
}

/// Q8 blocks of four under one coarser scale a row, over rows whose magnitudes differ by orders.
fn row_factored(device: &Device) -> QuantScheme {
    device
        .settings()
        .quantization
        .scheme
        .with_value(QuantValue::Q8S)
        .with_levels([
            BlockScale::per_block([4], ScaleDtype::F32),
            BlockScale::per_block(
                BlockSize::with_rest(&[BlockExtent::WholeAxis], BlockExtent::VALUE),
                ScaleDtype::F32,
            ),
        ])
}

/// Four rows of eight, the second a thousand times the first, each block of four its own step.
fn rows_of_magnitudes(device: &Device) -> TestTensor<2> {
    let values: Vec<f32> = (0..32)
        .map(|i| {
            let (row, col) = (i / 8, i % 8);
            [0.01f32, 10.0, 0.5, 3.0][row] * (col as f32 - 3.5) * (1.0 + (col / 4) as f32)
        })
        .collect();
    TestTensor::<2>::from_data(TensorData::new(values, [4, 8]), device)
}

fn assert_round_trips(actual: TestTensor<2>, expected: TestTensor<2>) {
    actual.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::absolute(1e-4).set_relative(1e-2),
    );
}

#[test]
fn should_quantize_each_row_under_its_own_coarser_scale() {
    let device = Default::default();
    let input = rows_of_magnitudes(&device);

    let output = input.clone().quantize_dynamic(&row_factored(&device));

    assert_round_trips(output.dequantize(), input);
}

/// The coarser grid moves with the values: read along the wrong axis, a transposed row would
/// come back under another row's scale.
#[test]
fn should_transpose_a_tensor_under_a_coarser_scale_for_each_row() {
    let device = Default::default();
    let input = rows_of_magnitudes(&device);

    let output = input
        .clone()
        .quantize_dynamic(&row_factored(&device))
        .transpose();

    assert_round_trips(output.dequantize(), input.transpose());
}

#[test]
fn should_reshape_the_rows_of_a_tensor_under_a_coarser_scale_for_each_row() {
    let device = Default::default();
    let input = rows_of_magnitudes(&device);
    let stacked: TestTensor<3> = input.clone().reshape([2, 2, 8]);

    let output = stacked
        .quantize_dynamic(&row_factored(&device))
        .reshape([4, 8]);

    assert_round_trips(output.dequantize(), input);
}

#[test]
fn should_read_back_the_data_of_a_tensor_under_a_coarser_scale_for_each_row() {
    let device = Default::default();
    let input = rows_of_magnitudes(&device);
    let data = input
        .clone()
        .quantize_dynamic(&row_factored(&device))
        .into_data();

    let output = TestTensor::<2>::from_data(data, &device);

    assert_round_trips(output.dequantize(), input);
}

/// MXFP4 factored by row: `e2m1` codes under `ue8m0` block scales, those under one `f32` a row.
/// Each block holds a 6, so its scale is its row's times a power of two, the second block of a row
/// three below the first: every value is exact under the factored scales.
#[test]
fn should_round_trip_mxfp4_factored_by_row() {
    let device = Default::default();
    let codes = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let rows = [0.37f32, 5.0, 0.002, 1.0];
    let values: Vec<f32> = (0..rows.len() * 64)
        .map(|i| {
            let (row, col) = (i / 64, i % 64);
            let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
            let block = if col < 32 { 1.0 } else { 0.125 };
            sign * codes[col % 8] * block * rows[row]
        })
        .collect();
    let input = TestTensor::<2>::from_data(TensorData::new(values, [4, 64]), &device);

    let output = input
        .clone()
        .quantize_dynamic(&QuantScheme::mxfp4_row_factored());

    output
        .dequantize()
        .into_data()
        .assert_approx_eq::<FloatElem>(&input.into_data(), Tolerance::relative(1e-5));
}
