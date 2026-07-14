use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{
    Shape,
    quantization::{BlockSize, QuantLevel, QuantScheme, QuantStore, QuantValue},
};

fn should_quantize_dequantize_per_block_arange_reshaped<const D1: usize, const D2: usize>(
    level: QuantLevel,
    value: QuantValue,
    store: QuantStore,
    shape: [usize; D1],
    new_shape: [usize; D2],
) {
    let numel = Shape::from(shape).num_elements() as i64;

    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let scheme = QuantScheme::default()
        .with_level(level)
        .with_value(value)
        .with_store(store);

    let data = TestTensorInt::arange(0..numel, &ref_device)
        .float()
        .div_scalar(numel)
        .reshape::<D1, _>(shape)
        .into_data();

    let input_ref = TestTensor::<D1>::from_data(data.clone(), &device).reshape::<D2, _>(new_shape);
    let input = TestTensor::<D1>::from_data(data.clone(), &device)
        .quantize_dynamic(&scheme)
        .reshape::<D2, _>(new_shape);

    let output_ref = input_ref.into_data();
    let output = input.dequantize().into_data();

    output.assert_approx_eq::<FloatElem>(&output_ref, Tolerance::permissive());
}

#[test]
// https://github.com/tracel-ai/burn/issues/4659
// Edge case where a single block is used, essentially like `QuantLevel::Tensor`
fn should_quantize_dequantize_per_block_reshaped_global_block_q8s_packed() {
    should_quantize_dequantize_per_block_arange_reshaped(
        QuantLevel::Block(BlockSize::new([16])),
        QuantValue::Q8S,
        QuantStore::PackedU32(0),
        [32],
        [2, 16],
    )
}

#[test]
// FIXME: should work like tensor-level
#[should_panic] // "Reshape with sub-byte values is not supported"] error is shadowed by the CallError
fn should_quantize_dequantize_per_block_reshaped_global_block_q4s_packed() {
    should_quantize_dequantize_per_block_arange_reshaped(
        QuantLevel::Block(BlockSize::new([16])),
        QuantValue::Q4S,
        QuantStore::PackedU32(0),
        [32],
        [2, 16],
    )
}

#[test]
// FIXME: should work
#[should_panic] // "Reshape with sub-byte values is not supported" error is shadowed by the CallError
fn should_quantize_dequantize_per_tensor_reshaped_q4s_packed() {
    should_quantize_dequantize_per_block_arange_reshaped(
        QuantLevel::Tensor,
        QuantValue::Q4S,
        QuantStore::PackedU32(0),
        [32],
        [2, 16],
    )
}

#[test]
fn should_quantize_dequantize_per_block_reshaped_1d_q8s_native() {
    if supports_native() {
        should_quantize_dequantize_per_block_arange_reshaped(
            QuantLevel::Block(BlockSize::new([16])),
            QuantValue::Q8S,
            QuantStore::Native,
            [32],
            [2, 16],
        )
    }
}

#[test]
fn should_quantize_dequantize_per_block_unsqueezed_q8s_packed() {
    should_quantize_dequantize_per_block_arange_reshaped(
        QuantLevel::Block(BlockSize::new([32])),
        QuantValue::Q8S,
        QuantStore::PackedU32(0),
        [32],
        [1, 1, 1, 32],
    )
}

#[test]
#[should_panic] // "Reshape of ND block-quantized tensor is not yet supported" error is shadowed by the CallError
fn quantize_2d_block_reshape_should_panic() {
    should_quantize_dequantize_per_block_arange_reshaped(
        QuantLevel::Block(BlockSize::new([2, 4])),
        QuantValue::Q8S,
        QuantStore::PackedU32(0),
        [4, 8],
        [32], // invalid shape for 2D block boundaries
    )
}

#[test]
#[should_panic] // "Reshape would split a block across multiple rows" error is shadowed by the CallError
fn quantize_per_block_reshaped_should_not_split_block() {
    if supports_native() {
        should_quantize_dequantize_per_block_arange_reshaped(
            QuantLevel::Block(BlockSize::new([32])),
            QuantValue::Q8S,
            QuantStore::Native,
            [2, 32],
            [4, 16],
        )
    } else {
        // So it also panics with the same message when `QuantStore::Native` is not supported
        panic!("Reshape would split a block across multiple rows")
    }
}

#[test]
#[should_panic] // "Reshape would split a block across multiple rows"] error is shadowed by the CallError
fn should_quantize_dequantize_per_block_reshaped_2d_q8s_packed() {
    should_quantize_dequantize_per_block_arange_reshaped(
        QuantLevel::Block(BlockSize::new([32])),
        QuantValue::Q8S,
        QuantStore::PackedU32(0),
        [2, 32],
        [4, 16],
    )
}

// TODO: add tests for
// - ND reshape split (validation should panic)
// - broadcasted
// - multi-block successful reshape (all current success tests use exactly 1 block, e.g. [4, 32] -> [1, 4, 32] with block_size 32)
// - packed dimension alignment failure (invalid shape for packed num_quants)
// - ND-block unsqueeze (should succeed per the is_unsqueeze exemption, but nothing tests it)
