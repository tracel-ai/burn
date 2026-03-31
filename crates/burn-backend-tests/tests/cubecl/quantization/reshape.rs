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
        QuantLevel::Block(BlockSize::new([32])),
        QuantValue::Q8S,
        QuantStore::PackedU32(0),
        [32],
        [2, 16],
    )
}

#[test]
#[ignore] // FIXME: should work like tensor-level
fn should_quantize_dequantize_per_block_reshaped_global_block_q4s_packed() {
    should_quantize_dequantize_per_block_arange_reshaped(
        QuantLevel::Block(BlockSize::new([32])),
        QuantValue::Q4S,
        QuantStore::PackedU32(0),
        [32],
        [2, 16],
    )
}

#[test]
#[ignore] // FIXME: should work
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
            QuantLevel::Block(BlockSize::new([32])),
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
#[should_panic = "Cannot reshape a block-quantized tensor when the reshape requires recomputing the buffer"]
fn quantize_2d_block_reshape_should_be_valid() {
    should_quantize_dequantize_per_block_arange_reshaped(
        QuantLevel::Block(BlockSize::new([2, 4])),
        QuantValue::Q8S,
        QuantStore::PackedU32(0),
        [4, 8],
        [32], // invalid shape for 2D block boundaries
    )
}

#[test]
#[should_panic = "Reshape would split a block across multiple rows"]
fn quantize_per_block_reshaped_should_not_split_block() {
    should_quantize_dequantize_per_block_arange_reshaped(
        QuantLevel::Block(BlockSize::new([32])),
        QuantValue::Q8S,
        QuantStore::Native,
        [2, 32],
        [4, 16],
    )
}

#[test]
#[should_panic = "Reshape would split a block across multiple rows"]
fn should_quantize_dequantize_per_block_reshaped_2d_q8s_packed() {
    should_quantize_dequantize_per_block_arange_reshaped(
        QuantLevel::Block(BlockSize::new([32])),
        QuantValue::Q8S,
        QuantStore::PackedU32(0),
        [2, 32],
        [4, 16],
    )
}

// TODO: add tests for ND reshape split (validation should panic) and broadcasted
