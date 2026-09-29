use super::*;
use burn_tensor::Tolerance;
use burn_tensor::{
    Shape,
    quantization::{BlockSize, QuantScheme, QuantStore, QuantValue, ScaleDtype},
};

fn scheme_for(block: Option<BlockSize>, value: QuantValue, store: QuantStore) -> QuantScheme {
    let scheme = QuantScheme::default().with_value(value).with_store(store);
    match block {
        Some(block) => scheme.per_block(block.as_slice(), ScaleDtype::F32),
        None => scheme,
    }
}

fn should_quantize_dequantize_per_block_arange_reshaped<const D1: usize, const D2: usize>(
    block: Option<BlockSize>,
    value: QuantValue,
    store: QuantStore,
    shape: [usize; D1],
    new_shape: [usize; D2],
) {
    let numel = Shape::from(shape).num_elements() as i64;

    let device = Default::default();
    let ref_device = ReferenceDevice::new();

    let scheme = scheme_for(block, value, store);

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
// Edge case where a single block is used, essentially like per-tensor quantization
fn should_quantize_dequantize_per_block_reshaped_global_block_q8s_packed() {
    should_quantize_dequantize_per_block_arange_reshaped(
        Some(BlockSize::new([16])),
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
        Some(BlockSize::new([16])),
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
        None,
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
            Some(BlockSize::new([16])),
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
        Some(BlockSize::new([32])),
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
        Some(BlockSize::new([2, 4])),
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
            Some(BlockSize::new([32])),
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
        Some(BlockSize::new([32])),
        QuantValue::Q8S,
        QuantStore::PackedU32(0),
        [2, 32],
        [4, 16],
    )
}

/// Compare a view applied to a quantized tensor with the same view applied after dequantization.
/// Both paths reconstruct from the same quantized values and scales, so they must be bit-exact.
fn assert_layout_matches_dequantize<const D: usize>(
    layout_quantized: TestTensor<D>,
    layout_dequantized: TestTensor<D>,
) {
    layout_quantized
        .dequantize()
        .into_data()
        .assert_eq(&layout_dequantized.into_data(), true);
}

fn reshape_matches_dequantize_then_reshape<const D1: usize, const D2: usize>(
    value: QuantValue,
    block: Option<BlockSize>,
    shape: [usize; D1],
    new_shape: [usize; D2],
) {
    let numel = Shape::from(shape).num_elements() as i64;
    let device = Default::default();
    let ref_device = ReferenceDevice::new();
    let scheme = scheme_for(block, value, QuantStore::PackedU32(0));

    let data = TestTensorInt::arange(0..numel, &ref_device)
        .float()
        .div_scalar(numel)
        .reshape::<D1, _>(shape)
        .into_data();

    let q = TestTensor::<D1>::from_data(data, &device).quantize_dynamic(&scheme);

    assert_layout_matches_dequantize(
        q.clone().reshape::<D2, _>(new_shape),
        q.dequantize().reshape::<D2, _>(new_shape),
    );
}

#[test]
fn q8s_reshape_matches_dequantize_then_reshape() {
    reshape_matches_dequantize_then_reshape(QuantValue::Q8S, None, [32], [2, 16]);
}

#[test]
fn q4s_reshape_matches_dequantize_then_reshape() {
    reshape_matches_dequantize_then_reshape(QuantValue::Q4S, None, [32], [2, 16]);
}

#[test]
fn q4s_block_reshape_matches_dequantize_then_reshape() {
    reshape_matches_dequantize_then_reshape(
        QuantValue::Q4S,
        Some(BlockSize::new([16])),
        [32],
        [2, 16],
    );
}

#[test]
fn multi_block_reshape_matches_dequantize_then_reshape() {
    // Four independent blocks make sure the scale grid is reshaped along with the values.
    reshape_matches_dequantize_then_reshape(
        QuantValue::Q8S,
        Some(BlockSize::new([32])),
        [4, 32],
        [1, 4, 32],
    );
}

#[test]
fn nd_block_unsqueeze_matches_dequantize_then_reshape() {
    // Adding a leading unit dimension preserves every two-dimensional block boundary.
    reshape_matches_dequantize_then_reshape(
        QuantValue::Q8S,
        Some(BlockSize::new([2, 4])),
        [4, 8],
        [1, 4, 8],
    );
}

#[test]
fn broadcasted_reshape_matches_dequantize_then_reshape() {
    let shape = [2, 4, 8];
    let numel = Shape::from(shape).num_elements() as i64;
    let device = Default::default();
    let ref_device = ReferenceDevice::new();
    let scheme = scheme_for(None, QuantValue::Q8S, QuantStore::PackedU32(0));

    let data = TestTensorInt::arange(0..numel, &ref_device)
        .float()
        .div_scalar(numel)
        .reshape(shape)
        .into_data();

    // Permuting only batch dimensions keeps the packed dimension last while making the tensor
    // non-contiguous. Prepending a unit dimension then exercises ReshapeAnalysis::Broadcasted.
    let q = TestTensor::<3>::from_data(data, &device)
        .quantize_dynamic(&scheme)
        .permute([1, 0, 2]);

    assert_layout_matches_dequantize(
        q.clone().reshape::<4, _>([1, 4, 2, 8]),
        q.dequantize().reshape::<4, _>([1, 4, 2, 8]),
    );
}

#[test]
#[should_panic] // "Cannot reshape packed tensor" error is shadowed by the CallError
fn packed_dimension_alignment_failure_should_panic() {
    // Q8 packs four values per u32, but the new packed dimension has length six.
    reshape_matches_dequantize_then_reshape(QuantValue::Q8S, None, [24], [4, 6]);
}

#[test]
#[should_panic] // "Split reshape of ND block-quantized tensor" error is shadowed by the CallError
fn nd_block_split_reshape_should_panic() {
    let shape = [2, 4, 8];
    let numel = Shape::from(shape).num_elements() as i64;
    let device = Default::default();
    let ref_device = ReferenceDevice::new();
    let scheme = scheme_for(
        Some(BlockSize::new([1, 2, 4])),
        QuantValue::Q8S,
        QuantStore::PackedU32(0),
    );

    let data = TestTensorInt::arange(0..numel, &ref_device)
        .float()
        .div_scalar(numel)
        .reshape(shape)
        .into_data();

    // Make the quantized tensor non-contiguous, then split its first physical dimension while
    // retaining the packed trailing dimension. This reaches ReshapeAnalysis::Split.
    let _ = TestTensor::<3>::from_data(data, &device)
        .quantize_dynamic(&scheme)
        .permute([1, 0, 2])
        .reshape::<4, _>([2, 2, 2, 8])
        .dequantize()
        .into_data();
}
