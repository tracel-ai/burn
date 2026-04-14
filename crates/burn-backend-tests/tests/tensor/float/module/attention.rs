use super::*;
use burn_tensor::Distribution;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::module::attention;
use burn_tensor::module::attention_fallback;
use burn_tensor::ops::AttentionModuleOptions;
use num_traits::{Signed, cast::cast};

#[test]
fn test_attention_no_mask() {
    // Skip on metal with f16 - flash attention returns zeros
    // Enable once this issue is fixed: https://github.com/tracel-ai/burn/issues/4325
    #[cfg(feature = "metal")]
    if core::any::TypeId::of::<FloatElem>() == core::any::TypeId::of::<burn_tensor::f16>() {
        return;
    }

    let num_batches = 1;
    let num_heads = 1;
    let seq_q = 128;
    let seq_kv = 128;
    let head_dim = 64;
    let val_dim = 64;

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_q, head_dim],
        Distribution::Uniform(0., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_kv, head_dim],
        Distribution::Uniform(0., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_kv, val_dim],
        Distribution::Uniform(0., 1.),
        &Default::default(),
    );

    let output = attention(
        query.clone(),
        key.clone(),
        value.clone(),
        None,
        None,
        Default::default(),
    );

    let expected =
        attention_fallback::<TestBackend>(query, key, value, None, None, Default::default());

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::rel_abs(1e-2, 1e-3).set_half_precision_relative(1e-1),
    );
}

#[test]
fn test_attention_custom_scale() {
    let [num_batches, num_heads, seq_len, head_dim] = [1, 2, 16, 32];

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );

    let options = AttentionModuleOptions {
        scale: Some(0.1),
        ..Default::default()
    };

    let output = attention(
        query.clone(),
        key.clone(),
        value.clone(),
        None,
        None,
        options,
    );

    let expected = attention_fallback::<TestBackend>(query, key, value, None, None, options);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::rel_abs(1e-2, 1e-3).set_half_precision_relative(1e-1),
    );
}

#[test]
fn test_attention_attn_bias() {
    let [num_batches, num_heads, seq_len, head_dim] = [1, 2, 16, 32];

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let bias = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, seq_len],
        Distribution::Uniform(-0.5, 0.5),
        &Default::default(),
    );

    let output = attention(
        query.clone(),
        key.clone(),
        value.clone(),
        None,
        Some(bias.clone()),
        Default::default(),
    );

    let expected =
        attention_fallback::<TestBackend>(query, key, value, None, Some(bias), Default::default());

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::rel_abs(1e-2, 1e-3).set_half_precision_relative(1e-1),
    );
}

#[test]
fn test_attention_softcap() {
    let [num_batches, num_heads, seq_len, head_dim] = [1, 2, 16, 32];

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );

    let options = AttentionModuleOptions {
        softcap: Some(50.0),
        ..Default::default()
    };

    let output = attention(
        query.clone(),
        key.clone(),
        value.clone(),
        None,
        None,
        options,
    );

    let expected = attention_fallback::<TestBackend>(query, key, value, None, None, options);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::rel_abs(1e-2, 1e-3).set_half_precision_relative(1e-1),
    );
}

#[test]
fn test_attention_is_causal() {
    let [num_batches, num_heads, seq_len, head_dim] = [2, 4, 16, 32];

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );

    let options = AttentionModuleOptions {
        is_causal: true,
        ..Default::default()
    };

    let output = attention(
        query.clone(),
        key.clone(),
        value.clone(),
        None,
        None,
        options,
    );

    let expected = attention_fallback::<TestBackend>(query, key, value, None, None, options);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::rel_abs(1e-2, 1e-3).set_half_precision_relative(1e-1),
    );
}

/// Cross-attention: seq_q != seq_k, with causal masking and additive bias.
#[test]
fn test_attention_cross_attention_with_bias() {
    let [num_batches, num_heads, seq_q, seq_k, head_dim] = [2, 2, 8, 24, 32];

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_q, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_k, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_k, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let bias = TestTensor::<4>::random(
        [num_batches, num_heads, seq_q, seq_k],
        Distribution::Uniform(-0.5, 0.5),
        &Default::default(),
    );

    let options = AttentionModuleOptions {
        is_causal: true,
        ..Default::default()
    };

    let output = attention(
        query.clone(),
        key.clone(),
        value.clone(),
        None,
        Some(bias.clone()),
        options,
    );

    let expected = attention_fallback::<TestBackend>(query, key, value, None, Some(bias), options);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::rel_abs(1e-2, 1e-3).set_half_precision_relative(1e-1),
    );
}

/// Regression: softcap must be applied before -inf masking.
/// With causal masking, position 0 can only attend to itself, so output[0] == value[0].
/// If softcap were applied after masking, tanh(-inf/softcap) = -softcap (finite),
/// and the masked position would leak into the output.
#[test]
fn test_attention_softcap_preserves_causal_mask() {
    let [num_batches, num_heads, seq_len, head_dim] = [1, 1, 4, 8];

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );

    let options = AttentionModuleOptions {
        softcap: Some(20.0),
        is_causal: true,
        ..Default::default()
    };

    let output = attention_fallback::<TestBackend>(query, key, value.clone(), None, None, options);

    // With causal masking, position 0 can only attend to itself (softmax = [1, 0, 0, 0]).
    // So output[..., 0, :] must equal value[..., 0, :].
    let output_row0 = output.slice([0..1, 0..1, 0..1, 0..head_dim]);
    let value_row0 = value.slice([0..1, 0..1, 0..1, 0..head_dim]);

    output_row0
        .into_data()
        .assert_approx_eq::<FloatElem>(&value_row0.into_data(), Tolerance::rel_abs(1e-4, 1e-4));
}

/// Regression: fully-masked rows must produce 0, not NaN.
/// When a bool mask masks every key position for a query row, all attention
/// scores are -inf and naive softmax yields NaN.
#[test]
fn test_attention_fully_masked_rows_no_nan() {
    // Skip test with f16 (fallback uses too big epsilon value)
    if core::any::TypeId::of::<FloatElem>() == core::any::TypeId::of::<burn_tensor::f16>() {
        return;
    }
    let [num_batches, num_heads, seq_len, head_dim] = [1, 1, 4, 8];

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );

    let mask = TestTensorBool::<4>::full(
        [num_batches, num_heads, seq_len, seq_len],
        true,
        &Default::default(),
    );

    let output =
        attention_fallback::<TestBackend>(query, key, value, Some(mask), None, Default::default());

    let output_data = output.into_data();
    let values = output_data.as_slice::<FloatElem>().unwrap();
    let tol: FloatElem = cast(1e-4f64).unwrap();
    assert!(
        !values.iter().any(|v| v.is_nan()),
        "Fully-masked rows should produce 0, not NaN"
    );
    assert!(
        values.iter().all(|v| v.abs() < tol),
        "Fully-masked rows should produce values near 0"
    );
}

/// Causal + partial bool mask combine to fully mask early rows.
/// With seq_len=4, the causal mask allows row 0 to attend only to key 0.
/// The bool mask masks key 0, so row 0 is fully masked while rows 1-3 still
/// have valid positions. Row 0 output must be 0, not NaN.
#[test]
fn test_attention_fully_masked_rows_causal_no_nan() {
    // Skip test with f16 (fallback uses too big epsilon value)
    if core::any::TypeId::of::<FloatElem>() == core::any::TypeId::of::<burn_tensor::f16>() {
        return;
    }

    let [num_batches, num_heads, seq_len, head_dim] = [1, 1, 4, 8];

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );

    // Mask only column 0 (key position 0). Combined with causal mask,
    // row 0 becomes fully masked since it can only attend to key 0.
    #[rustfmt::skip]
    let mask_data = TensorData::from([[[[
        true,  false, false, false,
        true,  false, false, false,
        true,  false, false, false,
        true,  false, false, false,
    ]]]]);
    let mask = TestTensorBool::<4>::from_data(mask_data, &Default::default()).reshape([
        num_batches,
        num_heads,
        seq_len,
        seq_len,
    ]);

    let options = AttentionModuleOptions {
        is_causal: true,
        ..Default::default()
    };

    let output = attention_fallback::<TestBackend>(query, key, value, Some(mask), None, options);

    let output_data = output.into_data();
    let values = output_data.as_slice::<FloatElem>().unwrap();
    let tol: FloatElem = cast(1e-4f64).unwrap();
    assert!(
        !values.iter().any(|v| v.is_nan()),
        "Fully-masked rows should produce 0, not NaN"
    );
    // Row 0 (indices 0..head_dim) should be ~0 since it's fully masked
    let row0 = &values[..head_dim];
    assert!(
        row0.iter().all(|v| v.abs() < tol),
        "Fully-masked row 0 should produce values near 0"
    );
    // Rows 1-3 should have non-zero output (they have valid positions)
    let rest = &values[head_dim..];
    assert!(
        rest.iter().any(|v| v.abs() > tol),
        "Non-masked rows should produce non-zero output"
    );
}

/// Combined: mask + bias + custom scale + softcap together.
#[test]
fn test_attention_all_options() {
    let [num_batches, num_heads, seq_len, head_dim] = [2, 2, 16, 32];

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let bias = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, seq_len],
        Distribution::Uniform(-0.5, 0.5),
        &Default::default(),
    );
    // Create a random bool mask by thresholding a uniform float tensor
    let mask = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, seq_len],
        Distribution::Uniform(0., 1.),
        &Default::default(),
    )
    .greater_elem(0.7);

    let options = AttentionModuleOptions {
        scale: Some(0.05),
        softcap: Some(30.0),
        is_causal: true,
    };

    let output = attention(
        query.clone(),
        key.clone(),
        value.clone(),
        Some(mask.clone()),
        Some(bias.clone()),
        options,
    );

    let expected =
        attention_fallback::<TestBackend>(query, key, value, Some(mask), Some(bias), options);

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::rel_abs(1e-2, 1e-3).set_half_precision_relative(1e-1),
    );
}

/// Regression for burn#4772: ONNX Attention-23 allows a `[1, 1, seq_q, seq_kv]` bias
/// that's shared across all batches and heads. The main attention path must accept this
/// shape and produce the same result as the fallback (whose elementwise `float_add`
/// broadcasts the bias naturally).
#[test]
fn test_attention_bias_broadcast_batch_and_heads() {
    let [num_batches, num_heads, seq_len, head_dim] = [2, 3, 16, 32];

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let bias = TestTensor::<4>::random(
        [1, 1, seq_len, seq_len],
        Distribution::Uniform(-0.5, 0.5),
        &Default::default(),
    );

    let output = attention(
        query.clone(),
        key.clone(),
        value.clone(),
        None,
        Some(bias.clone()),
        Default::default(),
    );

    let expected =
        attention_fallback::<TestBackend>(query, key, value, None, Some(bias), Default::default());

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::rel_abs(1e-2, 1e-3).set_half_precision_relative(1e-1),
    );
}

/// Regression for burn#4772: `[batch, 1, seq_q, seq_kv]` bias is shared across heads
/// but distinct per batch (the ONNX `test_attention_4d_attn_mask_3d` pattern).
#[test]
fn test_attention_bias_broadcast_heads_only() {
    let [num_batches, num_heads, seq_len, head_dim] = [2, 3, 16, 32];

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let bias = TestTensor::<4>::random(
        [num_batches, 1, seq_len, seq_len],
        Distribution::Uniform(-0.5, 0.5),
        &Default::default(),
    );

    let output = attention(
        query.clone(),
        key.clone(),
        value.clone(),
        None,
        Some(bias.clone()),
        Default::default(),
    );

    let expected =
        attention_fallback::<TestBackend>(query, key, value, None, Some(bias), Default::default());

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::rel_abs(1e-2, 1e-3).set_half_precision_relative(1e-1),
    );
}

/// Regression for burn#4772: a `[1, 1, seq_q, seq_kv]` bool mask must be shared across
/// all batches and heads (the ONNX `test_attention_4d_attn_mask_bool` pattern).
#[test]
fn test_attention_bool_mask_broadcast_batch_and_heads() {
    // Skip on cubecl backends with f16: the main attention path diverges from
    // attention_fallback starting at batch 0 head 1, the fingerprint of a broadcast
    // mask being applied only to head 0. The equivalent full-shape mask case
    // (test_attention_all_options) passes on the same backends, so this is a latent
    // cubecl flash-attention broadcast issue (tracked in #4778, under the umbrella
    // of #4325), not a burn-flex or fallback bug. Enable once #4778 is fixed.
    #[cfg(feature = "cube")]
    if core::any::TypeId::of::<FloatElem>() == core::any::TypeId::of::<burn_tensor::f16>() {
        return;
    }

    let [num_batches, num_heads, seq_len, head_dim] = [2, 3, 16, 32];

    let query = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let key = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    let value = TestTensor::<4>::random(
        [num_batches, num_heads, seq_len, head_dim],
        Distribution::Uniform(-1., 1.),
        &Default::default(),
    );
    // Random bool mask via thresholding, matching test_attention_all_options.
    let mask = TestTensor::<4>::random(
        [1, 1, seq_len, seq_len],
        Distribution::Uniform(0., 1.),
        &Default::default(),
    )
    .greater_elem(0.7);

    let output = attention(
        query.clone(),
        key.clone(),
        value.clone(),
        Some(mask.clone()),
        None,
        Default::default(),
    );

    let expected =
        attention_fallback::<TestBackend>(query, key, value, Some(mask), None, Default::default());

    output.into_data().assert_approx_eq::<FloatElem>(
        &expected.into_data(),
        Tolerance::rel_abs(1e-2, 1e-3).set_half_precision_relative(1e-1),
    );
}
