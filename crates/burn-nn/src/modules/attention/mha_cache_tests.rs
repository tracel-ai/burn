use super::*;
use burn::module::{ModuleMapper, Param};
use burn::tensor::{DType, TensorData, Tolerance};
use rstest::rstest;

struct CastParameters(DType);

impl ModuleMapper for CastParameters {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        param.map(|tensor| tensor.cast(self.0))
    }
}

fn tokens(batch: usize, length: usize, features: usize, dtype: DType) -> Tensor<3> {
    let values = (0..batch * length * features)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) / 11.0)
        .collect::<alloc::vec::Vec<_>>();
    Tensor::from_data(
        TensorData::new(values, [batch, length, features]),
        (&crate::test_device(), dtype),
    )
}

fn prefix(tensor: &Tensor<3>, length: usize) -> Tensor<3> {
    let [batch, _, features] = tensor.dims();
    tensor.clone().slice([0..batch, 0..length, 0..features])
}

fn assert_context_eq(actual: Tensor<3>, expected: Tensor<3>, dtype: DType) {
    assert_eq!(actual.dtype(), dtype);
    assert_eq!(actual.dims(), expected.dims());
    let tolerance = match dtype {
        DType::F16 => Tolerance::rel_abs(2e-3, 2e-3),
        DType::BF16 => Tolerance::rel_abs(2e-2, 2e-2),
        _ => Tolerance::rel_abs(1e-5, 1e-6),
    };
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), tolerance);
}

#[rstest]
#[case(DType::F32)]
#[case(DType::F16)]
#[case(DType::BF16)]
fn cached_context_matches_full_attention_after_prefill(#[case] dtype: DType) {
    let device = crate::test_device();
    let mha = MultiHeadAttentionConfig::new(4, 2)
        .with_dropout(0.5)
        .init(&device)
        .map(&mut CastParameters(dtype));
    let tensor = tokens(2, 6, 4, dtype);
    let mut reference_cache = MhaCache::autoregressive();
    let mut context_cache = MhaCache::autoregressive();
    let mut prefill = None;

    for length in 3..=6 {
        let input = MhaInput::self_attn(prefix(&tensor, length));
        let expected = mha
            .forward_cache(input.clone(), &mut reference_cache)
            .context;
        let actual = mha.forward_cache_context(input, &mut context_cache);
        if length == 3 {
            prefill = Some((actual.clone(), actual.to_data()));
        }
        assert_context_eq(actual, expected, dtype);
    }

    // Appending later contexts must not overwrite an earlier returned tensor.
    let (prefill, data) = prefill.unwrap();
    prefill.into_data().assert_eq(&data, true);
}

#[rstest]
#[case(false)]
#[case(true)]
fn cached_context_uses_latest_mask_row(#[case] quiet_softmax: bool) {
    let device = crate::test_device();
    let mha = MultiHeadAttentionConfig::new(4, 2)
        .with_dropout(0.0)
        .with_quiet_softmax(quiet_softmax)
        .init(&device);
    let tensor = tokens(2, 5, 4, DType::F32);
    let mut reference_cache = MhaCache::autoregressive();
    let mut context_cache = MhaCache::autoregressive();

    for length in 2..=5 {
        let mut masks = alloc::vec::Vec::new();
        for batch in 0..2 {
            for query in 0..length {
                for key in 0..length {
                    masks.push(
                        query == length - 1 && length == 4 || key == (query + batch) % length,
                    );
                }
            }
        }
        let mut padding = alloc::vec![false; 2 * length];
        padding[length] = true;
        let input = MhaInput::self_attn(prefix(&tensor, length))
            .mask_attn(Tensor::from_data(
                TensorData::new(masks, [2, length, length]),
                &device,
            ))
            .mask_pad(Tensor::from_data(
                TensorData::new(padding, [2, length]),
                &device,
            ));
        let expected = mha
            .forward_cache(input.clone(), &mut reference_cache)
            .context;
        let actual = mha.forward_cache_context(input, &mut context_cache);
        if length == 4 {
            // The newest attention rows are fully masked, leaving only output bias.
            let bias = mha.output.bias.as_ref().unwrap().val();
            let expected = bias.reshape([1, 1, 4]).expand([2, 1, 4]);
            assert_context_eq(
                actual.clone().slice([0..2, 3..4, 0..4]),
                expected,
                DType::F32,
            );
        }
        assert_context_eq(actual, expected, DType::F32);
    }
}

#[test]
fn cached_cross_attention_context_uses_fixed_memory_and_broadcast_mask() {
    let device = crate::test_device();
    let mha = MultiHeadAttentionConfig::new(4, 2)
        .with_dropout(0.0)
        .init(&device);
    let query = tokens(2, 5, 4, DType::F32);
    let key = tokens(2, 3, 4, DType::F32);
    let value = key.clone().mul_scalar(-2.0);
    let mask =
        Tensor::<3, Bool>::from_data([[[false, true, false]], [[true, false, false]]], &device);
    let mut reference_cache = MhaCache::autoregressive_cross_attention();
    let mut context_cache = MhaCache::autoregressive_cross_attention();

    for length in 2..=5 {
        // Changes to the supplied memory after prefill are ignored by both caches.
        let input = MhaInput::new(
            prefix(&query, length),
            key.clone().add_scalar(length as f32),
            value.clone().sub_scalar(length as f32),
        )
        .mask_attn(mask.clone());
        let expected = mha
            .forward_cache(input.clone(), &mut reference_cache)
            .context;
        let actual = mha.forward_cache_context(input, &mut context_cache);
        assert_context_eq(actual, expected, DType::F32);
    }
}

#[test]
fn cached_attention_methods_can_alternate() {
    let device = crate::test_device();
    let mha = MultiHeadAttentionConfig::new(4, 2)
        .with_dropout(0.0)
        .init(&device);
    let tensor = tokens(1, 7, 4, DType::F32);
    let mut reference_cache = MhaCache::autoregressive();
    let mut mixed_cache = MhaCache::autoregressive();

    for length in 3..=7 {
        let input = MhaInput::self_attn(prefix(&tensor, length));
        let expected = mha.forward_cache(input.clone(), &mut reference_cache);
        let actual = if length % 2 == 0 {
            let actual = mha.forward_cache(input, &mut mixed_cache);
            assert_eq!(actual.weights.dims(), [1, 2, length, length]);
            actual
                .weights
                .into_data()
                .assert_eq(&expected.weights.into_data(), true);
            actual.context
        } else {
            mha.forward_cache_context(input, &mut mixed_cache)
        };
        assert_context_eq(actual, expected.context, DType::F32);
    }
}

#[test]
#[should_panic(expected = "axis 1 expected 3, got 2")]
fn cached_context_rejects_mask_with_wrong_query_length() {
    let device = crate::test_device();
    let mha = MultiHeadAttentionConfig::new(4, 2).init(&device);
    let tensor = tokens(1, 3, 4, DType::F32);
    let mut cache = MhaCache::autoregressive();
    let _ = mha.forward_cache_context(MhaInput::self_attn(prefix(&tensor, 2)), &mut cache);

    let input = MhaInput::self_attn(tensor).mask_attn(Tensor::<3, Bool>::zeros([1, 2, 3], &device));
    let _ = mha.forward_cache_context(input, &mut cache);
}

#[test]
#[should_panic(expected = "axis 2 expected 4, got 3")]
fn cached_context_rejects_skipped_tokens() {
    let device = crate::test_device();
    let mha = MultiHeadAttentionConfig::new(4, 2).init(&device);
    let tensor = tokens(1, 4, 4, DType::F32);
    let mut cache = MhaCache::autoregressive();
    let _ = mha.forward_cache_context(MhaInput::self_attn(prefix(&tensor, 2)), &mut cache);
    let _ = mha.forward_cache_context(MhaInput::self_attn(tensor), &mut cache);
}
