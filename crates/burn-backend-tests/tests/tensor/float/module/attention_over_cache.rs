use super::*;
use burn_tensor::Distribution;
use burn_tensor::TensorData;
use burn_tensor::Tolerance;
use burn_tensor::module::{attention_fallback, attention_over_cache};
use burn_tensor::ops::{AttentionModuleOptions, CausalAlignment};
use burn_tensor::s;

/// A query and a cache of `capacity` positions per sequence, each sequence attending its own
/// length of them.
struct CacheCall {
    query: TestTensor<4>,
    key: TestTensor<4>,
    value: TestTensor<4>,
    lengths: Vec<usize>,
}

impl CacheCall {
    const HEAD_DIM: usize = 32;

    fn random(
        heads: usize,
        kv_heads: usize,
        seq_q: usize,
        capacity: usize,
        lengths: &[usize],
    ) -> Self {
        let batch = lengths.len();
        let operand = |heads: usize, seq: usize| {
            TestTensor::<4>::random(
                [batch, heads, seq, Self::HEAD_DIM],
                Distribution::Uniform(-1., 1.),
                &Default::default(),
            )
        };
        Self {
            query: operand(heads, seq_q),
            key: operand(kv_heads, capacity),
            value: operand(kv_heads, capacity),
            lengths: lengths.to_vec(),
        }
    }

    fn lengths(&self) -> TestTensorInt<1> {
        let lengths: Vec<i32> = self.lengths.iter().map(|&len| len as i32).collect();
        let count = lengths.len();
        TestTensorInt::<1>::from_data(TensorData::new(lengths, [count]), &Default::default())
    }

    fn attend(&self, options: AttentionModuleOptions) -> TestTensor<4> {
        attention_over_cache(
            self.query.clone(),
            self.key.clone(),
            self.value.clone(),
            None,
            self.lengths(),
            options,
        )
    }

    /// The composed attention of each sequence over its attended keys alone.
    fn expected(&self, options: AttentionModuleOptions) -> TestTensor<4> {
        let sequences = self
            .lengths
            .iter()
            .enumerate()
            .map(|(b, &len)| {
                attention_fallback(
                    self.query.clone().slice(s![b..b + 1]),
                    self.key.clone().slice(s![b..b + 1, .., 0..len]),
                    self.value.clone().slice(s![b..b + 1, .., 0..len]),
                    None,
                    None,
                    options,
                )
            })
            .collect();
        TestTensor::cat(sequences, 0)
    }

    /// The cache as pools of `page_size`-position pages, stored in reverse order so a page's
    /// index is never its position, and the table naming each sequence's pages.
    fn paged(&self, page_size: usize) -> (TestTensor<4>, TestTensor<4>, TestTensorInt<2>) {
        let [batch, kv_heads, capacity, dim] = self.key.dims();
        let per_sequence = capacity / page_size;
        let num_pages = batch * per_sequence;
        let reversed =
            TestTensorInt::<1>::arange(0..num_pages as i64, &Default::default()).flip([0]);
        let pool = |cache: &TestTensor<4>| {
            cache
                .clone()
                .reshape([batch, kv_heads, per_sequence, page_size, dim])
                .permute([0, 2, 1, 3, 4])
                .reshape([num_pages, kv_heads, page_size, dim])
                .select(0, reversed.clone())
        };
        let (key, value) = (pool(&self.key), pool(&self.value));
        (key, value, reversed.reshape([batch, per_sequence]))
    }

    fn assert_matches(output: TestTensor<4>, expected: TestTensor<4>) {
        output.into_data().assert_approx_eq::<FloatElem>(
            &expected.into_data(),
            Tolerance::rel_abs(1e-2, 1e-3).set_half_precision_relative(1e-1),
        );
    }
}

#[test]
fn test_attention_over_cache_prompt_chunk_per_sequence_lengths() {
    // A causal chunk of 4 queries over a grouped cache of 32, the sequences 9 and 23 long: each
    // query sees its own sequence's keys up to its own position, none past the length.
    let call = CacheCall::random(4, 2, 4, 32, &[9, 23]);
    let options = AttentionModuleOptions {
        is_causal: true,
        ..Default::default()
    };

    CacheCall::assert_matches(call.attend(options), call.expected(options));
}

#[test]
fn test_attention_over_cache_windowed_decode_twice() {
    // One query per sequence, a window of 5: a second call reads the same arrival counters as the
    // first, which it only can if the first left them at zero.
    let call = CacheCall::random(4, 1, 1, 64, &[17, 40, 3]);
    let options = AttentionModuleOptions {
        sliding_window: Some(5),
        ..Default::default()
    };
    let expected = call.expected(options);

    CacheCall::assert_matches(call.attend(options), expected.clone());
    CacheCall::assert_matches(call.attend(options), expected);
}

#[test]
fn test_attention_over_cache_paged_matches_contiguous() {
    let call = CacheCall::random(2, 2, 3, 64, &[20, 64]);
    let options = AttentionModuleOptions {
        is_causal: true,
        ..Default::default()
    };
    let (key, value, pages) = call.paged(16);

    let paged = attention_over_cache(
        call.query.clone(),
        key,
        value,
        Some(pages),
        call.lengths(),
        options,
    );

    CacheCall::assert_matches(paged, call.attend(options));
}

#[test]
#[should_panic(expected = "last attended positions")]
fn test_attention_over_cache_refuses_top_left() {
    let call = CacheCall::random(1, 1, 2, 8, &[4]);
    call.attend(AttentionModuleOptions {
        is_causal: true,
        causal_alignment: CausalAlignment::TopLeft,
        ..Default::default()
    });
}
