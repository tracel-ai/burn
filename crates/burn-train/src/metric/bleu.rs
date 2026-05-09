use super::state::{FormatOptions, NumericMetricState};
use super::{MetricMetadata, SerializedEntry};
use crate::metric::{Metric, MetricAttributes, MetricName, Numeric, NumericAttributes, NumericEntry};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{Int, Tensor};
use core::marker::PhantomData;
use std::collections::HashMap;
use std::sync::Arc;

/// Computes the BLEU (Bilingual Evaluation Understudy) score between predicted
/// and reference token sequences.
///
/// BLEU measures the quality of machine-translated text by comparing n-gram
/// overlap between the candidate (prediction) and reference sequences. The
/// score combines modified n-gram precision for n = 1..max_n with a brevity
/// penalty that discourages overly short translations.
///
/// The metric operates on integer token IDs (not raw text), matching the
/// convention used by [`CharErrorRate`](super::CharErrorRate) and
/// [`WordErrorRate`](super::WordErrorRate).
///
/// # References
///
/// Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine
/// Translation", ACL 2002.
#[derive(Clone)]
pub struct BleuScore<B: Backend> {
    name: MetricName,
    state: NumericMetricState,
    max_n: usize,
    pad_token: Option<usize>,
    _b: PhantomData<B>,
}

/// The [BLEU score metric](BleuScore) input type.
#[derive(new)]
pub struct BleuInput<B: Backend> {
    /// The predicted token sequences (2-D tensor of token indices).
    pub outputs: Tensor<B, 2, Int>,
    /// The reference token sequences (2-D tensor of token indices).
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Default for BleuScore<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> BleuScore<B> {
    /// Creates the metric with default max_n = 4 (BLEU-4).
    pub fn new() -> Self {
        Self {
            name: Arc::new("BLEU".to_string()),
            state: NumericMetricState::default(),
            max_n: 4,
            pad_token: None,
            _b: PhantomData,
        }
    }

    /// Sets the maximum n-gram order.
    ///
    /// Common values: 1 (BLEU-1), 2 (BLEU-2), 4 (BLEU-4, default).
    pub fn with_max_n(mut self, max_n: usize) -> Self {
        assert!(max_n >= 1, "max_n must be at least 1");
        self.max_n = max_n;
        self.name = Arc::new(format!("BLEU-{max_n}"));
        self
    }

    /// Sets the pad token index. Tokens matching this value are stripped from
    /// the right of each sequence before scoring.
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}

/// Extracts n-grams of order `n` from a slice and returns their counts.
fn ngram_counts(tokens: &[i32], n: usize) -> HashMap<Vec<i32>, usize> {
    let mut counts = HashMap::new();
    if tokens.len() >= n {
        for window in tokens.windows(n) {
            *counts.entry(window.to_vec()).or_insert(0) += 1;
        }
    }
    counts
}

/// Computes BLEU score for a single candidate/reference pair.
///
/// Returns a value in [0, 100].
fn sentence_bleu(candidate: &[i32], reference: &[i32], max_n: usize) -> f64 {
    if candidate.is_empty() {
        return 0.0;
    }

    // Brevity penalty
    let bp = if candidate.len() < reference.len() {
        (1.0 - reference.len() as f64 / candidate.len() as f64).exp()
    } else {
        1.0
    };

    // Modified n-gram precisions
    let mut log_avg = 0.0;
    let mut effective_order = 0;

    for n in 1..=max_n {
        let cand_ngrams = ngram_counts(candidate, n);
        let ref_ngrams = ngram_counts(reference, n);

        let mut clipped = 0usize;
        let mut total = 0usize;

        for (ngram, &count) in &cand_ngrams {
            let ref_count = ref_ngrams.get(ngram).copied().unwrap_or(0);
            clipped += count.min(ref_count);
            total += count;
        }

        if total == 0 {
            // No n-grams of this order in candidate (sequence too short).
            // Skip this order entirely instead of returning 0.
            continue;
        }

        if clipped == 0 {
            // Zero precision at this order means the whole score is 0.
            return 0.0;
        }

        log_avg += (clipped as f64 / total as f64).ln();
        effective_order += 1;
    }

    if effective_order == 0 {
        return 0.0;
    }

    let score = bp * (log_avg / effective_order as f64).exp();
    score * 100.0
}

impl<B: Backend> Metric for BleuScore<B> {
    type Input = BleuInput<B>;

    fn update(&mut self, input: &BleuInput<B>, _metadata: &MetricMetadata) -> SerializedEntry {
        let outputs = &input.outputs;
        let targets = &input.targets;
        let [batch_size, seq_len] = targets.dims();

        let outputs_data = outputs.to_data().iter::<i32>().collect::<Vec<_>>();
        let targets_data = targets.to_data().iter::<i32>().collect::<Vec<_>>();

        let pad_token = self.pad_token.map(|p| p as i32);

        let mut total_bleu = 0.0;

        for i in 0..batch_size {
            let start = i * seq_len;
            let end = (i + 1) * seq_len;

            let output_seq = &outputs_data[start..end];
            let target_seq = &targets_data[start..end];

            // Strip right-padding if configured.
            let output_seq = match pad_token {
                Some(pad) => {
                    let len = output_seq
                        .iter()
                        .position(|&x| x == pad)
                        .unwrap_or(output_seq.len());
                    &output_seq[..len]
                }
                None => output_seq,
            };
            let target_seq = match pad_token {
                Some(pad) => {
                    let len = target_seq
                        .iter()
                        .position(|&x| x == pad)
                        .unwrap_or(target_seq.len());
                    &target_seq[..len]
                }
                None => target_seq,
            };

            total_bleu += sentence_bleu(output_seq, target_seq, self.max_n);
        }

        let value = total_bleu / batch_size as f64;

        self.state.update(
            value,
            batch_size,
            FormatOptions::new(self.name()).unit("%").precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: Some("%".to_string()),
            higher_is_better: true,
        }
        .into()
    }
}

impl<B: Backend> Numeric for BleuScore<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    /// Perfect match => BLEU = 100 %.
    #[test]
    fn test_bleu_perfect_match() {
        let device = Default::default();
        let mut metric = BleuScore::<TestBackend>::new();

        let preds = Tensor::from_data([[1, 2, 3, 4, 5]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());

        assert!((metric.value().current() - 100.0).abs() < 1e-6);
    }

    /// Completely different sequences => BLEU = 0 %.
    #[test]
    fn test_bleu_no_match() {
        let device = Default::default();
        let mut metric = BleuScore::<TestBackend>::new();

        let preds = Tensor::from_data([[1, 2, 3, 4, 5]], &device);
        let tgts = Tensor::from_data([[6, 7, 8, 9, 10]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());

        assert_eq!(0.0, metric.value().current());
    }

    /// Partial overlap with BLEU-1 (unigram precision only).
    #[test]
    fn test_bleu1_partial_match() {
        let device = Default::default();
        let mut metric = BleuScore::<TestBackend>::new().with_max_n(1);

        // 3 out of 5 unigrams match, same length so BP = 1
        let preds = Tensor::from_data([[1, 2, 3, 6, 7]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());

        // BLEU-1 = 3/5 * 100 = 60.0
        assert!((metric.value().current() - 60.0).abs() < 1e-6);
    }

    /// Brevity penalty applied when candidate is shorter than reference.
    #[test]
    fn test_bleu_brevity_penalty() {
        let device = Default::default();
        let pad = 0_i64;
        let mut metric = BleuScore::<TestBackend>::new()
            .with_max_n(1)
            .with_pad_token(pad as usize);

        // Candidate has 3 tokens, reference has 5 tokens.
        // Unigram precision = 3/3 = 1.0, BP = exp(1 - 5/3) ~= 0.5134
        let preds = Tensor::from_data([[1, 2, 3, pad, pad]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());

        let expected = 100.0 * (1.0 - 5.0 / 3.0_f64).exp();
        assert!((metric.value().current() - expected).abs() < 0.1);
    }

    /// With padding, padding tokens should be stripped.
    #[test]
    fn test_bleu_with_padding() {
        let device = Default::default();
        let pad = 99_i64;
        let mut metric = BleuScore::<TestBackend>::new().with_pad_token(pad as usize);

        // Same non-pad content => should be 100%
        let preds = Tensor::from_data([[1, 2, 3, 4, 5, pad, pad]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5, pad, pad]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());

        assert!((metric.value().current() - 100.0).abs() < 1e-6);
    }

    /// Batch of two sequences: one perfect, one zero.
    #[test]
    fn test_bleu_batch_average() {
        let device = Default::default();
        let mut metric = BleuScore::<TestBackend>::new().with_max_n(1);

        // Sequence 1: perfect match (BLEU = 100)
        // Sequence 2: no match (BLEU = 0)
        let preds = Tensor::from_data([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());

        // Average: (100 + 0) / 2 = 50
        assert!((metric.value().current() - 50.0).abs() < 1e-6);
    }

    /// `clear()` must reset the running statistics.
    #[test]
    fn test_clear_resets_state() {
        let device = Default::default();
        let mut metric = BleuScore::<TestBackend>::new();

        let preds = Tensor::from_data([[1, 2, 3, 4, 5]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());
        assert!(metric.value().current() > 0.0);

        metric.clear();
        assert!(metric.value().current().is_nan());
    }

    /// BLEU-2 on a partial match verifies bigram counting.
    #[test]
    fn test_bleu2_bigrams() {
        let device = Default::default();
        let mut metric = BleuScore::<TestBackend>::new().with_max_n(2);

        // Candidate: [1, 2, 3, 4]  Reference: [1, 2, 5, 6]
        // Unigrams: candidate {1,2,3,4}, reference {1,2,5,6}
        //   matches: 1,2 => clipped 2/4
        // Bigrams: candidate {(1,2),(2,3),(3,4)}, reference {(1,2),(2,5),(5,6)}
        //   matches: (1,2) => clipped 1/3
        // BP = 1.0 (same length)
        // BLEU-2 = exp((ln(2/4) + ln(1/3)) / 2) * 100
        let preds = Tensor::from_data([[1, 2, 3, 4]], &device);
        let tgts = Tensor::from_data([[1, 2, 5, 6]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());

        let expected = 100.0 * ((0.5_f64.ln() + (1.0 / 3.0_f64).ln()) / 2.0).exp();
        assert!((metric.value().current() - expected).abs() < 0.1);
    }

    /// BLEU with custom name reflects max_n.
    #[test]
    fn test_bleu_custom_name() {
        let metric = BleuScore::<crate::TestBackend>::new().with_max_n(2);
        assert_eq!(*metric.name(), "BLEU-2");
    }
}
