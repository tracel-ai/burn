use super::state::{FormatOptions, NumericMetricState};
use super::{MetricMetadata, SerializedEntry};
use crate::metric::{
    Metric, MetricAttributes, MetricName, Numeric, NumericAttributes, NumericEntry,
};
use burn_core::tensor::{Int, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Smoothing method for BLEU score computation.
///
/// Sentence-level BLEU often produces zero scores when higher-order n-grams
/// have no matches. Smoothing techniques address this by assigning small
/// non-zero values to zero-count precisions.
///
/// # References
///
/// Chen & Cherry, "A Systematic Comparison of Smoothing Techniques for
/// Sentence-Level BLEU", WMT 2014.
#[derive(Clone, Debug, Default)]
pub enum BleuSmoothing {
    /// No smoothing. Zero precision at any n-gram order produces a zero
    /// overall score (standard corpus-level BLEU behaviour).
    #[default]
    None,

    /// Add a small constant (`epsilon`, default 0.1) to zero-count
    /// precisions. Corresponds to method 1 in Chen & Cherry (2014).
    AddEpsilon(f64),

    /// Exponential decay: for each n-gram order with zero matches, double a
    /// running multiplier `k` (starting at 1 and doubling on every zero) and
    /// replace the precision with `1 / (k * total_n)`. Corresponds to
    /// method 3 in Chen & Cherry (2014) and the default smoothing in
    /// SacreBLEU for sentence-level BLEU.
    Exponential,
}

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
/// # Batch-level scoring
///
/// Within each batch the metric accumulates n-gram counts across all
/// sentences and computes a single corpus-style BLEU score, following the
/// same pattern CER/WER use for edit-distance aggregation.
///
/// Epoch-level (running) aggregation averages these batch scores, which is
/// slightly inaccurate compared to true corpus BLEU. Correct corpus-level
/// accumulation would require a custom metric state; a TODO is left for
/// future work.
///
/// # References
///
/// Papineni et al., "BLEU: a Method for Automatic Evaluation of Machine
/// Translation", ACL 2002.
///
/// Chen & Cherry, "A Systematic Comparison of Smoothing Techniques for
/// Sentence-Level BLEU", WMT 2014.
#[derive(Clone)]
pub struct BleuScore {
    name: MetricName,
    state: NumericMetricState,
    max_n: usize,
    pad_token: Option<usize>,
    smoothing: BleuSmoothing,
}

/// The [BLEU score metric](BleuScore) input type.
#[derive(new)]
pub struct BleuInput {
    /// The predicted token sequences (2-D tensor of token indices).
    pub outputs: Tensor<2, Int>,
    /// The reference token sequences (2-D tensor of token indices).
    pub targets: Tensor<2, Int>,
}

impl Default for BleuScore {
    fn default() -> Self {
        Self::with_max_n(4)
    }
}

impl BleuScore {
    /// Creates a BLEU metric with the given maximum n-gram order.
    ///
    /// Common values: 1 (BLEU-1), 2 (BLEU-2), 4 (BLEU-4).
    ///
    /// # Panics
    ///
    /// Panics if `max_n` is zero.
    pub fn with_max_n(max_n: usize) -> Self {
        assert!(max_n >= 1, "max_n must be at least 1");
        Self {
            name: Arc::new(format!("BLEU-{max_n}")),
            state: NumericMetricState::default(),
            max_n,
            pad_token: None,
            smoothing: BleuSmoothing::default(),
        }
    }

    /// Creates a BLEU-4 metric (the most common configuration).
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the pad token index. Tokens matching this value are stripped from
    /// the right of each sequence before scoring.
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }

    /// Sets the smoothing method for handling zero-count n-gram precisions.
    ///
    /// Smoothing is recommended when evaluating short sentences where
    /// higher-order n-gram matches are sparse.
    pub fn with_smoothing(mut self, smoothing: BleuSmoothing) -> Self {
        self.smoothing = smoothing;
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

/// Computes corpus-style BLEU score from accumulated n-gram statistics.
///
/// `clipped_counts[n]` and `total_counts[n]` hold the clipped and total
/// n-gram counts for order `n+1` across all sentences.
/// `candidate_len` and `reference_len` are the total token counts.
///
/// Returns a value in [0, 100].
fn corpus_bleu(
    clipped_counts: &[usize],
    total_counts: &[usize],
    candidate_len: usize,
    reference_len: usize,
    max_n: usize,
    smoothing: &BleuSmoothing,
) -> f64 {
    if candidate_len == 0 {
        return 0.0;
    }

    // Brevity penalty
    let bp = if candidate_len < reference_len {
        (1.0 - reference_len as f64 / candidate_len as f64).exp()
    } else {
        1.0
    };

    // Modified n-gram precisions (log-space)
    let mut log_avg = 0.0;
    let mut counted_orders = 0;
    // Stateful multiplier for exponential smoothing (Chen & Cherry 2014, method 3;
    // also used by SacreBLEU). Doubles on every n-gram order with zero matches.
    let mut smooth_mult = 1.0_f64;

    for n in 0..max_n {
        let total = total_counts[n];
        let clipped = clipped_counts[n];

        if total == 0 {
            // Candidate has no n-grams of this order (too short).
            // Standard BLEU: this order contributes 0 to the geometric mean,
            // which collapses the entire score to 0.
            return 0.0;
        }

        let precision = if clipped == 0 {
            // Apply smoothing to zero-match precisions.
            match smoothing {
                BleuSmoothing::None => return 0.0,
                BleuSmoothing::AddEpsilon(eps) => *eps / total as f64,
                BleuSmoothing::Exponential => {
                    smooth_mult *= 2.0;
                    1.0 / (smooth_mult * total as f64)
                }
            }
        } else {
            clipped as f64 / total as f64
        };

        log_avg += precision.ln();
        counted_orders += 1;
    }

    if counted_orders == 0 {
        return 0.0;
    }

    let score = bp * (log_avg / counted_orders as f64).exp();
    score * 100.0
}

impl Metric for BleuScore {
    type Input = BleuInput;

    fn update(&mut self, input: &BleuInput, _metadata: &MetricMetadata) -> SerializedEntry {
        let outputs = &input.outputs;
        let targets = &input.targets;
        let [batch_size, seq_len] = targets.dims();

        let outputs_data = outputs.to_data().iter::<i32>().collect::<Vec<_>>();
        let targets_data = targets.to_data().iter::<i32>().collect::<Vec<_>>();

        let pad_token = self.pad_token.map(|p| p as i32);

        // Accumulate n-gram counts across the batch (corpus-style).
        let mut clipped_counts = vec![0usize; self.max_n];
        let mut total_counts = vec![0usize; self.max_n];
        let mut total_candidate_len = 0usize;
        let mut total_reference_len = 0usize;

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

            total_candidate_len += output_seq.len();
            total_reference_len += target_seq.len();

            for n in 1..=self.max_n {
                let cand_ngrams = ngram_counts(output_seq, n);
                let ref_ngrams = ngram_counts(target_seq, n);

                for (ngram, &count) in &cand_ngrams {
                    let ref_count = ref_ngrams.get(ngram).copied().unwrap_or(0);
                    clipped_counts[n - 1] += count.min(ref_count);
                    total_counts[n - 1] += count;
                }
            }
        }

        let value = corpus_bleu(
            &clipped_counts,
            &total_counts,
            total_candidate_len,
            total_reference_len,
            self.max_n,
            &self.smoothing,
        );

        // TODO: Epoch-level aggregation averages batch BLEU scores, which is
        // slightly inaccurate compared to true corpus BLEU. Correct
        // accumulation would require a custom metric state that tracks raw
        // n-gram counts across batches.
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

impl Numeric for BleuScore {
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

    /// Perfect match => BLEU = 100 %.
    #[test]
    fn test_bleu_perfect_match() {
        let device = Default::default();
        let mut metric = BleuScore::new();

        let preds = Tensor::from_data([[1, 2, 3, 4, 5]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());

        assert!((metric.value().current() - 100.0).abs() < 1e-6);
    }

    /// Completely different sequences => BLEU = 0 %.
    #[test]
    fn test_bleu_no_match() {
        let device = Default::default();
        let mut metric = BleuScore::new();

        let preds = Tensor::from_data([[1, 2, 3, 4, 5]], &device);
        let tgts = Tensor::from_data([[6, 7, 8, 9, 10]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());

        assert_eq!(0.0, metric.value().current());
    }

    /// Partial overlap with BLEU-1 (unigram precision only).
    #[test]
    fn test_bleu1_partial_match() {
        let device = Default::default();
        let mut metric = BleuScore::with_max_n(1);

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
        let mut metric = BleuScore::with_max_n(1).with_pad_token(pad as usize);

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
        let mut metric = BleuScore::new().with_pad_token(pad as usize);

        // Same non-pad content => should be 100%
        let preds = Tensor::from_data([[1, 2, 3, 4, 5, pad, pad]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5, pad, pad]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());

        assert!((metric.value().current() - 100.0).abs() < 1e-6);
    }

    /// Batch of two sequences: corpus-style accumulation.
    #[test]
    fn test_bleu_batch_corpus_style() {
        let device = Default::default();
        let mut metric = BleuScore::with_max_n(1);

        // Sequence 1: perfect match [1,2,3,4,5]
        // Sequence 2: no match [6,7,8,9,10] vs [11,12,13,14,15]
        // Corpus unigram: clipped = 5, total = 10, precision = 0.5
        // BP: candidate_len = 10, ref_len = 10, BP = 1.0
        // BLEU-1 = 50.0
        let preds = Tensor::from_data([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());

        assert!((metric.value().current() - 50.0).abs() < 1e-6);
    }

    /// `clear()` must reset the running statistics.
    #[test]
    fn test_clear_resets_state() {
        let device = Default::default();
        let mut metric = BleuScore::new();

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
        let mut metric = BleuScore::with_max_n(2);

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
        let metric = BleuScore::with_max_n(2);
        assert_eq!(*metric.name(), "BLEU-2");
    }

    /// Default name is BLEU-4.
    #[test]
    fn test_bleu_default_name() {
        let metric = BleuScore::new();
        assert_eq!(*metric.name(), "BLEU-4");
    }

    /// Short candidate with BLEU-4 returns 0 without smoothing
    /// (too few tokens for 4-grams).
    #[test]
    fn test_bleu_short_candidate_no_smoothing() {
        let device = Default::default();
        let pad = 0_i64;
        let mut metric = BleuScore::new().with_pad_token(pad as usize);

        // Only 3 tokens — no 4-grams exist, score must be 0.
        let preds = Tensor::from_data([[1, 2, 3, pad, pad]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());
        assert_eq!(0.0, metric.value().current());
    }

    /// Exponential smoothing produces a non-zero score for short candidates.
    #[test]
    fn test_bleu_exponential_smoothing() {
        let device = Default::default();
        let mut metric = BleuScore::with_max_n(2).with_smoothing(BleuSmoothing::Exponential);

        // Unigrams: {1,3,5,7,9} vs {1,2,3,4,5} — clipped = 2/5
        // Bigrams: {(1,3),(3,5),(5,7),(7,9)} vs {(1,2),(2,3),(3,4),(4,5)} — clipped = 0/4
        let preds = Tensor::from_data([[1, 3, 5, 7, 9]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5]], &device);

        // Verify without smoothing this is 0.
        let mut metric_no_smooth = BleuScore::with_max_n(2);
        metric_no_smooth.update(
            &BleuInput::new(preds.clone(), tgts.clone()),
            &MetricMetadata::fake(),
        );
        assert_eq!(0.0, metric_no_smooth.value().current());

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());
        assert!(
            metric.value().current() > 0.0,
            "smoothing should produce non-zero score"
        );
    }

    /// Add-epsilon smoothing produces a non-zero score for zero-precision orders.
    #[test]
    fn test_bleu_add_epsilon_smoothing() {
        let device = Default::default();
        let mut metric = BleuScore::with_max_n(2).with_smoothing(BleuSmoothing::AddEpsilon(0.1));

        // Unigrams: {1,3,5,7,9} vs {1,2,3,4,5} — clipped 2, total 5
        // Bigrams: {(1,3),(3,5),(5,7),(7,9)} vs {(1,2),(2,3),(3,4),(4,5)} — clipped 0, total 4
        let preds = Tensor::from_data([[1, 3, 5, 7, 9]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5]], &device);

        metric.update(&BleuInput::new(preds, tgts), &MetricMetadata::fake());
        assert!(
            metric.value().current() > 0.0,
            "epsilon smoothing should produce non-zero score"
        );
    }
}
