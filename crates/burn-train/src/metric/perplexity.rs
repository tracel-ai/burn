use super::state::FormatOptions;
use super::{MetricMetadata, NumericEntry, SerializedEntry, format_float};
use crate::metric::{Metric, MetricAttributes, MetricName, Numeric, NumericAttributes};
use burn_core::tensor::{Int, Tensor};

/// Custom state for perplexity metric that correctly accumulates negative log-likelihood.
///
/// Unlike other metrics that can be averaged, perplexity requires special handling:
/// - Accumulate total negative log-likelihood across all tokens
/// - Accumulate total number of effective tokens
/// - Compute perplexity as exp(total_nll / total_tokens) only at the end
#[derive(Clone)]
struct PerplexityState {
    /// Sum of negative log-likelihood across all tokens
    sum_nll: f64,
    /// Total number of effective tokens (excluding padding)
    total_tokens: usize,
    /// Current batch perplexity (for display purposes)
    current: f64,
    current_tokens: usize,
}

impl PerplexityState {
    fn new() -> Self {
        Self {
            sum_nll: 0.0,
            total_tokens: 0,
            current: f64::NAN,
            current_tokens: 0,
        }
    }

    fn reset(&mut self) {
        self.sum_nll = 0.0;
        self.total_tokens = 0;
        self.current = f64::NAN;
        self.current_tokens = 0;
    }

    /// Update state with negative log-likelihood and token count from current batch
    fn update(&mut self, sum_log_prob: f64, effective_tokens: usize) {
        // sum_log_prob is already the sum of log probabilities (negative values)
        // We need to negate it to get negative log-likelihood
        let batch_nll = -sum_log_prob;

        // Accumulate across batches
        self.sum_nll += batch_nll;
        self.total_tokens += effective_tokens;

        // Compute current batch perplexity for display
        let batch_perplexity = if effective_tokens > 0 {
            (batch_nll / effective_tokens as f64).exp()
        } else {
            f64::INFINITY
        };
        self.current = batch_perplexity;
        self.current_tokens = effective_tokens;
    }

    /// Compute the metric for the current update.
    pub fn compute_update(&self, format: FormatOptions) -> SerializedEntry {
        self.compute(format, false)
    }

    /// Compute the final metric for the accumulated global state.
    pub fn compute_final(&self, format: FormatOptions) -> SerializedEntry {
        self.compute(format, true)
    }

    fn compute(&self, format: FormatOptions, final_entry: bool) -> SerializedEntry {
        let batch_perplexity = self.current;

        // Compute running epoch perplexity
        let epoch_perplexity = if self.total_tokens > 0 {
            (self.sum_nll / self.total_tokens as f64).exp()
        } else {
            f64::INFINITY
        };

        // Format for display
        let (formatted_current, formatted_running) = match format.precision_value() {
            Some(precision) => (
                format_float(batch_perplexity, precision),
                format_float(epoch_perplexity, precision),
            ),
            None => (format!("{batch_perplexity}"), format!("{epoch_perplexity}")),
        };

        let formatted = match format.unit_value() {
            Some(unit) => {
                format!("epoch {formatted_running} {unit} - batch {formatted_current} {unit}")
            }
            None => format!("epoch {formatted_running} - batch {formatted_current}"),
        };

        // Serialize the state for aggregation
        let serialized = if final_entry {
            NumericEntry::Final(epoch_perplexity).serialize()
        } else {
            NumericEntry::Aggregated {
                aggregated_value: batch_perplexity,
                count: self.current_tokens,
            }
            .serialize()
        };

        SerializedEntry::new(formatted, serialized)
    }

    fn current_value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Aggregated {
            // self.current is already exp(batch_nll / effective_tokens)
            aggregated_value: self.current,
            count: self.current_tokens,
        })
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Aggregated {
            aggregated_value: entry_value(self.sum_nll, self.total_tokens),
            count: self.total_tokens,
        })
    }

    fn final_value(&self) -> NumericEntry {
        NumericEntry::Final(entry_value(self.sum_nll, self.total_tokens))
    }
}

fn entry_value(value: f64, count: usize) -> f64 {
    if count > 0 {
        (value / count as f64).exp()
    } else {
        f64::INFINITY
    }
}

/// The perplexity metric.
///
/// Perplexity is a measure of how well a probability distribution or probability model
/// predicts a sample. It's commonly used to evaluate language models. A lower perplexity
/// indicates that the model is more confident in its predictions.
///
/// Mathematically, perplexity is defined as the exponentiation of the cross-entropy loss:
/// PPL = exp(H(p, q)) = exp(-1/N * Σ log(p(x_i)))
///
/// where:
/// - H(p, q) is the cross-entropy between the true distribution p and predicted distribution q
/// - N is the number of tokens
/// - p(x_i) is the predicted probability of the i-th token
///
/// # Aggregation
/// Unlike other metrics, perplexity cannot be simply averaged across batches.
/// This implementation correctly accumulates the total negative log-likelihood and
/// total token count across batches, then computes perplexity as exp(total_nll / total_tokens).
#[derive(Clone)]
pub struct PerplexityMetric {
    name: MetricName,
    state: PerplexityState,
    pad_token: Option<usize>,
}

/// The [perplexity metric](PerplexityMetric) input type.
#[derive(new)]
pub struct PerplexityInput {
    /// Logits tensor of shape [batch_size * sequence_length, vocab_size]
    outputs: Tensor<2>,
    /// Target tokens tensor of shape [batch_size * sequence_length]
    targets: Tensor<1, Int>,
}

impl Default for PerplexityMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl PerplexityMetric {
    /// Creates the metric.
    pub fn new() -> Self {
        Self {
            name: MetricName::new("Perplexity".to_string()),
            state: PerplexityState::new(),
            pad_token: Default::default(),
        }
    }

    /// Sets the pad token to exclude from perplexity calculation.
    ///
    /// When a pad token is set, predictions for padding tokens are masked out
    /// and do not contribute to the perplexity calculation. This is important
    /// for variable-length sequences where padding is used.
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}

impl Metric for PerplexityMetric {
    type Input = PerplexityInput;

    fn update(&mut self, input: &PerplexityInput, _metadata: &MetricMetadata) -> SerializedEntry {
        let targets = input.targets.clone();
        let outputs = input.outputs.clone();

        let [total_tokens, _vocab_size] = outputs.dims();

        // Convert logits to log probabilities using log_softmax for numerical stability
        let log_probs = burn_core::tensor::activation::log_softmax(outputs, 1);

        // Gather the log probabilities for the target tokens
        let target_log_probs = log_probs
            .gather(1, targets.clone().unsqueeze_dim(1))
            .squeeze_dim(1);

        let (sum_log_prob, effective_tokens) = match self.pad_token {
            Some(pad_token) => {
                // Create a mask for non-padding tokens
                let mask = targets.clone().not_equal_scalar(pad_token as i64);

                // Apply mask to log probabilities (set padding log probs to 0)
                let masked_log_probs = target_log_probs.mask_fill(mask.clone().bool_not(), 0.0);

                // Sum the log probabilities and count effective tokens
                let sum_log_prob = masked_log_probs.sum().into_scalar::<f64>();
                let effective_tokens = mask.int().sum().into_scalar::<i64>() as usize;

                (sum_log_prob, effective_tokens)
            }
            None => {
                // No padding, use all tokens
                let sum_log_prob = target_log_probs.sum().into_scalar::<f64>();
                (sum_log_prob, total_tokens)
            }
        };

        // Pass the sum_log_prob and effective_tokens to the state
        // The state will handle the correct accumulation and perplexity calculation
        self.state.update(sum_log_prob, effective_tokens);
        self.state
            .compute_update(FormatOptions::new(self.name()).precision(2))
    }

    fn compute(&mut self) -> SerializedEntry {
        self.state
            .compute_final(FormatOptions::new(self.name()).precision(2))
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }
}

impl Numeric for PerplexityMetric {
    fn value(&self) -> Option<NumericEntry> {
        self.state.current_value()
    }

    fn running_value(&self) -> Option<NumericEntry> {
        self.state.running_value()
    }

    fn final_value(&self) -> NumericEntry {
        self.state.final_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perplexity_perfect_prediction() {
        let device = Default::default();
        let mut metric = PerplexityMetric::new();

        // Perfect prediction: target is always the highest probability class
        let input = PerplexityInput::new(
            Tensor::from_data(
                [
                    [10.0, 0.0, 0.0], // Very confident prediction for class 0
                    [0.0, 10.0, 0.0], // Very confident prediction for class 1
                    [0.0, 0.0, 10.0], // Very confident prediction for class 2
                ],
                &device,
            ),
            Tensor::from_data([0, 1, 2], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        let perplexity = metric.value().unwrap().current();

        // Perfect predictions should result in very low perplexity (close to 1.0)
        assert!(
            perplexity < 1.1,
            "Perfect predictions should have low perplexity, got {}",
            perplexity
        );
    }

    #[test]
    fn test_perplexity_uniform_prediction() {
        let device = Default::default();
        let mut metric = PerplexityMetric::new();

        // Uniform prediction: all classes have equal probability
        let input = PerplexityInput::new(
            Tensor::from_data(
                [
                    [0.0, 0.0, 0.0], // Uniform distribution (after softmax)
                    [0.0, 0.0, 0.0], // Uniform distribution (after softmax)
                    [0.0, 0.0, 0.0], // Uniform distribution (after softmax)
                ],
                &device,
            ),
            Tensor::from_data([0, 1, 2], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        let perplexity = metric.value().unwrap().current();

        // Uniform distribution over 3 classes should have perplexity ≈ 3.0
        assert!(
            (perplexity - 3.0).abs() < 0.1,
            "Uniform distribution perplexity should be ~3.0, got {}",
            perplexity
        );
    }

    #[test]
    fn test_perplexity_with_padding() {
        let device = Default::default();
        let mut metric = PerplexityMetric::new().with_pad_token(3);

        let input = PerplexityInput::new(
            Tensor::from_data(
                [
                    [10.0, 0.0, 0.0, 0.0], // Good prediction for class 0
                    [0.0, 10.0, 0.0, 0.0], // Good prediction for class 1
                    [0.0, 0.0, 0.0, 1.0],  // This is padding - should be ignored
                    [0.0, 0.0, 0.0, 1.0],  // This is padding - should be ignored
                ],
                &device,
            ),
            Tensor::from_data([0, 1, 3, 3], &device), // 3 is pad token
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        let perplexity = metric.value().unwrap().current();

        // Should only consider the first two predictions, both of which are confident
        assert!(
            perplexity < 1.1,
            "Good predictions with padding should have low perplexity, got {}",
            perplexity
        );
    }

    #[test]
    fn test_perplexity_wrong_prediction() {
        let device = Default::default();
        let mut metric = PerplexityMetric::new();

        // Wrong predictions: target class has very low probability
        let input = PerplexityInput::new(
            Tensor::from_data(
                [
                    [0.0, 10.0, 0.0], // Predicts class 1, but target is 0
                    [10.0, 0.0, 0.0], // Predicts class 0, but target is 1
                    [0.0, 0.0, 10.0], // Predicts class 2, but target is 0
                ],
                &device,
            ),
            Tensor::from_data([0, 1, 0], &device),
        );

        let _entry = metric.update(&input, &MetricMetadata::fake());
        let perplexity = metric.value().unwrap().current();

        // Wrong predictions should result in high perplexity
        assert!(
            perplexity > 10.0,
            "Wrong predictions should have high perplexity, got {}",
            perplexity
        );
    }

    #[test]
    fn test_perplexity_multi_batch_aggregation() {
        let device = Default::default();
        let mut metric = PerplexityMetric::new();

        // First batch: 2 tokens with uniform distribution (log_prob ≈ -1.0986 each)
        let input1 = PerplexityInput::new(
            Tensor::from_data(
                [
                    [0.0, 0.0, 0.0], // Uniform distribution (log_prob ≈ -1.0986)
                    [0.0, 0.0, 0.0], // Uniform distribution (log_prob ≈ -1.0986)
                ],
                &device,
            ),
            Tensor::from_data([0, 1], &device),
        );

        // Second batch: 1 token with uniform distribution
        let input2 = PerplexityInput::new(
            Tensor::from_data(
                [
                    [0.0, 0.0, 0.0], // Uniform distribution (log_prob ≈ -1.0986)
                ],
                &device,
            ),
            Tensor::from_data([2], &device),
        );

        // Update with both batches
        let _entry1 = metric.update(&input1, &MetricMetadata::fake());
        let _entry2 = metric.update(&input2, &MetricMetadata::fake());

        let aggregated_perplexity = metric.value().unwrap().current();

        // For uniform distribution over 3 classes: log_prob ≈ -log(3) ≈ -1.0986
        // Total negative log-likelihood: 3 * 1.0986 ≈ 3.2958
        // Total tokens: 3
        // Expected perplexity: exp(3.2958 / 3) = exp(1.0986) ≈ 3.0
        assert!(
            (aggregated_perplexity - 3.0).abs() < 0.1,
            "Multi-batch aggregated perplexity should be ~3.0, got {}",
            aggregated_perplexity
        );

        // Compare with single batch containing all data
        let mut single_batch_metric = PerplexityMetric::new();
        let single_input = PerplexityInput::new(
            Tensor::from_data([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], &device),
            Tensor::from_data([0, 1, 2], &device),
        );

        let _single_entry = single_batch_metric.update(&single_input, &MetricMetadata::fake());
        let single_batch_perplexity = single_batch_metric.value().unwrap().current();

        // Multi-batch and single-batch should give the same result
        assert!(
            (aggregated_perplexity - single_batch_perplexity).abs() < 0.01,
            "Multi-batch ({}) and single-batch ({}) perplexity should match",
            aggregated_perplexity,
            single_batch_perplexity
        );
    }

    #[test]
    fn test_perplexity_global_aggregation_end_to_end() {
        let device = Default::default();
        let mut metric = PerplexityMetric::new();

        // Batch 1 (1 token, confident prediction -> low NLL):
        // logits lead to log_prob ≈ -0.1 -> NLL = 0.1
        let input_batch1 = PerplexityInput::new(
            Tensor::from_data([[5.0, 0.0, 0.0]], &device),
            Tensor::from_data([0], &device),
        );
        let _ = metric.update(&input_batch1, &MetricMetadata::fake());

        let batch1_current = metric.value().unwrap().current();
        let batch1_running = metric.running_value().unwrap().current();
        assert_eq!(batch1_current, batch1_running);

        // Batch 2 (5 tokens, uniform predictions -> higher NLL):
        // uniform over 3 classes -> log_prob ≈ -log(3) ≈ -1.0986 -> NLL = 1.0986 per token
        let input_batch2 = PerplexityInput::new(
            Tensor::from_data(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                &device,
            ),
            Tensor::from_data([0, 1, 2, 0, 1], &device),
        );
        let _ = metric.update(&input_batch2, &MetricMetadata::fake());

        // Batch 2 current batch perplexity should be ≈ 3.0
        let batch2_current = metric.value().unwrap().current();
        assert!((batch2_current - 3.0).abs() < 0.1);

        // Compute final epoch entry
        let _serialized_compute = metric.compute();
        let final_ppl = metric.final_value().current();

        // Total NLL = 0.013416 + 5.493061 = 5.506477
        // Global PPL = exp(5.506477 / 6 tokens) ≈ 2.5036
        // Note: Simple unweighted average of batch PPLs would incorrectly give (1.0135 + 3.0) / 2 = 2.0067
        let expected_global_ppl = 2.5036;

        assert!(
            (final_ppl - expected_global_ppl).abs() < 1e-3,
            "Expected final aggregated perplexity ~{}, got {}",
            expected_global_ppl,
            final_ppl
        );
    }
}
