use core::marker::PhantomData;

use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, MetricName, Numeric};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{ElementConversion, Int, Tensor};

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
#[derive(Clone)]
pub struct PerplexityMetric<B: Backend> {
    name: MetricName,
    state: NumericMetricState,
    pad_token: Option<usize>,
    _b: PhantomData<B>,
}

/// The [perplexity metric](PerplexityMetric) input type.
#[derive(new)]
pub struct PerplexityInput<B: Backend> {
    /// Logits tensor of shape [batch_size * sequence_length, vocab_size]
    outputs: Tensor<B, 2>,
    /// Target tokens tensor of shape [batch_size * sequence_length]
    targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Default for PerplexityMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> PerplexityMetric<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self {
            name: MetricName::new("Perplexity".to_string()),
            state: Default::default(),
            pad_token: Default::default(),
            _b: PhantomData,
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

impl<B: Backend> Metric for PerplexityMetric<B> {
    type Input = PerplexityInput<B>;

    fn update(&mut self, input: &PerplexityInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let targets = input.targets.clone();
        let outputs = input.outputs.clone();

        let [total_tokens, _vocab_size] = outputs.dims();

        // Convert logits to log probabilities using log_softmax for numerical stability
        let log_probs = burn_core::tensor::activation::log_softmax(outputs, 1);

        // Gather the log probabilities for the target tokens
        let target_log_probs = log_probs.gather(1, targets.clone().unsqueeze_dim(1)).squeeze(1);

        let (sum_log_prob, effective_tokens) = match self.pad_token {
            Some(pad_token) => {
                // Create a mask for non-padding tokens
                let mask = targets.clone().not_equal_elem(pad_token as i64);
                
                // Apply mask to log probabilities (set padding log probs to 0)
                let masked_log_probs = target_log_probs.mask_fill(mask.clone().bool_not(), 0.0);
                
                // Sum the log probabilities and count effective tokens
                let sum_log_prob = masked_log_probs.sum().into_scalar().elem::<f64>();
                let effective_tokens = mask.int().sum().into_scalar().elem::<i64>() as usize;
                
                (sum_log_prob, effective_tokens)
            }
            None => {
                // No padding, use all tokens
                let sum_log_prob = target_log_probs.sum().into_scalar().elem::<f64>();
                (sum_log_prob, total_tokens)
            }
        };

        // Calculate perplexity: exp(-average_log_likelihood)
        // Note: sum_log_prob is already negative log likelihood, so we negate it
        let perplexity = if effective_tokens > 0 {
            (-sum_log_prob / effective_tokens as f64).exp()
        } else {
            // If no effective tokens, return a very high perplexity
            f64::INFINITY
        };

        self.state.update(
            perplexity,
            effective_tokens,
            FormatOptions::new(self.name()).precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }
}

impl<B: Backend> Numeric for PerplexityMetric<B> {
    fn value(&self) -> super::NumericEntry {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_perplexity_perfect_prediction() {
        let device = Default::default();
        let mut metric = PerplexityMetric::<TestBackend>::new();
        
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
        let perplexity = metric.value().current();
        
        // Perfect predictions should result in very low perplexity (close to 1.0)
        assert!(perplexity < 1.1, "Perfect predictions should have low perplexity, got {}", perplexity);
    }

    #[test]
    fn test_perplexity_uniform_prediction() {
        let device = Default::default();
        let mut metric = PerplexityMetric::<TestBackend>::new();
        
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
        let perplexity = metric.value().current();
        
        // Uniform distribution over 3 classes should have perplexity ≈ 3.0
        assert!((perplexity - 3.0).abs() < 0.1, "Uniform distribution perplexity should be ~3.0, got {}", perplexity);
    }

    #[test]
    fn test_perplexity_with_padding() {
        let device = Default::default();
        let mut metric = PerplexityMetric::<TestBackend>::new().with_pad_token(3);
        
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
        let perplexity = metric.value().current();
        
        // Should only consider the first two predictions, both of which are confident
        assert!(perplexity < 1.1, "Good predictions with padding should have low perplexity, got {}", perplexity);
    }

    #[test]
    fn test_perplexity_wrong_prediction() {
        let device = Default::default();
        let mut metric = PerplexityMetric::<TestBackend>::new();
        
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
        let perplexity = metric.value().current();
        
        // Wrong predictions should result in high perplexity
        assert!(perplexity > 10.0, "Wrong predictions should have high perplexity, got {}", perplexity);
    }
}