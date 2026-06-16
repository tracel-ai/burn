use super::state::{FormatOptions, NumericMetricState};
use super::{MetricMetadata, SerializedEntry};
use crate::metric::{
    Metric, MetricAttributes, MetricName, Numeric, NumericAttributes, NumericEntry,
};
use burn_core::tensor::{Int, Tensor};
use std::sync::Arc;

fn lcs_length(a: &[i32], b: &[i32]) -> usize {
    let (shorter, longer) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    let mut prev = vec![0usize; shorter.len() + 1];
    let mut curr = vec![0usize; shorter.len() + 1];

    for &x in longer {
        for (j, &y) in shorter.iter().enumerate() {
            if x == y {
                curr[j + 1] = prev[j] + 1;
            } else {
                curr[j + 1] = curr[j].max(prev[j + 1]);
            }
        }
        core::mem::swap(&mut prev, &mut curr);
    }
    prev[shorter.len()]
}

/// ROUGE-L metric based on longest common subsequence.
#[derive(Clone)]
pub struct RougeLScore {
    name: MetricName,
    state: NumericMetricState,
    pad_token: Option<usize>,
}

/// Input for [RougeLScore].
#[derive(new)]
pub struct RougeLInput {
    /// Predicted token sequences.
    pub outputs: Tensor<2, Int>,
    /// Reference token sequences.
    pub targets: Tensor<2, Int>,
}

impl Default for RougeLScore {
    fn default() -> Self {
        Self::new()
    }
}

impl RougeLScore {
    /// Creates a new ROUGE-L metric.
    pub fn new() -> Self {
        Self {
            name: Arc::new("ROUGE-L".to_string()),
            state: NumericMetricState::default(),
            pad_token: None,
        }
    }

    /// Sets the pad token index. Tokens matching this value are stripped
    /// from the right of each sequence before scoring.
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}

impl Metric for RougeLScore {
    type Input = RougeLInput;

    fn update(&mut self, input: &RougeLInput, _metadata: &MetricMetadata) -> SerializedEntry {
        let outputs = &input.outputs;
        let targets = &input.targets;
        let [batch_size, seq_len] = targets.dims();

        let outputs_data = outputs.to_data().iter::<i32>().collect::<Vec<_>>();
        let targets_data = targets.to_data().iter::<i32>().collect::<Vec<_>>();

        let pad_token = self.pad_token.map(|p| p as i32);

        let mut total_f1 = 0.0_f64;

        for i in 0..batch_size {
            let start = i * seq_len;
            let end = (i + 1) * seq_len;

            let output_seq = &outputs_data[start..end];
            let target_seq = &targets_data[start..end];

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

            let lcs_len = lcs_length(target_seq, output_seq) as f64;
            let ref_len = target_seq.len() as f64;
            let cand_len = output_seq.len() as f64;

            if ref_len == 0.0 && cand_len == 0.0 {
                total_f1 += 100.0;
                continue;
            }

            if ref_len == 0.0 || cand_len == 0.0 {
                continue;
            }

            let precision = lcs_len / cand_len;
            let recall = lcs_len / ref_len;

            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            total_f1 += f1 * 100.0;
        }

        let value = total_f1 / batch_size as f64;

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
            ..Default::default()
        }
        .into()
    }
}

impl Numeric for RougeLScore {
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
    use burn_core::tensor::TensorData;

    #[test]
    fn test_rouge_l_perfect_match() {
        let device = Default::default();
        let mut metric = RougeLScore::new();

        let preds = Tensor::from_data([[1, 2, 3, 4, 5]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5]], &device);

        metric.update(&RougeLInput::new(preds, tgts), &MetricMetadata::fake());
        assert!((metric.value().current() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_l_no_match() {
        let device = Default::default();
        let mut metric = RougeLScore::new();

        let preds = Tensor::from_data([[1, 2, 3, 4, 5]], &device);
        let tgts = Tensor::from_data([[6, 7, 8, 9, 10]], &device);

        metric.update(&RougeLInput::new(preds, tgts), &MetricMetadata::fake());
        assert_eq!(0.0, metric.value().current());
    }

    #[test]
    fn test_rouge_l_partial_match() {
        let device = Default::default();
        let mut metric = RougeLScore::new();

        let preds = Tensor::from_data([[1, 3, 5, 7, 9]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5]], &device);

        metric.update(&RougeLInput::new(preds, tgts), &MetricMetadata::fake());

        let expected = 60.0;
        assert!((metric.value().current() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_l_shorter_candidate() {
        let device = Default::default();
        let pad = 99_i64;
        let mut metric = RougeLScore::new().with_pad_token(pad as usize);

        let preds = Tensor::from_data([[1, 2, 3, pad, pad]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5]], &device);

        metric.update(&RougeLInput::new(preds, tgts), &MetricMetadata::fake());

        let expected = 75.0;
        assert!((metric.value().current() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_l_with_padding() {
        let device = Default::default();
        let pad = 99_i64;
        let mut metric = RougeLScore::new().with_pad_token(pad as usize);

        let preds = Tensor::from_data([[1, 2, 3, 4, 5, pad, pad]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5, pad, pad]], &device);

        metric.update(&RougeLInput::new(preds, tgts), &MetricMetadata::fake());
        assert!((metric.value().current() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_l_batch() {
        let device = Default::default();
        let mut metric = RougeLScore::new();

        let preds = Tensor::from_data([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]], &device);

        metric.update(&RougeLInput::new(preds, tgts), &MetricMetadata::fake());
        assert!((metric.value().current() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge_l_both_empty() {
        let device = Default::default();
        let mut metric = RougeLScore::new();

        let preds = Tensor::<2, Int>::from_data(TensorData::from([[0i32; 0]; 1]), &device);
        let tgts = Tensor::<2, Int>::from_data(TensorData::from([[0i32; 0]; 1]), &device);

        metric.update(&RougeLInput::new(preds, tgts), &MetricMetadata::fake());
        assert!((metric.value().current() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_clear_resets_state() {
        let device = Default::default();
        let mut metric = RougeLScore::new();

        let preds = Tensor::from_data([[1, 2, 3, 4, 5]], &device);
        let tgts = Tensor::from_data([[1, 2, 3, 4, 5]], &device);

        metric.update(&RougeLInput::new(preds, tgts), &MetricMetadata::fake());
        assert!(metric.value().current() > 0.0);

        metric.clear();
        assert!(metric.value().current().is_nan());
    }
}
