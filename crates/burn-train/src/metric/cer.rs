use super::state::{FormatOptions, NumericMetricState};
use super::{MetricMetadata, SerializedEntry};
use crate::metric::{Metric, MetricAttributes, MetricName, Numeric, NumericEntry};
use burn_core::tensor::{Int, Tensor};
use std::sync::Arc;

/// Computes the edit distance (Levenshtein distance) between two sequences of integers.
///
/// The edit distance is defined as the minimum number of single-element edits (insertions,
/// deletions, or substitutions) required to change one sequence into the other. This
/// implementation is optimized for space, using only two rows of the dynamic programming table.
///
pub(crate) fn edit_distance(reference: &[i32], prediction: &[i32]) -> usize {
    let mut prev = (0..=prediction.len()).collect::<Vec<_>>();
    let mut curr = vec![0; prediction.len() + 1];

    for (i, &r) in reference.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &p) in prediction.iter().enumerate() {
            curr[j + 1] = if r == p {
                prev[j] // no operation needed
            } else {
                1 + prev[j].min(prev[j + 1]).min(curr[j]) // substitution, insertion, deletion
            };
        }
        core::mem::swap(&mut prev, &mut curr);
    }
    prev[prediction.len()]
}

/// Character error rate (CER) is defined as the edit distance (e.g. Levenshtein distance) between the predicted
/// and reference character sequences, divided by the total number of characters in the reference.
/// This metric is commonly used in tasks such as speech recognition, OCR, or text generation
/// to quantify how closely the predicted output matches the ground truth at a character level.
///
#[derive(Clone)]
pub struct CharErrorRate {
    name: MetricName,
    state: NumericMetricState,
    pad_token: Option<usize>,
}

/// The [character error rate metric](CharErrorRate) input type.
#[derive(new)]
pub struct CerInput {
    /// The predicted token sequences (as a 2-D tensor of token indices).
    pub outputs: Tensor<2, Int>,
    /// The target token sequences (as a 2-D tensor of token indices).
    pub targets: Tensor<2, Int>,
}

impl Default for CharErrorRate {
    fn default() -> Self {
        Self::new()
    }
}

impl CharErrorRate {
    /// Creates the metric.
    pub fn new() -> Self {
        Self {
            name: Arc::new("CER".to_string()),
            state: NumericMetricState::default(),
            pad_token: None,
        }
    }

    /// Sets the pad token.
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}

/// The [character error rate metric](CharErrorRate) implementation.
impl Metric for CharErrorRate {
    type Input = CerInput;

    fn update(&mut self, input: &CerInput, _metadata: &MetricMetadata) -> SerializedEntry {
        let outputs = &input.outputs;
        let targets = &input.targets;
        let [batch_size, seq_len] = targets.dims();

        let (output_lengths, target_lengths) = if let Some(pad) = self.pad_token {
            // Create boolean masks for non-padding tokens.
            let output_mask = outputs.clone().not_equal_elem(pad as i64);
            let target_mask = targets.clone().not_equal_elem(pad as i64);

            let output_lengths_tensor = output_mask.int().sum_dim(1);
            let target_lengths_tensor = target_mask.int().sum_dim(1);

            (
                output_lengths_tensor
                    .into_data()
                    .convert::<i32>()
                    .to_vec()
                    .unwrap(),
                target_lengths_tensor
                    .into_data()
                    .convert::<i32>()
                    .to_vec()
                    .unwrap(),
            )
        } else {
            // If there's no padding, all sequences have the full length.
            (
                vec![seq_len as i32; batch_size],
                vec![seq_len as i32; batch_size],
            )
        };

        let outputs_data = outputs.to_data().convert::<i32>().to_vec().unwrap();
        let targets_data = targets.to_data().convert::<i32>().to_vec().unwrap();

        let total_edit_distance: usize = (0..batch_size)
            .map(|i| {
                let start = i * seq_len;

                // Get pre-calculated lengths for the current sequence.
                let output_len = output_lengths[i] as usize;
                let target_len = target_lengths[i] as usize;

                let output_seq_slice = &outputs_data[start..(start + output_len)];
                let target_seq_slice = &targets_data[start..(start + target_len)];

                edit_distance(target_seq_slice, output_seq_slice)
            })
            .sum();

        let total_target_length = target_lengths.iter().map(|&x| x as f64).sum::<f64>();

        let value = if total_target_length > 0.0 {
            100.0 * total_edit_distance as f64 / total_target_length
        } else {
            0.0
        };

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
        super::NumericAttributes {
            unit: Some("%".to_string()),
            higher_is_better: false,
        }
        .into()
    }
}

impl Numeric for CharErrorRate {
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

    /// Perfect match ⇒ CER = 0 %.
    #[test]
    fn test_cer_without_padding() {
        let device = Default::default();
        let mut metric = CharErrorRate::new();

        // Batch size = 2, sequence length = 2
        let preds = Tensor::from_data([[1, 2], [3, 4]], &device);
        let tgts = Tensor::from_data([[1, 2], [3, 4]], &device);

        metric.update(&CerInput::new(preds, tgts), &MetricMetadata::fake());

        assert_eq!(0.0, metric.value().current());
    }

    /// Two edits in four target tokens ⇒ 50 %.
    #[test]
    fn test_cer_without_padding_two_errors() {
        let device = Default::default();
        let mut metric = CharErrorRate::new();

        // One substitution in each sequence.
        let preds = Tensor::from_data([[1, 2], [3, 5]], &device);
        let tgts = Tensor::from_data([[1, 3], [3, 4]], &device);

        metric.update(&CerInput::new(preds, tgts), &MetricMetadata::fake());

        // 2 edits / 4 tokens = 50 %
        assert_eq!(50.0, metric.value().current());
    }

    /// Same scenario as above, but with right-padding (token 9) ignored.
    #[test]
    fn test_cer_with_padding() {
        let device = Default::default();
        let pad = 9_i64;
        let mut metric = CharErrorRate::new().with_pad_token(pad as usize);

        // Each row has three columns, last one is the pad token.
        let preds = Tensor::from_data([[1, 2, pad], [3, 5, pad]], &device);
        let tgts = Tensor::from_data([[1, 3, pad], [3, 4, pad]], &device);

        metric.update(&CerInput::new(preds, tgts), &MetricMetadata::fake());
        assert_eq!(50.0, metric.value().current());
    }

    /// `clear()` must reset the running statistics to zero.
    #[test]
    fn test_clear_resets_state() {
        let device = Default::default();
        let mut metric = CharErrorRate::new();

        let preds = Tensor::from_data([[1, 2]], &device);
        let tgts = Tensor::from_data([[1, 3]], &device); // one error

        metric.update(
            &CerInput::new(preds.clone(), tgts.clone()),
            &MetricMetadata::fake(),
        );
        assert!(metric.value().current() > 0.0);

        metric.clear();
        assert!(metric.value().current().is_nan());
    }
}
