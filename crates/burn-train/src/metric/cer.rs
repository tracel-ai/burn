use super::state::{FormatOptions, NumericMetricState};
use super::{MetricMetadata, SerializedEntry};
use crate::metric::{Metric, MetricAttributes, MetricName, Numeric, NumericEntry};
use burn_core::tensor::{Int, Tensor, TensorReadError};
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
/// Predictions and targets must have the same batch size, but their sequence lengths may differ.
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

    /// Sets the pad token, which is ignored wherever it appears in predictions and targets.
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}

/// The [character error rate metric](CharErrorRate) implementation.
impl Metric for CharErrorRate {
    type Input = CerInput;

    fn update(
        &mut self,
        input: &CerInput,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let outputs = &input.outputs;
        let targets = &input.targets;
        let [output_batch_size, output_seq_len] = outputs.dims();
        let [batch_size, target_seq_len] = targets.dims();
        assert_eq!(
            output_batch_size, batch_size,
            "CER predictions and targets must have the same batch size"
        );

        let outputs_data: Vec<i32> = outputs.try_to_vec_as()?;
        let targets_data: Vec<i32> = targets.try_to_vec_as()?;
        let pad_token = self.pad_token.map(|pad| pad as i64);

        let mut total_edit_distance = 0;
        let mut total_target_length = 0;

        for i in 0..batch_size {
            let output_start = i * output_seq_len;
            let target_start = i * target_seq_len;
            let output_seq = &outputs_data[output_start..output_start + output_seq_len];
            let target_seq = &targets_data[target_start..target_start + target_seq_len];

            let (distance, target_len) = match pad_token {
                Some(pad) => {
                    let output_seq_no_pad = output_seq
                        .iter()
                        .copied()
                        .filter(|&token| i64::from(token) != pad)
                        .collect::<Vec<_>>();
                    let target_seq_no_pad = target_seq
                        .iter()
                        .copied()
                        .filter(|&token| i64::from(token) != pad)
                        .collect::<Vec<_>>();

                    (
                        edit_distance(&target_seq_no_pad, &output_seq_no_pad),
                        target_seq_no_pad.len(),
                    )
                }
                None => (edit_distance(target_seq, output_seq), target_seq.len()),
            };

            total_edit_distance += distance;
            total_target_length += target_len;
        }

        let value = if total_target_length > 0 {
            100.0 * total_edit_distance as f64 / total_target_length as f64
        } else {
            0.0
        };

        self.state.update(value, total_target_length);
        Ok(self
            .state
            .compute_update(FormatOptions::new(self.name()).unit("%").precision(2)))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self
            .state
            .compute_final(FormatOptions::new(self.name()).unit("%").precision(2)))
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
    fn value(&self) -> Option<NumericEntry> {
        Some(self.state.current_value())
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(self.state.running_value())
    }

    fn final_value(&self) -> NumericEntry {
        self.state.final_value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    /// Perfect match ⇒ CER = 0 %.
    #[test]
    fn test_cer_without_padding() {
        let device = Default::default();
        let mut metric = CharErrorRate::new();

        // Batch size = 2, sequence length = 2
        let preds = Tensor::from_data([[1, 2], [3, 4]], &device);
        let tgts = Tensor::from_data([[1, 2], [3, 4]], &device);

        metric
            .update(&CerInput::new(preds, tgts), &MetricMetadata::fake())
            .unwrap();

        assert_eq!(0.0, metric.value().unwrap().current());
    }

    /// Two edits in four target tokens ⇒ 50 %.
    #[test]
    fn test_cer_without_padding_two_errors() {
        let device = Default::default();
        let mut metric = CharErrorRate::new();

        // One substitution in each sequence.
        let preds = Tensor::from_data([[1, 2], [3, 5]], &device);
        let tgts = Tensor::from_data([[1, 3], [3, 4]], &device);

        metric
            .update(&CerInput::new(preds, tgts), &MetricMetadata::fake())
            .unwrap();

        // 2 edits / 4 tokens = 50 %
        assert_eq!(50.0, metric.value().unwrap().current());
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

        metric
            .update(&CerInput::new(preds, tgts), &MetricMetadata::fake())
            .unwrap();
        assert_eq!(50.0, metric.value().unwrap().current());
    }

    /// Both sequences contain [1, 2] after removing padding, so CER should be 0 %.
    #[test]
    fn test_cer_with_interspersed_padding() {
        let device = Default::default();
        let mut metric = CharErrorRate::new().with_pad_token(0);
        let preds = Tensor::from_data([[1, 2, 0]], &device);
        let tgts = Tensor::from_data([[1, 0, 2]], &device);

        metric
            .update(&CerInput::new(preds, tgts), &MetricMetadata::fake())
            .unwrap();

        assert_eq!(0.0, metric.value().unwrap().current());
    }

    /// Leading padding must also be ignored in predictions and targets.
    #[test]
    fn test_cer_with_leading_padding() {
        let device = Default::default();
        let mut metric = CharErrorRate::new().with_pad_token(0);
        let preds = Tensor::from_data([[0, 1, 2], [3, 4, 0]], &device);
        let tgts = Tensor::from_data([[1, 2, 0], [0, 3, 4]], &device);

        metric
            .update(&CerInput::new(preds, tgts), &MetricMetadata::fake())
            .unwrap();

        assert_eq!(0.0, metric.value().unwrap().current());
    }

    /// One deletion in four non-padding target tokens ⇒ 25 %.
    #[test]
    fn test_cer_with_mixed_padding_and_unequal_lengths() {
        let device = Default::default();
        let mut metric = CharErrorRate::new().with_pad_token(0);
        let preds = Tensor::from_data([[0, 1, 0, 2], [3, 0, 0, 0]], &device);
        let tgts = Tensor::from_data([[1, 0, 2, 0], [0, 3, 0, 4]], &device);

        metric
            .update(&CerInput::new(preds, tgts), &MetricMetadata::fake())
            .unwrap();

        assert_eq!(25.0, metric.value().unwrap().current());
    }

    /// An all-padding batch keeps the existing zero CER behavior.
    #[test]
    fn test_cer_with_only_padding() {
        let device = Default::default();
        let mut metric = CharErrorRate::new().with_pad_token(0);
        let preds = Tensor::from_data([[0, 0]], &device);
        let tgts = Tensor::from_data([[0, 0]], &device);

        metric
            .update(&CerInput::new(preds, tgts), &MetricMetadata::fake())
            .unwrap();

        assert_eq!(0.0, metric.value().unwrap().current());
    }

    /// Extra prediction tokens must count as insertions in every row.
    #[rstest]
    #[case(None, 100.0)]
    #[case(Some(0), 50.0)]
    fn test_cer_with_longer_predictions(#[case] pad_token: Option<usize>, #[case] expected: f64) {
        let device = Default::default();
        let mut metric = CharErrorRate::new();
        if let Some(pad) = pad_token {
            metric = metric.with_pad_token(pad);
        }
        let preds = Tensor::from_data([[1, 0, 2, 5], [3, 0, 4, 6]], &device);
        let tgts = Tensor::from_data([[1, 2], [3, 4]], &device);

        metric
            .update(&CerInput::new(preds, tgts), &MetricMetadata::fake())
            .unwrap();

        assert_eq!(expected, metric.value().unwrap().current());
    }

    /// Shorter predictions must count as deletions without reading past a row.
    #[rstest]
    #[case(None, 50.0)]
    #[case(Some(0), 100.0 * 2.0 / 6.0)]
    fn test_cer_with_shorter_predictions(#[case] pad_token: Option<usize>, #[case] expected: f64) {
        let device = Default::default();
        let mut metric = CharErrorRate::new();
        if let Some(pad) = pad_token {
            metric = metric.with_pad_token(pad);
        }
        let preds = Tensor::from_data([[1, 2], [3, 4]], &device);
        let tgts = Tensor::from_data([[1, 0, 2, 5], [3, 0, 4, 6]], &device);

        metric
            .update(&CerInput::new(preds, tgts), &MetricMetadata::fake())
            .unwrap();

        assert_eq!(expected, metric.value().unwrap().current());
    }

    #[rstest]
    #[case::fewer_predictions(false)]
    #[case::more_predictions(true)]
    #[should_panic(expected = "CER predictions and targets must have the same batch size")]
    fn test_cer_with_mismatched_batch_sizes(#[case] more_predictions: bool) {
        let device = Default::default();
        let mut metric = CharErrorRate::new();
        let mut preds = Tensor::from_data([[1, 2]], &device);
        let mut tgts = Tensor::from_data([[1, 2], [3, 4]], &device);
        if more_predictions {
            std::mem::swap(&mut preds, &mut tgts);
        }

        metric
            .update(&CerInput::new(preds, tgts), &MetricMetadata::fake())
            .unwrap();
    }

    /// `clear()` must reset the running statistics to zero.
    #[test]
    fn test_clear_resets_state() {
        let device = Default::default();
        let mut metric = CharErrorRate::new();

        let preds = Tensor::from_data([[1, 2]], &device);
        let tgts = Tensor::from_data([[1, 3]], &device); // one error

        metric
            .update(
                &CerInput::new(preds.clone(), tgts.clone()),
                &MetricMetadata::fake(),
            )
            .unwrap();
        assert!(metric.value().unwrap().current() > 0.0);

        metric.clear();
        assert!(metric.value().unwrap().current().is_nan());
    }
}
