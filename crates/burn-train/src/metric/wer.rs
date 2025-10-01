use super::cer::edit_distance;
use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{Int, Tensor};
use core::marker::PhantomData;

// The edit_distance function remains the same as it calculates the Levenshtein distance
// between two sequences. The "units" within the sequences will now be treated as words.
/// The word error rate (WER) metric, similar to the CER, is defined as the edit distance (e.g. Levenshtein distance) between the predicted
/// and reference word sequences, divided by the total number of words in the reference. Here, the "units" within the sequences are words.
///
#[derive(Default)]
pub struct WordErrorRate<B: Backend> {
    state: NumericMetricState,
    pad_token: Option<usize>,
    _b: PhantomData<B>,
}

/// The [word error rate metric](WerMetric) input type.
#[derive(new)]
pub struct WerInput<B: Backend> {
    /// The predicted token sequences (as a 2-D tensor of token indices).
    pub outputs: Tensor<B, 2, Int>,
    /// The target token sequences (as a 2-D tensor of token indices).
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> WordErrorRate<B> {
    /// Creates the metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the pad token.
    pub fn with_pad_token(mut self, index: usize) -> Self {
        self.pad_token = Some(index);
        self
    }
}

impl<B: Backend> Metric for WordErrorRate<B> {
    type Input = WerInput<B>;

    fn update(&mut self, input: &WerInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let outputs = input.outputs.clone();
        let targets = input.targets.clone();
        let [batch_size, seq_len] = targets.dims();

        let outputs_data = outputs
            .to_data()
            .to_vec::<i64>()
            .expect("Failed to convert outputs to Vec");
        let targets_data = targets
            .to_data()
            .to_vec::<i64>()
            .expect("Failed to convert targets to Vec");

        let pad_token = self.pad_token;

        let mut total_edit_distance = 0.0;
        let mut total_target_length = 0.0;

        // Process each sequence in the batch
        for i in 0..batch_size {
            let start = i * seq_len;
            let end = (i + 1) * seq_len;
            let output_seq = &outputs_data[start..end];
            let target_seq = &targets_data[start..end];

            // Handle padding and map elements to i32.
            // These sequences now represent "words" (token IDs).
            let output_seq_no_pad = match pad_token {
                Some(pad) => output_seq
                    .iter()
                    .take_while(|&&x| x != pad as i64)
                    .map(|&x| x as i32)
                    .collect::<Vec<_>>(),
                None => output_seq.iter().map(|&x| x as i32).collect(),
            };

            let target_seq_no_pad = match pad_token {
                Some(pad) => target_seq
                    .iter()
                    .take_while(|&&x| x != pad as i64)
                    .map(|&x| x as i32)
                    .collect::<Vec<_>>(),
                None => target_seq.iter().map(|&x| x as i32).collect(),
            };

            let ed = edit_distance(&target_seq_no_pad, &output_seq_no_pad);
            total_edit_distance += ed as f64;
            total_target_length += target_seq_no_pad.len() as f64;
        }

        // Compute current WER value as a percentage
        let value = if total_target_length > 0.0 {
            100.0 * total_edit_distance / total_target_length
        } else {
            0.0
        };

        self.state.update(
            value,
            batch_size,
            FormatOptions::new(self.name()).unit("%").precision(2),
        )
    }

    fn name(&self) -> String {
        "WER".to_string()
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

/// The [word error rate metric](WerMetric) implementation.
impl<B: Backend> Numeric for WordErrorRate<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    /// Perfect match => WER = 0 %.
    #[test]
    fn test_wer_without_padding() {
        let device = Default::default();
        let mut metric = WordErrorRate::<TestBackend>::new();

        // Batch size = 2, sequence length = 2
        let preds = Tensor::from_data([[1, 2], [3, 4]], &device);
        let tgts = Tensor::from_data([[1, 2], [3, 4]], &device);

        metric.update(&WerInput::new(preds, tgts), &MetricMetadata::fake());

        assert_eq!(0.0, metric.value());
    }

    /// Two word edits in four target words => 50 %.
    #[test]
    fn test_wer_without_padding_two_errors() {
        let device = Default::default();
        let mut metric = WordErrorRate::<TestBackend>::new();

        // One substitution in each sequence.
        // Sequence 1: target [1, 3], pred [1, 2] -> 1 error (3 vs 2)
        // Sequence 2: target [3, 4], pred [3, 5] -> 1 error (4 vs 5)
        let preds = Tensor::from_data([[1, 2], [3, 5]], &device);
        let tgts = Tensor::from_data([[1, 3], [3, 4]], &device);

        metric.update(&WerInput::new(preds, tgts), &MetricMetadata::fake());

        // Total errors = 2, Total target words = 4. WER = (2/4) * 100 = 50 %
        assert_eq!(50.0, metric.value());
    }

    /// Same scenario as above, but with right-padding (token 9) ignored.
    #[test]
    fn test_wer_with_padding() {
        let device = Default::default();
        let pad = 9_i64;
        let mut metric = WordErrorRate::<TestBackend>::new().with_pad_token(pad as usize);

        // Each row has three columns, last one is the pad token.
        // Target sequences after removing pad: [1, 3] and [3, 4] (total length 4)
        // Predicted sequences after removing pad: [1, 2] and [3, 5]
        let preds = Tensor::from_data([[1, 2, pad], [3, 5, pad]], &device);
        let tgts = Tensor::from_data([[1, 3, pad], [3, 4, pad]], &device);

        metric.update(&WerInput::new(preds, tgts), &MetricMetadata::fake());
        assert_eq!(50.0, metric.value());
    }

    /// `clear()` must reset the running statistics to NaN.
    #[test]
    fn test_clear_resets_state() {
        let device = Default::default();
        let mut metric = WordErrorRate::<TestBackend>::new();

        let preds = Tensor::from_data([[1, 2]], &device);
        let tgts = Tensor::from_data([[1, 3]], &device); // one error

        metric.update(
            &WerInput::new(preds.clone(), tgts.clone()),
            &MetricMetadata::fake(),
        );
        assert!(metric.value() > 0.0);

        metric.clear();
        assert!(metric.value().is_nan());
    }
}
