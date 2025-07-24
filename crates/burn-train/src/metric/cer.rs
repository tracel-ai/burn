use super::state::{FormatOptions, NumericMetricState};
use super::{MetricEntry, MetricMetadata};
use crate::metric::{Metric, Numeric};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{Int, Tensor};
use core::marker::PhantomData;

fn edit_distance(a: &[i32], b: &[i32]) -> usize {
    let mut prev = (0..=b.len()).collect::<Vec<_>>();
    let mut curr = vec![0; b.len() + 1];

    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            curr[j + 1] = if ca == cb {
                prev[j] // match
            } else {
                1 + prev[j].min(prev[j + 1]).min(curr[j]) // subst/ins/del
            };
        }
        core::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// The character error rate metric.
#[derive(Default)]
pub struct CerMetric<B: Backend> {
    state: NumericMetricState,
    pad_token: Option<usize>,
    _b: PhantomData<B>,
}

/// The [character error rate metric](CerMetric) input type.
#[derive(new)]
pub struct CerInput<B: Backend> {
    /// The predicted token sequences (as a 2-D tensor of token indices).
    pub outputs: Tensor<B, 2, Int>,
    /// The target token sequences (as a 2-D tensor of token indices).
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> CerMetric<B> {
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

/// The [character error rate metric](CerMetric) implementation.
impl<B: Backend> Metric for CerMetric<B> {
    type Input = CerInput<B>;

    fn update(&mut self, input: &CerInput<B>, _metadata: &MetricMetadata) -> MetricEntry {
        let outputs = input.outputs.clone();
        let targets = input.targets.clone();
        let [batch_size, seq_len] = targets.dims();

        // FIX 1: Use .expect() for error handling and get Vec<i64>.
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

            // FIX 2 & 3: Handle padding and map elements to i32.
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

            // Now this call is valid.
            let ed = edit_distance(&target_seq_no_pad, &output_seq_no_pad);
            total_edit_distance += ed as f64;
            total_target_length += target_seq_no_pad.len() as f64;
        }

        // Compute current CER value as a percentage
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

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> String {
        "CER".to_string()
    }
}

/// The [character error rate metric](CerMetric) implementation.
impl<B: Backend> Numeric for CerMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    /// Perfect match ⇒ CER = 0 %.
    #[test]
    fn test_cer_without_padding() {
        let device = Default::default();
        let mut metric = CerMetric::<TestBackend>::new();

        // Batch size = 2, sequence length = 2
        let preds = Tensor::from_data([[1, 2], [3, 4]], &device);
        let tgts = Tensor::from_data([[1, 2], [3, 4]], &device);

        metric.update(&CerInput::new(preds, tgts), &MetricMetadata::fake());

        assert_eq!(0.0, metric.value());
    }

    /// Two edits in four target tokens ⇒ 50 %.
    #[test]
    fn test_cer_without_padding_two_errors() {
        let device = Default::default();
        let mut metric = CerMetric::<TestBackend>::new();

        // One substitution in each sequence.
        let preds = Tensor::from_data([[1, 2], [3, 5]], &device);
        let tgts = Tensor::from_data([[1, 3], [3, 4]], &device);

        metric.update(&CerInput::new(preds, tgts), &MetricMetadata::fake());

        // 2 edits / 4 tokens = 50 %
        assert_eq!(50.0, metric.value());
    }

    /// Same scenario as above, but with right-padding (token 9) ignored.
    #[test]
    fn test_cer_with_padding() {
        let device = Default::default();
        let pad = 9_i64;
        let mut metric = CerMetric::<TestBackend>::new().with_pad_token(pad as usize);

        // Each row has three columns, last one is the pad token.
        let preds = Tensor::from_data([[1, 2, pad], [3, 5, pad]], &device);
        let tgts = Tensor::from_data([[1, 3, pad], [3, 4, pad]], &device);

        metric.update(&CerInput::new(preds, tgts), &MetricMetadata::fake());
        assert_eq!(50.0, metric.value());
    }

    /// `clear()` must reset the running statistics to zero.
    #[test]
    fn test_clear_resets_state() {
        let device = Default::default();
        let mut metric = CerMetric::<TestBackend>::new();

        let preds = Tensor::from_data([[1, 2]], &device);
        let tgts = Tensor::from_data([[1, 3]], &device); // one error

        metric.update(
            &CerInput::new(preds.clone(), tgts.clone()),
            &MetricMetadata::fake(),
        );
        assert!(metric.value() > 0.0);

        metric.clear();
        assert!(metric.value().is_nan());
    }
}
