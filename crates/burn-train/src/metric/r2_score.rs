use std::sync::Arc;

use super::MetricMetadata;
use super::state::FormatOptions;
use crate::metric::{
    Metric, MetricAttributes, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry, format_float,
};
use burn_core::tensor::{Tensor, TensorReadError};

const EPSILON: f64 = 1e-8;

/// Internal state for tracking exact population R2 score across batches.
///
/// Supports both single-target and multi-target regression. For multi-target
/// outputs (D > 1), each target column's sum-of-squares is accumulated
/// independently and scores are uniformly averaged across target dimensions.
#[derive(Clone, Debug)]
struct R2ScoreState {
    sum_res: Vec<f64>,
    sum_y: Vec<f64>,
    sum_y_sq: Vec<f64>,
    total_samples: usize,
    current_r2: f64,
    current_count: usize,
}

impl Default for R2ScoreState {
    fn default() -> Self {
        Self {
            sum_res: Vec::new(),
            sum_y: Vec::new(),
            sum_y_sq: Vec::new(),
            total_samples: 0,
            current_r2: f64::NAN,
            current_count: 0,
        }
    }
}

impl R2ScoreState {
    fn update(
        &mut self,
        ss_res_vec: &[f64],
        sum_y_vec: &[f64],
        sum_y_sq_vec: &[f64],
        batch_size: usize,
    ) {
        let num_targets = ss_res_vec.len();
        if self.sum_res.is_empty() {
            self.sum_res = vec![0.0; num_targets];
            self.sum_y = vec![0.0; num_targets];
            self.sum_y_sq = vec![0.0; num_targets];
        }
        assert_eq!(
            self.sum_res.len(),
            num_targets,
            "Target dimension count must remain consistent across batches"
        );

        let mut batch_r2_sum = 0.0;
        for j in 0..num_targets {
            let ss_res = ss_res_vec[j];
            let sum_y = sum_y_vec[j];
            let sum_y_sq = sum_y_sq_vec[j];
            let batch_ss_tot = sum_y_sq - (sum_y * sum_y) / (batch_size as f64);
            let r2_j = if batch_size > 0 {
                if ss_res == 0.0 && batch_ss_tot.abs() < EPSILON {
                    1.0
                } else {
                    1.0 - (ss_res / (batch_ss_tot + EPSILON))
                }
            } else {
                f64::NAN
            };
            batch_r2_sum += r2_j;

            self.sum_res[j] += ss_res;
            self.sum_y[j] += sum_y;
            self.sum_y_sq[j] += sum_y_sq;
        }

        self.current_r2 = if num_targets > 0 {
            batch_r2_sum / num_targets as f64
        } else {
            f64::NAN
        };
        self.current_count = batch_size;
        self.total_samples += batch_size;
    }

    fn reset(&mut self) {
        *self = Self::default();
    }

    fn epoch_r2(&self) -> f64 {
        if self.total_samples == 0 || self.sum_res.is_empty() {
            return f64::NAN;
        }
        let num_targets = self.sum_res.len();
        let mut epoch_r2_sum = 0.0;
        for j in 0..num_targets {
            let ss_tot =
                self.sum_y_sq[j] - (self.sum_y[j] * self.sum_y[j]) / (self.total_samples as f64);
            let r2_j = if self.sum_res[j] == 0.0 && ss_tot.abs() < EPSILON {
                1.0
            } else {
                1.0 - (self.sum_res[j] / (ss_tot + EPSILON))
            };
            epoch_r2_sum += r2_j;
        }
        epoch_r2_sum / num_targets as f64
    }

    fn compute(&self, format: FormatOptions, final_entry: bool) -> SerializedEntry {
        let current_r2 = self.current_r2;
        let epoch_r2 = self.epoch_r2();

        let serialized = if final_entry {
            NumericEntry::Final(epoch_r2).serialize()
        } else {
            NumericEntry::Aggregated {
                aggregated_value: current_r2,
                count: self.current_count,
            }
            .serialize()
        };

        let (formatted_current, formatted_running) = match format.precision_value() {
            Some(precision) => (
                format_float(current_r2, precision),
                format_float(epoch_r2, precision),
            ),
            None => (format!("{current_r2}"), format!("{epoch_r2}")),
        };

        let formatted = match format.unit_value() {
            Some(unit) => {
                format!("epoch {formatted_running} {unit} - batch {formatted_current} {unit}")
            }
            None => format!("epoch {formatted_running} - batch {formatted_current}"),
        };

        SerializedEntry::new(formatted, serialized)
    }
}

/// The R2 Score (Coefficient of Determination) metric for regression tasks.
#[derive(Clone)]
pub struct R2ScoreMetric {
    name: MetricName,
    state: R2ScoreState,
}

/// The [R2 Score metric](R2ScoreMetric) input type.
#[derive(new)]
pub struct R2ScoreInput {
    /// The model outputs.
    pub outputs: Tensor<2>,
    /// The targets.
    pub targets: Tensor<2>,
}

impl Default for R2ScoreMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl R2ScoreMetric {
    /// Creates the metric.
    pub fn new() -> Self {
        Self {
            name: Arc::new("R2 Score".to_string()),
            state: R2ScoreState::default(),
        }
    }
}

impl Metric for R2ScoreMetric {
    type Input = R2ScoreInput;

    fn update(
        &mut self,
        input: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let [batch_size, _num_targets] = input.outputs.dims();
        assert_eq!(
            input.outputs.dims(),
            input.targets.dims(),
            "R2Score outputs and targets must have the same shape"
        );

        let residuals = input.targets.clone().sub(input.outputs.clone());
        let ss_res_vec = residuals
            .powf_scalar(2.0)
            .sum_dim(0)
            .try_into_vec_as::<f64>()?;
        let sum_y_vec = input.targets.clone().sum_dim(0).try_into_vec_as::<f64>()?;
        let sum_y_sq_vec = input
            .targets
            .clone()
            .powf_scalar(2.0)
            .sum_dim(0)
            .try_into_vec_as::<f64>()?;

        self.state
            .update(&ss_res_vec, &sum_y_vec, &sum_y_sq_vec, batch_size);
        Ok(self
            .state
            .compute(FormatOptions::new(self.name()).precision(4), false))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self
            .state
            .compute(FormatOptions::new(self.name()).precision(4), true))
    }

    fn clear(&mut self) {
        self.state.reset();
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }
}

impl Numeric for R2ScoreMetric {
    fn value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Aggregated {
            aggregated_value: self.state.current_r2,
            count: self.state.current_count,
        })
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Aggregated {
            aggregated_value: self.state.epoch_r2(),
            count: self.state.total_samples,
        })
    }

    fn final_value(&self) -> NumericEntry {
        NumericEntry::Final(self.state.epoch_r2())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learner::RegressionOutput;
    use crate::metric::Adaptor;
    use burn_core::tensor::Device;

    #[test]
    fn test_r2_perfect_prediction() {
        let device = Device::default();
        let mut metric = R2ScoreMetric::new();

        let input = R2ScoreInput::new(
            Tensor::from_data([[1.0], [2.0], [3.0]], &device),
            Tensor::from_data([[1.0], [2.0], [3.0]], &device),
        );

        let _ = metric.update(&input, &MetricMetadata::fake()).unwrap();
        assert_eq!(metric.final_value().current(), 1.0);
    }

    #[test]
    fn test_r2_mean_baseline_prediction() {
        let device = Device::default();
        let mut metric = R2ScoreMetric::new();

        // Targets: [1.0, 2.0, 3.0] -> mean is 2.0
        // Predictions: all 2.0 (predicting the mean)
        // R2 must be 0.0
        let input = R2ScoreInput::new(
            Tensor::from_data([[2.0], [2.0], [2.0]], &device),
            Tensor::from_data([[1.0], [2.0], [3.0]], &device),
        );

        let _ = metric.update(&input, &MetricMetadata::fake()).unwrap();
        assert!((metric.final_value().current() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_r2_known_values() {
        let device = Device::default();
        let mut metric = R2ScoreMetric::new();

        // y_true = [3.0, -0.5, 2.0, 7.0]
        // y_pred = [2.5, 0.0, 2.0, 8.0]
        // Exact R2 = 1.0 - 1.5 / 29.1875 = 0.9486081370449679
        let input = R2ScoreInput::new(
            Tensor::from_data([[2.5], [0.0], [2.0], [8.0]], &device),
            Tensor::from_data([[3.0], [-0.5], [2.0], [7.0]], &device),
        );

        let _ = metric.update(&input, &MetricMetadata::fake()).unwrap();
        let expected = 0.9486081370449679;
        assert!((metric.final_value().current() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_r2_multi_target_regression() {
        let device = Device::default();
        let mut metric = R2ScoreMetric::new();

        // Target with 2 output dimensions of different scale:
        // Col 0: y=[1, 2, 3], y_hat=[1.1, 1.9, 3.2] -> R2_0 = 0.97
        // Col 1: y=[10, 20, 30], y_hat=[10.5, 19.5, 30.2] -> R2_1 = 0.9973
        // Uniform average = (0.97 + 0.9973) / 2 = 0.98365
        let input = R2ScoreInput::new(
            Tensor::from_data([[1.1, 10.5], [1.9, 19.5], [3.2, 30.2]], &device),
            Tensor::from_data([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], &device),
        );

        let _ = metric.update(&input, &MetricMetadata::fake()).unwrap();
        let expected = (0.97 + 0.9973) / 2.0;
        assert!((metric.final_value().current() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_r2_multi_batch_consistency() {
        let device = Device::default();
        let mut metric = R2ScoreMetric::new();

        // Same dataset chunked into two batches:
        // Batch 1:
        let input1 = R2ScoreInput::new(
            Tensor::from_data([[2.5], [0.0]], &device),
            Tensor::from_data([[3.0], [-0.5]], &device),
        );
        let _ = metric.update(&input1, &MetricMetadata::fake()).unwrap();

        // Batch 2:
        let input2 = R2ScoreInput::new(
            Tensor::from_data([[2.0], [8.0]], &device),
            Tensor::from_data([[2.0], [7.0]], &device),
        );
        let _ = metric.update(&input2, &MetricMetadata::fake()).unwrap();

        // Across both batches, final epoch R2 must match single-batch exact value:
        let expected = 0.9486081370449679;
        assert!((metric.final_value().current() - expected).abs() < 1e-6);
    }

    #[test]
    fn test_r2_with_regression_output() {
        let device = Device::default();
        let mut metric = R2ScoreMetric::new();

        let output = RegressionOutput::new(
            Tensor::from_data([0.1], &device),
            Tensor::from_data([[1.0], [2.0]], &device),
            Tensor::from_data([[1.0], [2.0]], &device),
        );

        let input: R2ScoreInput = output.adapt();
        let _ = metric.update(&input, &MetricMetadata::fake()).unwrap();
        assert_eq!(metric.final_value().current(), 1.0);
    }

    #[test]
    fn test_r2_shape_mismatch_panics() {
        let device = Device::default();
        let mut metric = R2ScoreMetric::new();

        let input = R2ScoreInput::new(
            Tensor::from_data([[1.0, 2.0]], &device),
            Tensor::from_data([[1.0]], &device),
        );

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _ = metric.update(&input, &MetricMetadata::fake());
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_r2_clear() {
        let device = Device::default();
        let mut metric = R2ScoreMetric::new();

        let input = R2ScoreInput::new(
            Tensor::from_data([[1.0], [2.0]], &device),
            Tensor::from_data([[1.0], [2.0]], &device),
        );
        let _ = metric.update(&input, &MetricMetadata::fake()).unwrap();
        assert!(!metric.final_value().current().is_nan());

        metric.clear();
        assert!(metric.final_value().current().is_nan());
    }
}
