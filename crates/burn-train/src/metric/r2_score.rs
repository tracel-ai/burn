use super::MetricMetadata;
use crate::metric::{
    Metric, MetricAttributes, MetricName, Numeric, NumericEntry, SerializedEntry, format_float,
};
use burn_core::tensor::{Tensor, TensorReadError};

/// Coefficient of determination for regression tasks.
///
/// Scores are computed per target and averaged uniformly, as in scikit-learn's
/// `r2_score` default. Sufficient statistics are accumulated across batches so
/// the epoch score is computed over all samples, rather than averaging batch R².
/// Constant targets score 1 for perfect predictions and 0 otherwise.
#[derive(Clone)]
pub struct R2ScoreMetric {
    name: MetricName,
    count: usize,
    mean: Vec<f64>,
    total_ss: Vec<f64>,
    residual_ss: Vec<f64>,
    current_count: usize,
    current: Option<f64>,
    running: Option<f64>,
}

/// The [R² metric](R2ScoreMetric) input type.
#[derive(new)]
pub struct RegressionInput {
    /// The model outputs, shaped `[samples, targets]`.
    pub outputs: Tensor<2>,
    /// The targets, shaped `[samples, targets]`.
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
            name: MetricName::new("R2 Score".to_string()),
            count: 0,
            mean: Vec::new(),
            total_ss: Vec::new(),
            residual_ss: Vec::new(),
            current_count: 0,
            current: None,
            running: None,
        }
    }

    fn score(count: usize, total_ss: &[f64], residual_ss: &[f64]) -> f64 {
        if count < 2 {
            return f64::NAN;
        }
        total_ss
            .iter()
            .zip(residual_ss)
            .map(|(&total, &residual)| {
                if total == 0.0 {
                    if residual == 0.0 { 1.0 } else { 0.0 }
                } else {
                    1.0 - residual / total
                }
            })
            .sum::<f64>()
            / total_ss.len() as f64
    }

    fn entry(&self, final_entry: bool) -> SerializedEntry {
        let current = self.current.unwrap_or(f64::NAN);
        let running = self.running.unwrap_or(f64::NAN);
        let numeric = if final_entry {
            NumericEntry::Final(running)
        } else {
            NumericEntry::Aggregated {
                aggregated_value: current,
                count: self.current_count,
            }
        };
        SerializedEntry::new(
            format!(
                "epoch {} - batch {}",
                format_float(running, 4),
                format_float(current, 4)
            ),
            numeric.serialize(),
        )
    }
}

impl Metric for R2ScoreMetric {
    type Input = RegressionInput;

    fn update(
        &mut self,
        input: &RegressionInput,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let outputs = input.outputs.clone();
        let targets = input.targets.clone();
        assert_eq!(outputs.dims(), targets.dims(), "R² shapes must match");
        let [batch_size, num_targets] = targets.dims();
        assert!(
            batch_size > 0 && num_targets > 0,
            "R² requires non-empty input"
        );
        assert!(
            self.mean.is_empty() || self.mean.len() == num_targets,
            "R² target count must remain constant between batches"
        );

        let batch_mean = targets.clone().mean_dim(0);
        let batch_total_ss = (targets.clone() - batch_mean.clone())
            .powf_scalar(2.0)
            .sum_dim(0);
        let batch_residual_ss = (targets - outputs).powf_scalar(2.0).sum_dim(0);
        let batch_mean = batch_mean.try_into_data()?.try_into_vec_as::<f64>()?;
        let batch_total_ss = batch_total_ss.try_into_data()?.try_into_vec_as::<f64>()?;
        let batch_residual_ss = batch_residual_ss
            .try_into_data()?
            .try_into_vec_as::<f64>()?;

        self.current = Some(Self::score(batch_size, &batch_total_ss, &batch_residual_ss));
        self.current_count = batch_size;
        if self.count == 0 {
            self.mean = batch_mean;
            self.total_ss = batch_total_ss;
            self.residual_ss = batch_residual_ss;
        } else {
            let total_count = self.count + batch_size;
            let correction = self.count as f64 * batch_size as f64 / total_count as f64;
            for index in 0..num_targets {
                let delta = batch_mean[index] - self.mean[index];
                self.mean[index] += delta * batch_size as f64 / total_count as f64;
                self.total_ss[index] += batch_total_ss[index] + delta * delta * correction;
                self.residual_ss[index] += batch_residual_ss[index];
            }
        }
        self.count += batch_size;
        self.running = Some(Self::score(self.count, &self.total_ss, &self.residual_ss));
        Ok(self.entry(false))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self.entry(true))
    }

    fn clear(&mut self) {
        self.count = 0;
        self.mean.clear();
        self.total_ss.clear();
        self.residual_ss.clear();
        self.current_count = 0;
        self.current = None;
        self.running = None;
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        super::NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }
}

impl Numeric for R2ScoreMetric {
    fn value(&self) -> Option<NumericEntry> {
        self.current.map(|value| NumericEntry::Aggregated {
            aggregated_value: value,
            count: self.current_count,
        })
    }

    fn running_value(&self) -> Option<NumericEntry> {
        self.running.map(|value| NumericEntry::Aggregated {
            aggregated_value: value,
            count: self.count,
        })
    }

    fn final_value(&self) -> NumericEntry {
        NumericEntry::Final(self.running.unwrap_or(f64::NAN))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sklearn_reference_and_uneven_batches() {
        // scikit-learn 1.8.0: r2_score(y_true, y_pred) == 0.759375.
        // Batch scores are 0.25 and 0.11607142857; averaging them is wrong.
        let device = Default::default();
        let targets = [[1., 10.], [2., 10.], [3., 12.], [4., 14.], [5., 18.]];
        let outputs = [[1., 9.], [2.5, 11.], [2., 11.], [5., 14.], [4., 20.]];
        let mut metric = R2ScoreMetric::new();
        metric
            .update(
                &RegressionInput::new(
                    Tensor::from_data(outputs, &device),
                    Tensor::from_data(targets, &device),
                ),
                &MetricMetadata::fake(),
            )
            .unwrap();
        assert!((metric.final_value().current() - 0.759375).abs() < 1e-6);

        metric.clear();
        for range in [0..2, 2..5] {
            metric
                .update(
                    &RegressionInput::new(
                        Tensor::from_data(outputs, &device).slice([range.clone()]),
                        Tensor::from_data(targets, &device).slice([range]),
                    ),
                    &MetricMetadata::fake(),
                )
                .unwrap();
        }
        assert!((metric.running_value().unwrap().current() - 0.759375).abs() < 1e-6);
        metric.compute().unwrap();
        assert!((metric.final_value().current() - 0.759375).abs() < 1e-6);

        metric.clear();
        for index in 0..5 {
            metric
                .update(
                    &RegressionInput::new(
                        Tensor::from_data(outputs, &device).slice([index..index + 1]),
                        Tensor::from_data(targets, &device).slice([index..index + 1]),
                    ),
                    &MetricMetadata::fake(),
                )
                .unwrap();
        }
        assert!((metric.final_value().current() - 0.759375).abs() < 1e-6);
    }

    #[test]
    fn sklearn_constant_targets_and_single_sample() {
        let device = Default::default();
        let mut metric = R2ScoreMetric::new();
        metric
            .update(
                &RegressionInput::new(
                    Tensor::from_data([[3.], [3.]], &device),
                    Tensor::from_data([[3.], [3.]], &device),
                ),
                &MetricMetadata::fake(),
            )
            .unwrap();
        assert_eq!(metric.final_value().current(), 1.0);
        metric.clear();
        metric
            .update(
                &RegressionInput::new(
                    Tensor::from_data([[3.], [4.]], &device),
                    Tensor::from_data([[3.], [3.]], &device),
                ),
                &MetricMetadata::fake(),
            )
            .unwrap();
        assert_eq!(metric.final_value().current(), 0.0);
        metric.clear();
        metric
            .update(
                &RegressionInput::new(
                    Tensor::from_data([[3.]], &device),
                    Tensor::from_data([[3.]], &device),
                ),
                &MetricMetadata::fake(),
            )
            .unwrap();
        assert!(metric.final_value().current().is_nan());
    }
}
