use super::MetricMetadata;
use super::state::{FormatOptions, NumericMetricState};
use crate::metric::{Metric, MetricAttributes, MetricName, Numeric, SerializedEntry};
use burn_core::tensor::Tensor;

/// The R2 Score (Coefficient of Determination) metric for regression tasks.
#[derive(Clone)]
pub struct R2ScoreMetric {
    name: MetricName,
    state: NumericMetricState,
}

/// The [R2 Score metric](R2ScoreMetric) input type.
#[derive(new)]
pub struct RegressionInput {
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
            name: MetricName::new("R2 Score".to_string()),
            state: Default::default(),
        }
    }
}

impl Metric for R2ScoreMetric {
    type Input = RegressionInput;

    fn update(&mut self, input: &RegressionInput, _metadata: &MetricMetadata) -> SerializedEntry {
        let targets = input.targets.clone();
        let outputs = input.outputs.clone();

        let [batch_size, _] = outputs.dims();

        // Calculate residual sum of squares (SS_res)
        let residuals = targets.clone().sub(outputs);
        let ss_res = residuals.powf_scalar(2.0).sum();

        // Calculate total sum of squares (SS_tot)
        let mean_targets = targets.clone().mean_dim(0);
        let diff_mean = targets.sub(mean_targets);
        let ss_tot = diff_mean.powf_scalar(2.0).sum();

        let r2 = Tensor::ones_like(&ss_tot).sub(ss_res.div(ss_tot.add_scalar(1e-8)));

        self.state.update(
            r2.into_scalar::<f64>(),
            batch_size,
            FormatOptions::new(self.name()).precision(4),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        super::NumericAttributes {
            unit: None,
            higher_is_better: true,
            ..Default::default()
        }
        .into()
    }
}

impl Numeric for R2ScoreMetric {
    fn value(&self) -> super::NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> super::NumericEntry {
        self.state.running_value()
    }
}
