use super::{OptimizerMetricContext, OptimizerStepMetric, OptimizerStepOutput};
use burn_core::tensor::TensorReadError;
use std::sync::Arc;

use crate::metric::{
    MetricDefinition, MetricEntry, MetricId, MetricName, NumericAttributes, NumericEntry,
    SerializedEntry,
    store::{MetricsUpdate, NumericMetricUpdate},
};

/// Records the learning rates used for each applied optimizer update.
///
/// Register with [`SupervisedTraining::metrics`](crate::SupervisedTraining::metrics).
/// The base series is named `Learning Rate`; additional groups use `Learning Rate (group N)`
/// in scheduler registration order, starting at one. Plot samples count optimizer updates.
/// The final epoch entry is the last applied rate; an epoch without updates has no entries.
#[derive(Default)]
pub struct LearningRateMetric {
    series: Vec<LearningRateSeries>,
}

struct LearningRateSeries {
    definition: MetricDefinition,
    last: Option<f64>,
}

impl LearningRateMetric {
    /// Create a metric for all learning-rate groups.
    pub fn new() -> Self {
        Self::default()
    }
}

impl OptimizerStepMetric for LearningRateMetric {
    fn name(&self) -> MetricName {
        Arc::new("Learning Rate".into())
    }

    fn init(&mut self, context: &OptimizerMetricContext) -> Vec<MetricDefinition> {
        let num_groups = context.num_learning_rate_groups();
        self.series = (0..num_groups)
            .map(|index| {
                let name = Arc::new(match index {
                    0 => "Learning Rate".to_string(),
                    index => format!("Learning Rate (group {index})"),
                });
                LearningRateSeries {
                    definition: MetricDefinition {
                        metric_id: MetricId::new(name.clone()),
                        name,
                        description: Some("Applied learning rate per optimizer update".into()),
                        attributes: NumericAttributes {
                            unit: None,
                            higher_is_better: false,
                        }
                        .into(),
                    },
                    last: None,
                }
            })
            .collect();
        self.series
            .iter()
            .map(|series| series.definition.clone())
            .collect()
    }

    fn clear(&mut self) {
        for series in &mut self.series {
            series.last = None;
        }
    }

    fn update(&mut self, output: &OptimizerStepOutput) -> Result<MetricsUpdate, TensorReadError> {
        let rates = output.learning_rate().group_rates();
        assert_eq!(
            self.series.len(),
            rates.len(),
            "Learning-rate groups changed during training"
        );
        let entries = self
            .series
            .iter_mut()
            .zip(rates)
            .map(|(series, rate)| {
                series.last = Some(rate);
                series.entry(NumericEntry::Value(rate))
            })
            .collect();
        Ok(MetricsUpdate::new(Vec::new(), entries))
    }

    fn end_epoch(&mut self) -> Result<MetricsUpdate, TensorReadError> {
        let entries = self
            .series
            .iter()
            .filter_map(|series| {
                series
                    .last
                    .map(|rate| series.entry(NumericEntry::Final(rate)))
            })
            .collect();
        Ok(MetricsUpdate::new(Vec::new(), entries))
    }
}

impl LearningRateSeries {
    fn entry(&self, value: NumericEntry) -> NumericMetricUpdate {
        let serialized =
            SerializedEntry::new(format!("{:.2e}", value.current()), value.serialize());
        NumericMetricUpdate::new(
            MetricEntry::new(self.definition.metric_id.clone(), serialized),
            Some(value),
            None,
        )
    }
}
