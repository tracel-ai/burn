use burn_core::tensor::TensorReadError;

use super::{OptimizerMetricContext, OptimizerStepMetric, OptimizerStepOutput};
use crate::metric::{
    MetricDefinition,
    processor::MetricError,
    store::{MetricsUpdate, Split},
};

#[derive(Default)]
pub(crate) struct OptimizerStepMetrics {
    metrics: Vec<Box<dyn OptimizerStepMetric>>,
}

impl OptimizerStepMetrics {
    pub(crate) fn register<Me: OptimizerStepMetric>(&mut self, metric: Me) {
        self.metrics.push(Box::new(metric));
    }

    pub(crate) fn init(&mut self, context: &OptimizerMetricContext) -> Vec<MetricDefinition> {
        self.metrics
            .iter_mut()
            .flat_map(|metric| metric.init(context))
            .collect()
    }

    pub(crate) fn clear(&mut self) {
        for metric in &mut self.metrics {
            metric.clear();
        }
    }

    pub(crate) fn update(
        &mut self,
        observations: &OptimizerStepOutput,
    ) -> (MetricsUpdate, Vec<MetricError>) {
        self.collect(|metric| metric.update(observations))
    }

    pub(crate) fn end_epoch(&mut self) -> (MetricsUpdate, Vec<MetricError>) {
        self.collect(|metric| metric.end_epoch())
    }

    fn collect(
        &mut self,
        mut report: impl FnMut(&mut dyn OptimizerStepMetric) -> Result<MetricsUpdate, TensorReadError>,
    ) -> (MetricsUpdate, Vec<MetricError>) {
        let mut update: Option<MetricsUpdate> = None;
        let mut failures = Vec::new();
        for metric in &mut self.metrics {
            match report(metric.as_mut()) {
                Ok(mut next) => match &mut update {
                    Some(update) => {
                        update.entries.append(&mut next.entries);
                        update.entries_numeric.append(&mut next.entries_numeric);
                    }
                    None => update = Some(next),
                },
                Err(source) => failures.push(MetricError {
                    metric: metric.name(),
                    split: Split::Train,
                    source,
                }),
            }
        }
        (
            update.unwrap_or_else(|| MetricsUpdate::new(Vec::new(), Vec::new())),
            failures,
        )
    }
}
