use burn_core::tensor::TensorReadError;
use burn_std::ExecutionError;
use std::sync::Arc;

use crate::metric::{
    Metric, MetricAttributes, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
    SerializedEntry,
};

/// Registers learning-rate metrics for all scheduler parameter groups.
///
/// Register with [`SupervisedTraining::metrics`](crate::SupervisedTraining::metrics).
/// The base series is named `Learning Rate`; additional groups use `Learning Rate (group N)`
/// in scheduler registration order, starting at one. Plot samples count optimizer updates.
/// The final epoch entry is the last applied rate; an epoch without updates has no entries.
#[derive(Default)]
pub struct LearningRateMetric;

impl LearningRateMetric {
    /// Create a registration for all learning-rate groups.
    pub fn new() -> Self {
        Self
    }
}

#[derive(Clone)]
pub(crate) struct LearningRateGroupMetric {
    group: usize,
    name: MetricName,
    last: Option<f64>,
}

impl LearningRateGroupMetric {
    pub(crate) fn new(group: usize) -> Self {
        Self {
            group,
            name: Arc::new(match group {
                0 => "Learning Rate".into(),
                group => format!("Learning Rate (group {group})"),
            }),
            last: None,
        }
    }

    fn entry(value: NumericEntry) -> SerializedEntry {
        SerializedEntry::new(format!("{:.2e}", value.current()), value.serialize())
    }
}

impl Metric for LearningRateGroupMetric {
    type Input = ();

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn description(&self) -> Option<String> {
        Some("Applied learning rate per optimizer update".into())
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }

    fn update(
        &mut self,
        _: &(),
        metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let rate = metadata
            .lr
            .as_ref()
            .and_then(|rates| rates.group_rates().nth(self.group))
            .ok_or_else(|| ExecutionError::with_context("Missing learning-rate group"))?;
        self.last = Some(rate);
        Ok(Self::entry(NumericEntry::Value(rate)))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        let rate = self.last.ok_or_else(|| {
            ExecutionError::with_context("No learning rate was observed for this group")
        })?;
        Ok(Self::entry(NumericEntry::Final(rate)))
    }

    fn clear(&mut self) {
        self.last = None;
    }
}

impl Numeric for LearningRateGroupMetric {
    fn value(&self) -> Option<NumericEntry> {
        self.last.map(NumericEntry::Value)
    }

    fn running_value(&self) -> Option<NumericEntry> {
        None
    }

    fn final_value(&self) -> NumericEntry {
        NumericEntry::Final(
            self.last
                .expect("compute succeeds only after a rate is observed"),
        )
    }
}
