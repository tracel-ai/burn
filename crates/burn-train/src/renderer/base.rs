use std::sync::Arc;

use crate::{
    LearnerSummary,
    logger::{EvaluationProgressLogger, TrainingProgressLogger},
    metric::{MetricDefinition, MetricEntry, NumericEntry},
};

/// Trait for rendering metrics.
pub trait MetricsRendererTraining: Send + Sync {
    /// Updates the training metric state.
    ///
    /// # Arguments
    ///
    /// * `state` - The metric state.
    fn update_train(&mut self, state: MetricState);

    /// Updates the validation metric state.
    ///
    /// # Arguments
    ///
    /// * `state` - The metric state.
    fn update_valid(&mut self, state: MetricState);

    /// Callback method invoked when training ends, whether it
    /// completed successfully or was interrupted.
    ///
    /// # Returns
    ///
    /// A result indicating whether the end-of-training actions were successful.
    fn on_train_end(
        &mut self,
        summary: Option<LearnerSummary>,
    ) -> Result<(), Box<dyn core::error::Error>> {
        default_summary_action(summary);
        Ok(())
    }
}

/// A renderer that can be used for both training and evaluation.
pub trait MetricsRenderer:
    MetricsRendererEvaluation
    + MetricsRendererTraining
    + TrainingProgressLogger
    + EvaluationProgressLogger
{
    /// Keep the renderer from automatically closing, requiring manual action to close it.
    fn manual_close(&mut self);
    /// Register a new metric.
    fn register_metric(&mut self, definition: MetricDefinition);
}

#[derive(Clone)]
/// The name of an evaluation.
///
/// This is going to group metrics together for easier analysis.
pub struct EvaluationName {
    pub(crate) name: Arc<String>,
}

impl EvaluationName {
    /// Creates a new evaluation name.
    pub fn new<S: core::fmt::Display>(s: S) -> Self {
        Self {
            name: Arc::new(format!("{s}")),
        }
    }

    /// Returns the evaluation name.
    pub fn as_str(&self) -> &str {
        &self.name
    }
}

impl core::fmt::Display for EvaluationName {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.name)
    }
}

/// Trait for rendering metrics.
pub trait MetricsRendererEvaluation: Send + Sync {
    /// Updates the testing metric state.
    ///
    /// # Arguments
    ///
    /// * `state` - The metric state.
    fn update_test(&mut self, name: EvaluationName, state: MetricState);

    /// Callback method invoked when testing ends, whether it
    /// completed successfully or was interrupted.
    ///
    /// # Returns
    ///
    /// A result indicating whether the end-of-testing actions were successful.
    fn on_test_end(
        &mut self,
        summary: Option<LearnerSummary>,
    ) -> Result<(), Box<dyn core::error::Error>> {
        default_summary_action(summary);
        Ok(())
    }
}

/// The state of a metric.
#[derive(Debug)]
pub enum MetricState {
    /// A generic metric.
    Generic(MetricEntry),
    /// A numeric metric.
    Numeric(MetricEntry, NumericEntry),
}

fn default_summary_action(summary: Option<LearnerSummary>) {
    if let Some(summary) = summary {
        println!("{summary}");
    }
}
