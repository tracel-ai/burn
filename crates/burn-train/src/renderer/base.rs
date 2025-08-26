use std::sync::Arc;

use crate::metric::{MetricEntry, NumericEntry};
use burn_core::data::dataloader::Progress;

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

    /// Renders the training progress.
    ///
    /// # Arguments
    ///
    /// * `item` - The training progress.
    fn render_train(&mut self, item: TrainingProgress);

    /// Renders the validation progress.
    ///
    /// # Arguments
    ///
    /// * `item` - The validation progress.
    fn render_valid(&mut self, item: TrainingProgress);

    /// Callback method invoked when training ends, whether it
    /// completed successfully or was interrupted.
    ///
    /// # Returns
    ///
    /// A result indicating whether the end-of-training actions were successful.
    fn on_train_end(&mut self) -> Result<(), Box<dyn core::error::Error>> {
        Ok(())
    }
}

/// A renderer that can be used for both training and evaluation.
pub trait MetricsRenderer: MetricsRendererEvaluation + MetricsRendererTraining {
    /// Keep the renderer from automatically closing, requiring manual action to close it.
    fn manual_close(&mut self);
}

#[derive(Clone)]
/// The name of an evaluation.
///
/// This is going to group matrics together for easier analysis.
pub struct EvaluationName {
    pub(crate) name: Arc<String>,
}

impl EvaluationName {
    /// Creates a new metric name.
    pub fn new<S: core::fmt::Display>(s: S) -> Self {
        Self {
            name: Arc::new(format!("{s}")),
        }
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
    /// Renders the testing progress.
    ///
    /// # Arguments
    ///
    /// * `item` - The training progress.
    fn render_test(&mut self, item: EvaluationProgress);

    /// Callback method invoked when testing ends, whether it
    /// completed successfully or was interrupted.
    ///
    /// # Returns
    ///
    /// A result indicating whether the end-of-testing actions were successful.
    fn on_test_end(&mut self) -> Result<(), Box<dyn core::error::Error>> {
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

/// Training progress.
#[derive(Debug)]
pub struct TrainingProgress {
    /// The progress.
    pub progress: Progress,

    /// The epoch.
    pub epoch: usize,

    /// The total number of epochs.
    pub epoch_total: usize,

    /// The iteration.
    pub iteration: usize,
}

/// Evaluation progress.
#[derive(Debug)]
pub struct EvaluationProgress {
    /// The progress.
    pub progress: Progress,

    /// The iteration.
    pub iteration: usize,
}

impl TrainingProgress {
    /// Creates a new empty training progress.
    pub fn none() -> Self {
        Self {
            progress: Progress {
                items_processed: 0,
                items_total: 0,
            },
            epoch: 0,
            epoch_total: 0,
            iteration: 0,
        }
    }
}
