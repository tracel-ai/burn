use burn_core::data::dataloader::Progress;

use crate::metric::MetricEntry;

/// Trait for rendering metrics.
pub trait MetricsRenderer: Send + Sync {
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

    /// Enable manual quit. Default implementation warn that this feature is not implemented.
    ///
    fn enable_manual_quit(&mut self) {
        log::warn!(
            "Manual quit option will be ignored since it's not implemented for this renderer."
        )
    }
}

/// The state of a metric.
#[derive(Debug)]
pub enum MetricState {
    /// A generic metric.
    Generic(MetricEntry),

    /// A numeric metric.
    Numeric(MetricEntry, f64),
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
