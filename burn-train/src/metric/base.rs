use burn_core::{data::dataloader::Progress, LearningRate};

/// Metric metadata that can be used when computing metrics.
pub struct MetricMetadata {
    pub progress: Progress,
    pub epoch: usize,
    pub epoch_total: usize,
    pub iteration: usize,
    pub lr: Option<LearningRate>,
}

impl MetricMetadata {
    #[cfg(test)]
    pub fn fake() -> Self {
        Self {
            progress: Progress {
                items_processed: 1,
                items_total: 1,
            },
            epoch: 0,
            epoch_total: 1,
            iteration: 0,
            lr: None,
        }
    }
}

/// Metric trait.
///
/// # Notes
///
/// Implementations should define their own input type only used by the metric.
/// This is important since some conflict may happen when the model output is adapted for each
/// metric's input type.
pub trait Metric: Send + Sync {
    type Input;

    /// Update the metric state and returns the current metric entry.
    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry;
    /// Clear the metric state.
    fn clear(&mut self);
}

/// Adaptor are used to transform types so that they can be used by metrics.
///
/// This should be implemented by a model's output type for all [metric inputs](Metric::Input) that are
/// registed with the [leaner buidler](burn::train::LearnerBuilder).
pub trait Adaptor<T> {
    /// Adapt the type to be passed to a [metric](Metric).
    fn adapt(&self) -> T;
}

/// Declare a metric to be numeric.
///
/// This is usefull to plot the values of a metric during training.
pub trait Numeric {
    fn value(&self) -> f64;
}

/// Data type that contains the current state of a metric at a given time.
#[derive(new)]
pub struct MetricEntry {
    /// The name of the metric.
    pub name: String,
    /// The string to be displayed.
    pub formatted: String,
    /// The string to be saved.
    pub serialize: String,
}
