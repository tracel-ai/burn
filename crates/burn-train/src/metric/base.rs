use std::sync::Arc;

use burn_core::data::dataloader::Progress;
use burn_optim::LearningRate;

/// Metric metadata that can be used when computing metrics.
pub struct MetricMetadata {
    /// The current progress.
    pub progress: Progress,

    /// The current epoch.
    pub epoch: usize,

    /// The total number of epochs.
    pub epoch_total: usize,

    /// The current iteration.
    pub iteration: usize,

    /// The current learning rate.
    pub lr: Option<LearningRate>,
}

impl MetricMetadata {
    /// Fake metric metadata
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
pub trait Metric: Send + Sync + Clone {
    /// The input type of the metric.
    type Input;

    /// The parameterized name of the metric.
    ///
    /// This should be unique, so avoid using short generic names, prefer using the long name.
    ///
    /// For a metric that can exist at different parameters (e.g., top-k accuracy for different
    /// values of k), the name should be unique for each instance.
    fn name(&self) -> MetricName;

    /// Update the metric state and returns the current metric entry.
    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry;
    /// Clear the metric state.
    fn clear(&mut self);
}

/// Type used to store metric names efficiently.
pub type MetricName = Arc<String>;

/// Adaptor are used to transform types so that they can be used by metrics.
///
/// This should be implemented by a model's output type for all [metric inputs](Metric::Input) that are
/// registered with the [learner builder](crate::learner::LearnerBuilder) .
pub trait Adaptor<T> {
    /// Adapt the type to be passed to a [metric](Metric).
    fn adapt(&self) -> T;
}

impl<T> Adaptor<()> for T {
    fn adapt(&self) {}
}

/// Declare a metric to be numeric.
///
/// This is useful to plot the values of a metric during training.
pub trait Numeric {
    /// Returns the numeric value of the metric.
    fn value(&self) -> NumericEntry;
}

/// Data type that contains the current state of a metric at a given time.
#[derive(Debug, Clone)]
pub struct MetricEntry {
    /// The name of the metric.
    pub name: Arc<String>,
    /// The string to be displayed.
    pub formatted: String,
    /// The string to be saved.
    pub serialize: String,
    /// Tags linked to the metric.
    pub tags: Vec<Arc<String>>,
}

impl MetricEntry {
    /// Create a new metric.
    pub fn new(name: Arc<String>, formatted: String, serialize: String) -> Self {
        Self {
            name,
            formatted,
            serialize,
            tags: Vec::new(),
        }
    }
}

/// Numeric metric entry.
#[derive(Debug, Clone)]
pub enum NumericEntry {
    /// Single numeric value.
    Value(f64),
    /// Aggregated numeric (value, number of elements).
    Aggregated {
        /// The sum of all entries.
        sum: f64,
        /// The number of entries present in the sum.
        count: usize,
        /// The current aggregated value.
        current: f64,
    },
}

impl NumericEntry {
    /// Gets the current aggregated value of the metric.
    pub fn current(&self) -> f64 {
        match self {
            NumericEntry::Value(val) => *val,
            NumericEntry::Aggregated { current, .. } => *current,
        }
    }
}

impl NumericEntry {
    pub(crate) fn serialize(&self) -> String {
        match self {
            Self::Value(v) => v.to_string(),
            Self::Aggregated { sum, count, .. } => format!("{sum},{count}"),
        }
    }

    /// De-serializes a string representing a NumericEntry and returns a Result containing the corresponding NumericEntry.
    pub fn deserialize(entry: &str) -> Result<Self, String> {
        // Check for comma separated values
        let values = entry.split(',').collect::<Vec<_>>();
        let num_values = values.len();

        if num_values == 1 {
            // Numeric value
            match values[0].parse::<f64>() {
                Ok(value) => Ok(NumericEntry::Value(value)),
                Err(err) => Err(err.to_string()),
            }
        } else if num_values == 2 {
            // Aggregated numeric (value, number of elements)
            let (value, numel) = (values[0], values[1]);
            match value.parse::<f64>() {
                Ok(value) => match numel.parse::<usize>() {
                    Ok(numel) => Ok(NumericEntry::Aggregated {
                        sum: value,
                        count: numel,
                        current: value,
                    }),
                    Err(err) => Err(err.to_string()),
                },
                Err(err) => Err(err.to_string()),
            }
        } else {
            Err("Invalid number of values for numeric entry".to_string())
        }
    }
}

/// Format a float with the given precision. Will use scientific notation if necessary.
pub fn format_float(float: f64, precision: usize) -> String {
    let scientific_notation_threshold = 0.1_f64.powf(precision as f64 - 1.0);

    match scientific_notation_threshold >= float {
        true => format!("{float:.precision$e}"),
        false => format!("{float:.precision$}"),
    }
}
