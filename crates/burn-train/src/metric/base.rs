use std::sync::Arc;

use burn_core::data::dataloader::Progress;
use burn_optim::LearningRate;

/// Metric metadata that can be used when computing metrics.
pub struct MetricMetadata {
    /// The current progress.
    pub progress: Progress,

    /// The global progress of the training (e.g. epochs).
    pub global_progress: Progress,

    /// The current iteration.
    pub iteration: Option<usize>,

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
            global_progress: Progress {
                items_processed: 0,
                items_total: 1,
            },
            iteration: Some(0),
            lr: None,
        }
    }
}

/// Metric id that can be used to compare metrics and retrieve entries of the same metric.
/// For now we take the name as id to make sure that the same metric has the same id across different runs.
#[derive(Debug, Clone, new, PartialEq, Eq, Hash)]
pub struct MetricId {
    /// The metric id.
    id: Arc<String>,
}

/// Metric attributes define the properties intrinsic to different types of metric.
#[derive(Clone, Debug)]
pub enum MetricAttributes {
    /// Numeric attributes.
    Numeric(NumericAttributes),
    /// No attributes.
    None,
}

/// Definition of a metric.
#[derive(Clone, Debug)]
pub struct MetricDefinition {
    /// The metric's id.
    pub metric_id: MetricId,
    /// The name of the metric.
    pub name: String,
    /// The description of the metric.
    pub description: Option<String>,
    /// The attributes of the metric.
    pub attributes: MetricAttributes,
}

impl MetricDefinition {
    /// Create a new metric definition given the metric and a unique id.
    pub fn new<Me: Metric>(metric_id: MetricId, metric: &Me) -> Self {
        Self {
            metric_id,
            name: metric.name().to_string(),
            description: metric.description(),
            attributes: metric.attributes(),
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

    /// A short description of the metric.
    fn description(&self) -> Option<String> {
        None
    }

    /// Attributes of the metric.
    ///
    /// By default, metrics have no attributes.
    fn attributes(&self) -> MetricAttributes {
        MetricAttributes::None
    }

    /// Update the metric state and returns the current metric entry.
    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> SerializedEntry;

    /// Clear the metric state.
    fn clear(&mut self);
}

/// Type used to store metric names efficiently.
pub type MetricName = Arc<String>;

/// Adaptor are used to transform types so that they can be used by metrics.
///
/// This should be implemented by a model's output type for all [metric inputs](Metric::Input) that are
/// registered with the specific learning paradigm (i.e. [SupervisedTraining](crate::SupervisedTraining)).
pub trait Adaptor<T> {
    /// Adapt the type to be passed to a [metric](Metric).
    fn adapt(&self) -> T;
}

impl<T> Adaptor<()> for T {
    fn adapt(&self) {}
}

/// Attributes that describe intrinsic properties of a numeric metric.
#[derive(Clone, Debug)]
pub struct NumericAttributes {
    /// Optional unit (e.g. "%", "ms", "pixels")
    pub unit: Option<String>,
    /// Whether larger values are better (true) or smaller are better (false).
    pub higher_is_better: bool,
}

impl From<NumericAttributes> for MetricAttributes {
    fn from(attr: NumericAttributes) -> Self {
        MetricAttributes::Numeric(attr)
    }
}

impl Default for NumericAttributes {
    fn default() -> Self {
        Self {
            unit: None,
            higher_is_better: true,
        }
    }
}

/// Declare a metric to be numeric.
///
/// This is useful to plot the values of a metric during training.
pub trait Numeric {
    /// Returns the numeric value of the metric.
    fn value(&self) -> NumericEntry;
    /// Returns the current aggregated value of the metric over the global step (epoch).
    fn running_value(&self) -> NumericEntry;
}

/// Serialized form of a metric entry.
#[derive(Debug, Clone, new)]
pub struct SerializedEntry {
    /// The string to be displayed.
    pub formatted: String,
    /// The string to be saved.
    pub serialized: String,
}

/// Data type that contains the current state of a metric at a given time.
#[derive(Debug, Clone)]
pub struct MetricEntry {
    /// Id of the entry's metric.
    pub metric_id: MetricId,
    /// The serialized form of the entry.
    pub serialized_entry: SerializedEntry,
}

impl MetricEntry {
    /// Create a new metric.
    pub fn new(metric_id: MetricId, serialized_entry: SerializedEntry) -> Self {
        Self {
            metric_id,
            serialized_entry,
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
        /// The aggregated value of all entries.
        aggregated_value: f64,
        /// The number of entries present in the aggregated value.
        count: usize,
    },
}

impl NumericEntry {
    /// Gets the current aggregated value of the metric.
    pub fn current(&self) -> f64 {
        match self {
            NumericEntry::Value(val) => *val,
            NumericEntry::Aggregated {
                aggregated_value, ..
            } => *aggregated_value,
        }
    }

    /// Returns a String representing the NumericEntry
    pub fn serialize(&self) -> String {
        match self {
            Self::Value(v) => v.to_string(),
            Self::Aggregated {
                aggregated_value,
                count,
            } => format!("{aggregated_value},{count}"),
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
                        aggregated_value: value,
                        count: numel,
                    }),
                    Err(err) => Err(err.to_string()),
                },
                Err(err) => Err(err.to_string()),
            }
        } else {
            Err("Invalid number of values for numeric entry".to_string())
        }
    }

    /// Compare this numeric metric's value with another one using the specified direction.
    pub fn better_than(&self, other: &NumericEntry, higher_is_better: bool) -> bool {
        (self.current() > other.current()) == higher_is_better
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
