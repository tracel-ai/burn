use burn_core::prelude::{Backend, Bool, Tensor};
use burn_core::tensor::cast::ToElement;
use burn_core::tensor::Int;
use burn_core::{data::dataloader::Progress, LearningRate};
use strum::EnumIter;

///Aggregation types for Classification metrics
#[derive(EnumIter, Copy, Clone, Debug)]
pub enum AggregationType {
    Micro,
    Macro,
    //Weighted(Box<[f64]>), todo!()
}

impl AggregationType {
    pub fn aggregate<B: Backend>(self, cm_mask: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        match self {
            AggregationType::Macro => cm_mask.sum_dim(0).squeeze(0).float(),
            AggregationType::Micro => cm_mask.sum().float(), //MetricAverage::Weighted(weights) => Left(metric.float().sum_dim(0).squeeze(0) * Tensor::from_floats(weights.deref(), &B::Device::default())) todo!()
        }
    }

    pub fn aggregate_mean<B: Backend>(self, cm_mask: Tensor<B, 2, Int>) -> Tensor<B, 1> {
        match self {
            AggregationType::Macro => cm_mask.float().mean_dim(0).squeeze(0),
            AggregationType::Micro => cm_mask.float().mean(), //MetricAverage::Weighted(weights) => Left(metric.float().sum_dim(0).squeeze(0) * Tensor::from_floats(weights.deref(), &B::Device::default())) todo!()
        }
    }

    pub fn to_averaged_tensor<B: Backend>(self, mut metrics: Tensor<B, 1>) -> Tensor<B, 1> {
        match self {
            AggregationType::Macro => {
                if metrics.contains_nan().any().into_scalar() {
                    let nan_mask = metrics.is_nan();
                    metrics = metrics
                        .clone()
                        .select(0, nan_mask.bool_not().argwhere().squeeze(1))
                }
                metrics.mean()
            }
            AggregationType::Micro => metrics,
            //MetricAverage::Weighted(weights) =>
        }
    }

    pub fn to_averaged_metric<B: Backend>(self, metrics: Tensor<B, 1>) -> f64 {
        self.to_averaged_tensor(metrics).into_scalar().to_f64()
    }
}

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
    /// The name of the metric.
    ///
    /// This should be unique, so avoid using short generic names, prefer using the long name.
    const NAME: &'static str;

    /// The input type of the metric.
    type Input;

    /// Update the metric state and returns the current metric entry.
    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry;
    /// Clear the metric state.
    fn clear(&mut self);
}

/// The [classification metric](ClassificationMetric) input type.
#[derive(new, Debug)]
pub struct ClassificationInput<B: Backend> {
    /// Sample x Class Non thresholded predictions
    pub predictions: Tensor<B, 2>,
    /// Sample x Class target mask
    pub targets: Tensor<B, 2, Bool>,
}

/// Adaptor are used to transform types so that they can be used by metrics.
///
/// This should be implemented by a model's output type for all [metric inputs](Metric::Input) that are
/// registered with the [leaner buidler](crate::learner::LearnerBuilder) .
pub trait Adaptor<T> {
    /// Adapt the type to be passed to a [metric](Metric).
    fn adapt(&self) -> T;
}

/// Declare a metric to be numeric.
///
/// This is useful to plot the values of a metric during training.
pub trait Numeric {
    /// Returns the numeric value of the metric.
    fn value(&self) -> f64;
}

/// Data type that contains the current state of a metric at a given time.
#[derive(new, Debug, Clone)]
pub struct MetricEntry {
    /// The name of the metric.
    pub name: String,
    /// The string to be displayed.
    pub formatted: String,
    /// The string to be saved.
    pub serialize: String,
}

/// Numeric metric entry.
pub enum NumericEntry {
    /// Single numeric value.
    Value(f64),
    /// Aggregated numeric (value, number of elements).
    Aggregated(f64, usize),
}

impl NumericEntry {
    pub(crate) fn serialize(&self) -> String {
        match self {
            Self::Value(v) => v.to_string(),
            Self::Aggregated(v, n) => format!("{v},{n}"),
        }
    }

    pub(crate) fn deserialize(entry: &str) -> Result<Self, String> {
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
                    Ok(numel) => Ok(NumericEntry::Aggregated(value, numel)),
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
