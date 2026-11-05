use burn_core::tensor::TensorReadError;
use burn_std::ExecutionError;

use crate::metric::{MetricName, store::Split};

/// A metric that could not process an event.
#[derive(Debug)]
pub struct MetricError {
    /// The name of the metric that failed.
    pub metric: MetricName,
    /// The split during which the error occurred.
    pub split: Split,
    /// The source error.
    pub source: TensorReadError,
}

impl core::fmt::Display for MetricError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.split {
            Split::Test(Some(name)) => write!(f, "test/{name}/{}: {}", self.metric, self.source),
            split => write!(f, "{split}/{}: {}", self.metric, self.source),
        }
    }
}

impl core::error::Error for MetricError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// A failure an event processor reported.
#[derive(Debug)]
pub enum EventProcessorFailure {
    /// Error while evaluating metric.
    Metric(MetricError),
    /// An event could not be synced.
    Sync {
        /// The split the event belonged to.
        split: Split,
        /// The source error.
        source: ExecutionError,
    },
}

impl EventProcessorFailure {
    /// Whether the failure comes from a poisoned device.
    pub fn is_device_poisoned(&self) -> bool {
        match self {
            Self::Metric(error) => {
                matches!(&error.source, TensorReadError::Execution(err) if err.is_device_poisoned())
            }
            Self::Sync { source, .. } => source.is_device_poisoned(),
        }
    }
}

impl core::fmt::Display for EventProcessorFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Metric(error) => write!(f, "{error}"),
            Self::Sync {
                split: Split::Test(Some(name)),
                source,
            } => write!(f, "test/{name}: the event could not be synced: {source}"),
            Self::Sync { split, source } => {
                write!(f, "{split}: the event could not be synced: {source}")
            }
        }
    }
}

impl core::error::Error for EventProcessorFailure {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Metric(error) => Some(error),
            Self::Sync { source, .. } => Some(source),
        }
    }
}

impl From<MetricError> for EventProcessorFailure {
    fn from(error: MetricError) -> Self {
        Self::Metric(error)
    }
}

/// Every failure an event processor reported at once, oldest first.
#[derive(Debug)]
pub struct EventProcessorError {
    failures: Vec<EventProcessorFailure>,
}

impl EventProcessorError {
    /// `Ok` when no metric failed, otherwise every metric failure as one error.
    pub(crate) fn from_errors(errors: Vec<MetricError>) -> Result<(), Self> {
        match errors.is_empty() {
            true => Ok(()),
            false => Err(Self {
                failures: errors
                    .into_iter()
                    .map(EventProcessorFailure::Metric)
                    .collect(),
            }),
        }
    }

    /// An event of `split` could not be synced.
    pub(crate) fn sync(split: Split, source: ExecutionError) -> Self {
        Self {
            failures: vec![EventProcessorFailure::Sync { split, source }],
        }
    }

    /// Append `other`'s failures after this one's.
    pub(crate) fn merge(&mut self, other: Self) {
        self.failures.extend(other.failures);
    }

    /// Every failure, oldest first.
    pub fn failures(&self) -> &[EventProcessorFailure] {
        &self.failures
    }

    /// Every failure, oldest first.
    pub fn into_failures(self) -> Vec<EventProcessorFailure> {
        self.failures
    }

    /// Whether any failure comes from a poisoned device, on which every later read fails too.
    pub fn is_device_poisoned(&self) -> bool {
        self.failures
            .iter()
            .any(EventProcessorFailure::is_device_poisoned)
    }
}

impl core::fmt::Display for EventProcessorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.failures.as_slice() {
            [failure] => write!(f, "Event processing failed: {failure}"),
            failures => {
                write!(f, "Event processing failed {} times:", failures.len())?;
                for failure in failures {
                    write!(f, "\n  {failure}")?;
                }
                Ok(())
            }
        }
    }
}

impl core::error::Error for EventProcessorError {}
