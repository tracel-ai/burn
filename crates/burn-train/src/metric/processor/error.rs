use burn_core::tensor::TensorReadError;

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

/// Every metric failure an event processor reported at once, oldest first.
#[derive(Debug)]
pub struct MetricsError {
    errors: Vec<MetricError>,
}

impl MetricsError {
    /// `Ok` when there is no failure, otherwise every failure as one error.
    pub(crate) fn from_errors(errors: Vec<MetricError>) -> Result<(), Self> {
        match errors.is_empty() {
            true => Ok(()),
            false => Err(Self { errors }),
        }
    }

    /// Append `other`'s failures after this one's.
    pub(crate) fn merge(&mut self, other: Self) {
        self.errors.extend(other.errors);
    }

    /// Every failure, oldest first.
    pub fn errors(&self) -> &[MetricError] {
        &self.errors
    }

    /// Every failure, oldest first.
    pub fn into_errors(self) -> Vec<MetricError> {
        self.errors
    }

    /// Whether any failure comes from a poisoned device, on which every later read fails too.
    pub fn is_device_poisoned(&self) -> bool {
        self.errors.iter().any(|error| {
            matches!(&error.source, TensorReadError::Execution(err) if err.is_device_poisoned())
        })
    }
}

impl core::fmt::Display for MetricsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.errors.as_slice() {
            [error] => write!(f, "A metric failed: {error}"),
            errors => {
                write!(f, "{} metrics failed:", errors.len())?;
                for error in errors {
                    write!(f, "\n  {error}")?;
                }
                Ok(())
            }
        }
    }
}

impl core::error::Error for MetricsError {}
