use burn_tensor::Device;

pub use crate::data::dataset::{Dataset, DatasetIterator};
use burn_dataset::DatasetError;
use core::iter::Iterator;
use std::sync::Arc;

/// The expected number of items in a workload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TotalCount {
    /// A known output count.
    Exact(usize),
    /// An approximate output count.
    Estimated(usize),
    /// The output count is not known.
    Unknown,
}

impl TotalCount {
    /// Returns the count when one is available.
    pub fn value(self) -> Option<usize> {
        match self {
            Self::Exact(count) | Self::Estimated(count) => Some(count),
            Self::Unknown => None,
        }
    }
}

impl From<usize> for TotalCount {
    fn from(value: usize) -> Self {
        Self::Exact(value)
    }
}

impl std::iter::Sum for TotalCount {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::Exact(0), |sum, count| match (sum, count) {
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::Exact(a), Self::Exact(b)) => Self::Exact(a + b),
            (a, b) => Self::Estimated(a.value().unwrap() + b.value().unwrap()),
        })
    }
}

impl core::fmt::Display for TotalCount {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Exact(count) => write!(f, "{count}"),
            Self::Estimated(count) => write!(f, "~{count}"),
            Self::Unknown => f.write_str("?"),
        }
    }
}

/// A progress snapshot with a processed item count and an expected total.
///
/// Counts describe work observed, not execution state. A data loader is exhausted when its
/// iterator returns `None`; training and evaluation report their own terminal outcomes.
#[derive(Clone, Debug)]
pub struct Progress {
    /// The number of items that have been processed.
    pub items_processed: usize,

    /// The expected number of processed items.
    pub total: TotalCount,

    /// The unit of measurement for the progress (e.g., "items", "batches").
    pub unit: Option<String>,
}

impl Progress {
    /// Creates a progress snapshot.
    pub fn new(items_processed: usize, total: impl Into<TotalCount>, unit: Option<String>) -> Self {
        Self {
            items_processed,
            total: total.into(),
            unit,
        }
    }

    /// Resolves an uncertain total to the observed count once no more items will be emitted.
    /// An exact total retains its declared value. This does not indicate execution success.
    pub fn resolve_total(&mut self) {
        if !matches!(self.total, TotalCount::Exact(_)) {
            self.total = TotalCount::Exact(self.items_processed);
        }
    }

    /// Combines progress snapshots with the same unit.
    /// An empty collection produces a snapshot with an exact zero total.
    pub fn aggregate<'a>(progresses: impl IntoIterator<Item = &'a Self>) -> Self {
        let mut iter = progresses.into_iter();
        let Some(first) = iter.next() else {
            return Self::new(0, TotalCount::Exact(0), None);
        };
        iter.fold(first.clone(), |mut result, progress| {
            result.items_processed += progress.items_processed;
            result.total = [result.total, progress.total].into_iter().sum();
            result
        })
    }
}

impl core::fmt::Display for Progress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.items_processed)?;
        if self.total.value().is_some() {
            write!(f, "/{}", self.total)?;
        }
        Ok(())
    }
}

/// A data loader iterator that can be used to iterate over a data loader.
///
/// Yields `Err` instead of panicking when a dataset retrieval fails, so callers can decide
/// how to handle it (e.g. abort the training run with a clear message).
pub trait DataLoaderIterator<O>: Iterator<Item = Result<O, DatasetError>> {
    /// Returns the progress of the data loader.
    fn progress(&self) -> Progress;
}

/// A data loader that can be used to iterate over a dataset.
pub trait DataLoader<O>: Send + Sync {
    /// Returns a boxed [iterator](DataLoaderIterator) to iterate over the data loader.
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<O> + 'a>;

    /// The number of items in the input dataset.
    ///
    /// The number of items yielded depends on the batching strategy.
    fn num_items(&self) -> usize;

    /// The expected number of constituent items yielded by a full iteration, not the number of batches.
    fn output_items(&self) -> TotalCount {
        TotalCount::Unknown
    }

    /// Move the data loader to the given device, ensuring the batches are assigned to the correct device.
    fn to_device(&self, device: &Device) -> Arc<dyn DataLoader<O>>;

    /// Returns a new data loader containing a subset of the data.
    ///
    /// The subset includes items from `start` (inclusive) to `end` (exclusive),
    /// preserving the batch size and ordering of the original data loader.
    ///
    /// # Arguments
    ///
    /// * `start` - The starting index of the subset (inclusive).
    /// * `end` - The ending index of the subset (exclusive).
    ///
    /// # Returns
    ///
    /// A boxed [`DataLoader`] instance containing only the specified range.
    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<O>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use TotalCount::*;

    #[rstest::rstest]
    #[case::estimated(Estimated(1), Exact(2))]
    #[case::unknown(Unknown, Exact(2))]
    #[case::exact(Exact(3), Exact(3))]
    fn resolves_uncertain_totals_without_changing_exact_counts(
        #[case] total: TotalCount,
        #[case] expected: TotalCount,
    ) {
        let mut progress = Progress::new(2, total, None);
        progress.resolve_total();
        assert_eq!(progress.total, expected);
        assert_eq!(progress.items_processed, 2);
    }

    #[rstest::rstest]
    #[case::exact([Exact(2), Exact(3)], Exact(5))]
    #[case::estimated_last([Exact(2), Estimated(3)], Estimated(5))]
    #[case::estimated_first([Estimated(2), Exact(3)], Estimated(5))]
    #[case::unknown_last([Estimated(2), Unknown], Unknown)]
    #[case::unknown_first([Unknown, Exact(3)], Unknown)]
    fn aggregate_preserves_total_reliability(
        #[case] totals: [TotalCount; 2],
        #[case] expected: TotalCount,
    ) {
        let parts = [
            Progress::new(2, totals[0], None),
            Progress::new(3, totals[1], None),
        ];
        let progress = Progress::aggregate(&parts);
        assert_eq!(progress.items_processed, 5);
        assert_eq!(progress.total, expected);
    }
}
