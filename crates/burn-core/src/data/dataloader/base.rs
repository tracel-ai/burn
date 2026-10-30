use burn_tensor::Device;

pub use crate::data::dataset::{Dataset, DatasetIterator};
use burn_dataset::DatasetError;
use core::iter::Iterator;
use std::sync::Arc;

/// A progress snapshot with a processed item count and an expected total.
///
/// Counts describe work observed, not execution state. A data loader is exhausted when its
/// iterator returns `None`. Reaching the expected total does not establish exhaustion or success.
#[derive(Clone, Debug)]
pub struct Progress {
    /// The number of items processed, measured in `unit`. Data loaders count items in yielded batches.
    pub items_processed: usize,

    /// The exact total number of items, or `None` if it is not known.
    pub total: Option<usize>,

    /// The unit of measurement for the progress (e.g., "items", "batches").
    pub unit: Option<String>,
}

impl Progress {
    /// Creates a progress snapshot.
    pub fn new(
        items_processed: usize,
        total: impl Into<Option<usize>>,
        unit: Option<String>,
    ) -> Self {
        Self {
            items_processed,
            total: total.into(),
            unit,
        }
    }

    /// Resolves an unknown total to the observed count once no more items will be emitted.
    /// A known total retains its declared value. This does not indicate execution success.
    pub fn resolve_total(&mut self) {
        if self.total.is_none() {
            self.total = Some(self.items_processed);
        }
    }

    /// Combines progress snapshots with the same unit.
    /// An empty collection produces a snapshot with an exact zero total.
    pub fn aggregate<P: core::borrow::Borrow<Self>>(
        progresses: impl IntoIterator<Item = P>,
    ) -> Self {
        let mut iter = progresses.into_iter();
        let Some(first) = iter.next() else {
            return Self::new(0, Some(0), None);
        };
        iter.fold(first.borrow().clone(), |mut result, progress| {
            let progress = progress.borrow();
            result.items_processed += progress.items_processed;
            result.total = [result.total, progress.total].into_iter().sum();
            result
        })
    }
}

impl core::fmt::Display for Progress {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.items_processed)?;
        if let Some(total) = self.total {
            write!(f, "/{total}")?;
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

    /// Returns the exact number of samples yielded over a complete pass, if known.
    ///
    /// Counts samples within batches, not batches. This can differ from [`Self::num_items`]
    /// when the batching strategy drops or repeats samples. `Some(0)` means no samples
    /// will be yielded; `None` means the count cannot be determined in advance.
    /// Querying this count must not consume samples or advance the loader's random state.
    /// Iterator progress uses the same counting unit and may resolve an unknown total.
    fn expected_num_items(&self) -> Option<usize> {
        None
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

    #[test]
    fn resolves_unknown_total_to_processed_count() {
        let mut progress = Progress::new(2, None, None);
        progress.resolve_total();
        assert_eq!(progress.total, Some(2));
    }

    #[test]
    fn preserves_known_total_when_resolving() {
        let mut progress = Progress::new(2, 3, None);
        progress.resolve_total();
        assert_eq!(progress.total, Some(3));
    }

    #[test]
    fn aggregates_known_totals() {
        let parts = [Progress::new(2, 4, None), Progress::new(3, 6, None)];
        let progress = Progress::aggregate(&parts);
        assert_eq!(progress.items_processed, 5);
        assert_eq!(progress.total, Some(10));
    }

    #[test]
    fn aggregate_preserves_unknown_totals() {
        let progress =
            Progress::aggregate([Progress::new(2, 4, None), Progress::new(3, None, None)]);
        assert_eq!(progress.items_processed, 5);
        assert_eq!(progress.total, None);
    }
}
