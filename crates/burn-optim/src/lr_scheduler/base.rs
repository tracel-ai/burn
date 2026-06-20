use alloc::collections::BTreeMap;
pub(super) use alloc::string::String;
use alloc::vec::Vec;
use burn_core as burn;

use crate::{RecordState, StateSink, StateSource, join_path};
use burn::store::RecordError;
use burn::tensor::{Bytes, Device};
use burn_pack::{Reader, Scalar, Writer};

use crate::LearningRate;

/// Learning rate scheduler defines how the learning rate will evolve during training.
pub trait LrScheduler: Clone + Send + Sync {
    /// Perform the scheduler step, potentially updating its state, and returning the effective
    /// learning rate.
    fn step(&mut self) -> LearningRate;

    /// Get the current state of the scheduler as a [record](LrSchedulerRecord).
    fn to_record(&self) -> LrSchedulerRecord;

    /// Load the state of the scheduler from a [record](LrSchedulerRecord).
    fn load_record(self, record: LrSchedulerRecord) -> Self;
}

/// The serialized state of a [learning rate scheduler](LrScheduler), stored in the
/// [burnpack](burn_pack) format.
///
/// Scheduler state is just a handful of scalars (step counters, current learning rate), so the
/// record holds named typed scalars and no tensors. Composed schedulers nest their children's
/// records under an index prefix via [`with_record`](Self::with_record) / [`record`](Self::record).
#[derive(Default, Clone, Debug)]
pub struct LrSchedulerRecord {
    scalars: BTreeMap<String, Scalar>,
}

impl LrSchedulerRecord {
    /// Create an empty record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the record holds no scalars.
    pub fn is_empty(&self) -> bool {
        self.scalars.is_empty()
    }

    /// Store a scalar under `key`.
    pub fn with_scalar<V: Into<Scalar>>(mut self, key: &str, value: V) -> Self {
        self.scalars.insert(String::from(key), value.into());
        self
    }

    /// Read the scalar stored under `key`, if present and of a compatible type.
    pub fn scalar<V: TryFrom<Scalar>>(&self, key: &str) -> Option<V> {
        self.scalars
            .get(key)
            .copied()
            .and_then(|scalar| V::try_from(scalar).ok())
    }

    /// Merge a child `record`'s scalars under `prefix` (used to compose schedulers).
    pub fn with_record(mut self, prefix: &str, record: LrSchedulerRecord) -> Self {
        for (key, value) in record.scalars {
            self.scalars.insert(join_path(prefix, &key), value);
        }
        self
    }

    /// Extract the child record previously merged under `prefix`.
    pub fn record(&self, prefix: &str) -> LrSchedulerRecord {
        let head = join_path(prefix, "");
        let scalars = self
            .scalars
            .iter()
            .filter_map(|(key, value)| {
                key.strip_prefix(&head)
                    .map(|stripped| (String::from(stripped), *value))
            })
            .collect();
        LrSchedulerRecord { scalars }
    }

    /// Build a record from a scheduler [state](RecordState).
    ///
    /// Scheduler state is scalar-only, so this reuses the same [`RecordState`] decomposition as
    /// optimizer states (it panics in debug builds if a tensor leaf is produced).
    pub fn from_state<S: RecordState>(state: &S) -> Self {
        let mut sink = StateSink::default();
        state.state_flatten("", &mut sink);
        debug_assert!(
            sink.tensors.is_empty(),
            "learning rate scheduler state is expected to be scalar-only"
        );
        Self {
            scalars: sink.scalars.into_iter().collect(),
        }
    }

    /// Reconstruct a scheduler [state](RecordState) from this record.
    ///
    /// Uses the default device; scheduler state is scalar-only so no tensor is ever allocated.
    pub fn into_state<S: RecordState>(&self) -> Option<S> {
        let mut source = StateSource::new(self.scalars.clone());
        S::state_unflatten("", &mut source, &Device::default())
    }

    /// Serialize the record to an in-memory burnpack byte buffer.
    pub fn into_bytes(self) -> Result<Bytes, RecordError> {
        Ok(self.into_writer().into_bytes()?)
    }

    /// Reconstruct a record from an in-memory burnpack byte buffer.
    pub fn from_bytes(bytes: Bytes) -> Result<Self, RecordError> {
        let reader = Reader::from_bytes(bytes)?;
        Ok(Self {
            scalars: reader.scalars().clone(),
        })
    }

    /// Save the record to a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn save<P: AsRef<std::path::Path>>(self, path: P) -> Result<(), RecordError> {
        self.into_writer().write_to_file(path)?;
        Ok(())
    }

    /// Load the record from a burnpack file on disk.
    #[cfg(feature = "std")]
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, RecordError> {
        let reader = Reader::from_file(path)?;
        Ok(Self {
            scalars: reader.scalars().clone(),
        })
    }

    fn into_writer(self) -> Writer {
        let mut writer = Writer::new(Vec::new());
        for (key, value) in &self.scalars {
            writer = writer.with_scalar(key, *value);
        }
        writer
    }
}

#[cfg(test)]
pub(super) mod test_utils {
    use super::*;

    // A small tolerance for learning rate comparisons. Depending on how learning rates are
    // computed, floating-point arithmetic error might exceed f64::EPSILON, so a larger value is
    // used here.
    const LOOSE_EPSILON: LearningRate = 1e-10;

    pub fn check_lr_sequence<I, S>(mut scheduler: S, expected_lrs: I)
    where
        I: IntoIterator<Item = LearningRate>,
        S: LrScheduler,
    {
        expected_lrs
            .into_iter()
            .enumerate()
            .for_each(|(i, expected)| {
                let lr = scheduler.step();
                assert!(
                    (lr - expected).abs() < LOOSE_EPSILON,
                    "Scheduled learning rate {lr} is not approximately equal to the expected value \
                     {expected} at step {i}",
                );
            });
    }

    // save_at_step is the number of steps to run the scheduler before saving and loading back its
    // state.
    pub fn check_save_load<S>(mut scheduler: S, save_at_step: usize)
    where
        S: Clone + LrScheduler,
    {
        let mut truth = scheduler.clone();
        // Consume some steps before saving and loading back
        (0..save_at_step).for_each(|_| {
            truth.step();
            scheduler.step();
        });
        let rec = scheduler.to_record();
        scheduler = scheduler.load_record(rec);

        // Validate that the scheduler resumes from where it left off.
        compare_steps(&mut scheduler, &mut truth, save_at_step);
    }

    // Check if two schedulers produce the same learning rate sequences over the specified number of
    // steps.
    pub fn compare_steps<S: LrScheduler>(a: &mut S, b: &mut S, num_steps: usize) {
        (0..num_steps).for_each(|i| {
            let lr_a = a.step();
            let lr_b = b.step();
            assert!(
                (lr_a - lr_b).abs() < LOOSE_EPSILON,
                "The two learning rates ({lr_a}, {lr_b}) at position {i} in the remaining \
                 sequences are not approximately equal",
            );
        });
    }
}
