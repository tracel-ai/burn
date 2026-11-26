use super::{master::MasterWeights, optimizer::PrecisionOptimizer};
use crate::{Optimizer, OptimizerStateStorage};
use burn_core::tensor::FloatDType;
use serde::{Deserialize, Serialize};

/// Arithmetic and statistics storage for a per-parameter optimizer.
/// The update device must support the selected representations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizerPrecision<S = FloatDType> {
    compute: FloatDType,
    storage: S,
}

impl OptimizerPrecision {
    /// Evaluate the equation and retain its statistics in `compute`.
    pub const fn new(compute: FloatDType) -> Self {
        Self {
            compute,
            storage: compute,
        }
    }
}

impl<S> OptimizerPrecision<S> {
    /// Choose how to retain statistics between steps. A [`FloatDType`] casts all floating-point
    /// statistics to one dtype. A [`crate::OptimizerStateMapper`] can select individual tensor leaves.
    /// [`crate::AdamMomentStorage`] composes statistic codecs for Adam and AdamW.
    /// An [`OptimizerStateStorage`] can supply its own checkpoint record and decoder.
    pub fn with_state_storage<T>(self, storage: T) -> OptimizerPrecision<T> {
        OptimizerPrecision {
            compute: self.compute,
            storage,
        }
    }

    /// Start each update from the model parameter. Round the result back to its original dtype.
    pub fn wrap<O>(self, optimizer: O) -> impl Optimizer
    where
        O: Optimizer,
        S: OptimizerStateStorage<O>,
    {
        PrecisionOptimizer::new(optimizer, self.compute, self.storage)
    }

    /// Accumulate updates in the compute dtype. When the model has a different dtype,
    /// retain the unrounded parameter on device and in the optimizer checkpoint.
    /// Reset the optimizer state when replacing weights outside the optimizer.
    pub fn wrap_with_master_weights<O>(self, optimizer: O) -> impl Optimizer
    where
        O: Optimizer,
        S: OptimizerStateStorage<O>,
    {
        let dtype = self.compute;
        MasterWeights::new(self.wrap(optimizer), dtype)
    }
}
