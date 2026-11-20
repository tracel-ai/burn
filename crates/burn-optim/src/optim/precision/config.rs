use super::{optimizer::PrecisionOptimizer, storage_mapper::StorageOptimizerStateMapper};
use crate::{ConfigurableOptimizer, Optimizer, OptimizerStateMapper};
use burn_core::tensor::{FloatDType, quantization::QuantScheme};
use serde::{Deserialize, Serialize};

/// Precision for a per-parameter update on unquantized floating-point parameters and gradients.
/// The update device must support the selected computation and storage dtypes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OptimizerPrecision {
    /// Dtype of the parameter, gradient and decoded statistics supplied to the equation.
    pub compute: FloatDType,
    /// Keep the unrounded update for the next step when model and compute dtypes differ.
    pub master_weights: bool,
    /// Representation of statistics between updates. Scalar counters keep their original types.
    pub state: OptimizerStateStorage,
}

/// How the optimizer retains its tensor statistics between calls to `step`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerStateStorage {
    /// Round each updated statistic to the given floating-point dtype on its device.
    Float(FloatDType),
    /// Quantize each statistic dynamically on device; scales are stored with its values.
    Quantized(QuantScheme),
}

impl OptimizerPrecision {
    /// Apply this policy to a bare optimizer for use with a module or parameter group.
    /// Module gradient clipping runs after conversion to the computation dtype.
    pub fn wrap<O: ConfigurableOptimizer>(self, optimizer: O) -> impl Optimizer {
        self.wrap_with_storage(optimizer, StorageOptimizerStateMapper::new(self.state))
    }
}

impl OptimizerPrecision {
    /// Use a custom storage mapper to choose representations for individual state components.
    /// The mapper receives stable component-qualified leaf names; it must preserve shape and device.
    /// A fresh clone of the mapper is used per parameter update; persistent metadata belongs
    /// in the returned tensor. This replaces `state`. Checkpoints retain the encoded tensor representation. Supply the same mapper when resuming.
    pub fn wrap_with_storage<O, M>(self, optimizer: O, storage: M) -> impl Optimizer
    where
        O: ConfigurableOptimizer,
        M: OptimizerStateMapper + Clone + Send + Sync + 'static,
    {
        PrecisionOptimizer::new(optimizer, self, storage)
    }
}
