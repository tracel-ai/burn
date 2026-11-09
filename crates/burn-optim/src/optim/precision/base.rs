use burn::tensor::{Device, FloatDType, Tensor};
use burn_core as burn;

use super::{
    OptimizerPrecisionConfig, OptimizerState, OptimizerStateMapper, ParameterUpdate,
    StateTensorKind,
};
use crate::{LearningRate, Optimizer, RecordState};

/// An optimizer with shared computation, parameter, and state-storage precision policies.
///
/// Construct with [`Optimizer::with_precision`], then use directly or through
/// [`crate::ModuleOptimizer`], including its parameter groups. The inner optimizer receives
/// floating tensors in computation precision; storage conversions happen after its update.
#[derive(Clone)]
pub struct PrecisionOptimizer<O> {
    optimizer: O,
    config: OptimizerPrecisionConfig,
}

impl<O: Optimizer> PrecisionOptimizer<O> {
    pub(crate) fn new(optimizer: O, config: OptimizerPrecisionConfig) -> Self {
        Self { optimizer, config }
    }

    /// Inspect the configured precision policies.
    pub fn config(&self) -> &OptimizerPrecisionConfig {
        &self.config
    }
}

/// Algorithm statistics and optional authoritative parameters for a precision-aware optimizer.
///
/// Save this state together with the corresponding model. Discard it when replacing model
/// parameters or changing precision policies. Field names and dtype codes are part of its record.
#[derive(Clone, RecordState)]
pub struct PrecisionState<S: OptimizerState, const D: usize> {
    algorithm: Option<S>,
    master: Option<Tensor<D>>,
    model_dtype: u8,
    compute_dtype: u8,
    storage_dtype: u8,
    master_weights: bool,
}

impl<S: OptimizerState, const D: usize> OptimizerState for PrecisionState<S, D> {
    fn map_tensors<M: OptimizerStateMapper>(mut self, mapper: &mut M) -> Self {
        self.algorithm = self.algorithm.map(|state| state.map_tensors(mapper));
        self.master = self
            .master
            .map(|tensor| mapper.map(tensor, StateTensorKind::FullPrecision));
        self
    }
}

struct StateConversion {
    compute: FloatDType,
    storage: FloatDType,
}

impl OptimizerStateMapper for StateConversion {
    fn map<const D: usize>(&mut self, tensor: Tensor<D>, kind: StateTensorKind) -> Tensor<D> {
        let dtype = match kind {
            StateTensorKind::FullPrecision => self.compute,
            StateTensorKind::Signed | StateTensorKind::NonNegative => self.storage,
        };
        tensor.cast(dtype)
    }
}

// Explicit codes keep records independent of Rust's enum discriminants.
fn dtype_code(dtype: FloatDType) -> u8 {
    match dtype {
        FloatDType::F16 => 0,
        FloatDType::BF16 => 1,
        FloatDType::F32 => 2,
        FloatDType::F64 => 3,
        FloatDType::Flex32 => 4,
    }
}

impl<O: Optimizer> Optimizer for PrecisionOptimizer<O> {
    type State<const D: usize> = PrecisionState<O::State<D>, D>;

    fn gradient_dtype(&self, parameter_dtype: FloatDType) -> Option<FloatDType> {
        Some(self.config.compute.resolve(parameter_dtype))
    }

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let model_dtype: FloatDType = tensor.dtype().into();
        let compute_dtype = self.config.compute.resolve(model_dtype);
        let storage_dtype = self.config.state_storage.resolve(compute_dtype);
        let master_weights = self.config.parameter_update == ParameterUpdate::MasterWeights;
        let retain_master = master_weights && compute_dtype != model_dtype;

        let (parameter, algorithm) = match state {
            Some(state) => {
                assert!(
                    (
                        state.model_dtype,
                        state.compute_dtype,
                        state.storage_dtype,
                        state.master_weights
                    ) == (
                        dtype_code(model_dtype),
                        dtype_code(compute_dtype),
                        dtype_code(storage_dtype),
                        master_weights
                    ),
                    "Optimizer precision differs from its saved state; reset optimizer state before changing precision"
                );
                assert_eq!(
                    state.master.is_some(),
                    retain_master,
                    "Optimizer master-weight state is missing or incompatible"
                );
                let parameter = match state.master {
                    Some(master) => {
                        assert_eq!(
                            master.dims(),
                            tensor.dims(),
                            "Optimizer master weights must match the model shape; reset state after replacing parameters"
                        );
                        master
                    }
                    None => tensor.cast(compute_dtype),
                };
                (parameter, state.algorithm)
            }
            None => {
                let device = tensor.device();
                assert!(
                    device.supports_dtype(compute_dtype),
                    "Optimizer computation dtype {compute_dtype:?} is unsupported on {device:?}"
                );
                assert!(
                    device.supports_dtype(storage_dtype),
                    "Optimizer state dtype {storage_dtype:?} is unsupported on {device:?}"
                );
                (tensor.cast(compute_dtype), None)
            }
        };

        let algorithm = algorithm.map(|state| {
            state.map_tensors(&mut StateConversion {
                compute: compute_dtype,
                storage: compute_dtype,
            })
        });
        let (updated, algorithm) =
            self.optimizer
                .step(lr, parameter, grad.cast(compute_dtype), algorithm);
        // The update consumes the newly computed statistics before storage rounding.
        let algorithm = algorithm.map(|state| {
            state.map_tensors(&mut StateConversion {
                compute: compute_dtype,
                storage: storage_dtype,
            })
        });
        let master = retain_master.then(|| updated.clone());
        let model = updated.cast(model_dtype);
        let state = PrecisionState {
            algorithm,
            master,
            model_dtype: dtype_code(model_dtype),
            compute_dtype: dtype_code(compute_dtype),
            storage_dtype: dtype_code(storage_dtype),
            master_weights,
        };
        (model, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device) -> Self::State<D> {
        state.algorithm = state.algorithm.map(|state| O::to_device(state, device));
        state.master = state.master.map(|master| master.to_device(device));
        state
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
