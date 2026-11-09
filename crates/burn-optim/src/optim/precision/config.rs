use burn::config::Config;
use burn::tensor::FloatDType;
use burn_core as burn;

/// Precision used to evaluate optimizer equations.
#[derive(Config, Debug, PartialEq, Eq)]
pub enum ComputePrecision {
    /// Use the parameter's dtype.
    Parameter,
    /// Promote F16 and BF16 parameters to F32; preserve other floating dtypes.
    PromoteHalf,
    /// Use this dtype explicitly, including when it narrows the parameter's precision.
    Float(FloatDType),
}

impl ComputePrecision {
    pub(crate) fn resolve(&self, parameter: FloatDType) -> FloatDType {
        match self {
            Self::Parameter => parameter,
            Self::PromoteHalf => match parameter {
                FloatDType::F16 | FloatDType::BF16 => FloatDType::F32,
                other => other,
            },
            Self::Float(dtype) => *dtype,
        }
    }
}

/// How updated parameters are retained between optimizer steps.
#[derive(Config, Debug, PartialEq, Eq)]
pub enum ParameterUpdate {
    /// Round the updated parameter to the model dtype after each step.
    Direct,
    /// Retain authoritative parameters in computation precision when the model dtype differs.
    ///
    /// Masters are initialized from the current model parameters on the first step; promotion
    /// cannot recover precision already lost before that step.
    /// No extra copy is retained when the model already uses the computation dtype. Replace or
    /// reset optimizer state when replacing model parameters, otherwise the masters remain the
    /// source of truth. This does not enable forward/backward autocasting or loss scaling.
    MasterWeights,
}

/// Persistent representation of eligible optimizer statistics.
#[derive(Config, Debug, PartialEq, Eq)]
pub enum StateStorage {
    /// Retain computation precision.
    Compute,
    /// Store eligible statistics in this floating dtype, decoding them before the next update.
    ///
    /// State marked [`super::StateTensorKind::FullPrecision`] remains in computation precision.
    /// Lower precision can affect convergence and should be validated for the training workload.
    Float(FloatDType),
}

impl StateStorage {
    pub(crate) fn resolve(&self, compute: FloatDType) -> FloatDType {
        match self {
            Self::Compute => compute,
            Self::Float(dtype) => *dtype,
        }
    }
}

/// Shared precision configuration for any [`crate::Optimizer`].
///
/// Apply with [`crate::Optimizer::with_precision`]. Opting in promotes half-precision arithmetic
/// and retains master weights by default. Bare optimizers retain their existing behavior.
///
/// ```rust
/// use burn_optim::{AdamWConfig, Optimizer, OptimizerPrecisionConfig, StateStorage, ModuleOptimizer};
/// use burn_core::tensor::FloatDType;
///
/// let adam = AdamWConfig::new().build().with_precision(
///     OptimizerPrecisionConfig::new().with_state_storage(StateStorage::Float(FloatDType::BF16)),
/// );
/// let optimizer = ModuleOptimizer::from(adam);
/// ```
#[derive(Config, Debug, PartialEq, Eq)]
pub struct OptimizerPrecisionConfig {
    /// Optimizer arithmetic dtype, independent of model and stored-state dtypes.
    #[config(default = "ComputePrecision::PromoteHalf")]
    pub compute: ComputePrecision,
    /// Parameter retention policy.
    #[config(default = "ParameterUpdate::MasterWeights")]
    pub parameter_update: ParameterUpdate,
    /// Persistent representation of eligible statistics.
    #[config(default = "StateStorage::Compute")]
    pub state_storage: StateStorage,
}
