use burn_core as burn;

use burn::module::ParamId;
use burn::tensor::{GradientCheckpointingStrategy, Tensor};

/// The autodiff state a parameter carries, taken before an optimizer updates it on the inner
/// backend and put back onto the updated value.
///
/// [`Tensor::from_inner`] lifts a parameter back with the default state: gradients not retained,
/// not distributed, and the default gradient-checkpointing strategy. A parameter created on a
/// device with another strategy would then meet its inputs under two strategies on the next
/// forward pass, which autodiff refuses.
pub(crate) struct ParameterContext {
    require_grad: bool,
    #[cfg(feature = "std")]
    distributed: bool,
    checkpointing: Option<GradientCheckpointingStrategy>,
}

impl ParameterContext {
    pub(crate) fn capture<const D: usize>(parameter: &Tensor<D>) -> Self {
        Self {
            require_grad: parameter.is_require_grad(),
            #[cfg(feature = "std")]
            distributed: parameter.is_distributed(),
            checkpointing: parameter.gradient_checkpointing_strategy(),
        }
    }

    #[cfg_attr(not(feature = "std"), allow(unused_variables))]
    pub(crate) fn restore<const D: usize>(self, updated: Tensor<D>, id: ParamId) -> Tensor<D> {
        let mut parameter = Tensor::from_inner(updated);

        if let Some(strategy) = self.checkpointing {
            parameter = parameter.with_gradient_checkpointing_strategy(strategy);
        }
        if self.require_grad {
            parameter = parameter.require_grad();
        }
        #[cfg(feature = "std")]
        if self.distributed {
            parameter = parameter.set_distributed(id);
        }

        parameter
    }
}
