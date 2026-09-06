use burn_core as burn;

use burn::tensor::{GradientCheckpointingStrategy, Tensor};

/// Lifts an updated parameter back onto the autodiff tape with the gradient checkpointing
/// strategy the parameter trained under.
///
/// [`Tensor::from_inner`] assigns the default strategy. A parameter created on a device with
/// another strategy would then meet its inputs under two strategies on the next forward pass,
/// which autodiff refuses.
pub fn from_inner_with_strategy<const D: usize>(
    inner: Tensor<D>,
    strategy: Option<GradientCheckpointingStrategy>,
) -> Tensor<D> {
    let tensor = Tensor::from_inner(inner);
    match strategy {
        Some(strategy) => tensor.with_gradient_checkpointing_strategy(strategy),
        None => tensor,
    }
}
