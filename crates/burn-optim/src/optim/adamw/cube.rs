//! Bridge between [`AdamW`](super::AdamW) tensors and the fused CubeCL kernel.
//!
//! Everything that can make the kernel inapplicable is checked before a launch is submitted:
//! the dispatch is raw CubeCL (not fusion), every tensor is `F32`, on one device, with one
//! shape, and the parameter is non-empty. Only then are the tensors converted to primitives
//! (through cheap handle clones) and handed to `burn-cubecl`; if it declines, the untouched
//! inputs run the reference implementation. Once all conversions succeed, original wrapper
//! handles are dropped so the kernel can reuse exclusively owned buffers.
//!
//! The update is reached through a direct `burn-cubecl` dependency rather than a new
//! `Backend` operation: adding one would make every backend implement AdamW and change the
//! public backend contract, while this path must stay optional and fall back everywhere else.

use burn_core as burn;

use burn::tensor::{DType, Tensor};
use burn_cubecl::kernel::optim::{AdamWHostScalars, AdamWInputs, AdamWOutput, try_adamw};

use super::{AdamWState, AdaptiveMomentumW, StepInputs};
use crate::LearningRate;
use crate::optim::AdaptiveMomentumState;

type CubePrimitive = <burn_cubecl::Cube as burn::backend::BackendTypes>::FloatTensorPrimitive;

/// Try the single-kernel update, returning the inputs untouched when it does not apply.
///
/// The error carries the untouched inputs so the caller can run the reference implementation
/// without rebuilding them.
#[allow(clippy::result_large_err)]
pub(super) fn try_step<const D: usize>(
    lr: LearningRate,
    tensor: Tensor<D>,
    grad: Tensor<D>,
    state: Option<AdamWState<D>>,
    momentum: &AdaptiveMomentumW,
    weight_decay: f32,
    cautious_weight_decay: bool,
) -> Result<(Tensor<D>, Option<AdamWState<D>>), StepInputs<D>> {
    // Fusion tensors are handles into a deferred graph, and the fused kernel writes raw CubeCL
    // buffers. Decline before converting anything.
    if burn_cubecl::FUSED {
        return Err((tensor, grad, state));
    }
    if !eligible(&tensor, &grad, state.as_ref()) {
        return Err((tensor, grad, state));
    }

    let time = state
        .as_ref()
        .map(|state| state.momentum.time + 1)
        .unwrap_or(1);
    // Convert clones: a primitive conversion has no reverse on failure, and the originals must
    // stay whole for the reference fallback.
    let param = tensor.clone().try_into_primitive::<burn_cubecl::Cube>();
    let grad_primitive = grad.clone().try_into_primitive::<burn_cubecl::Cube>();
    let (param, grad_primitive) = match (param, grad_primitive) {
        (Ok(param), Ok(grad)) => (param, grad),
        _ => return Err((tensor, grad, state)),
    };

    let moments = match &state {
        Some(existing) => {
            let moment_1 = existing
                .momentum
                .moment_1
                .clone()
                .try_into_primitive::<burn_cubecl::Cube>();
            let moment_2 = existing
                .momentum
                .moment_2
                .clone()
                .try_into_primitive::<burn_cubecl::Cube>();
            match (moment_1, moment_2) {
                (Ok(moment_1), Ok(moment_2)) => Some((moment_1, moment_2)),
                _ => return Err((tensor, grad, state)),
            }
        }
        None => None,
    };

    let max_moment_2 = match state
        .as_ref()
        .and_then(|state| state.momentum.max_moment_2.as_ref())
    {
        Some(max) => match max.clone().try_into_primitive::<burn_cubecl::Cube>() {
            Ok(max) => Some(max),
            Err(_) => return Err((tensor, grad, state)),
        },
        _ => None,
    };

    let inputs = match moments {
        Some((moment_1, moment_2)) => AdamWInputs::State {
            param,
            grad: grad_primitive,
            moment_1,
            moment_2,
            max_moment_2,
        },
        None => AdamWInputs::Initial {
            param,
            grad: grad_primitive,
        },
    };

    let scalars = AdamWHostScalars::new(
        lr,
        momentum.beta_1,
        momentum.beta_2,
        momentum.epsilon,
        weight_decay,
        time,
    );

    // Every conversion succeeded. The error from try_adamw owns these same primitives,
    // so it can reconstruct the fallback inputs without retaining clones that block COW.
    drop((tensor, grad, state));
    let tensor = Tensor::<D>::from_primitive::<burn_cubecl::Cube>;

    let output = match try_adamw(scalars, cautious_weight_decay, momentum.amsgrad, inputs) {
        Ok(output) => output,
        Err(inputs) => {
            return Err(match inputs {
                AdamWInputs::Initial { param, grad } => (tensor(param), tensor(grad), None),
                AdamWInputs::State {
                    param,
                    grad,
                    moment_1,
                    moment_2,
                    max_moment_2,
                } => (
                    tensor(param),
                    tensor(grad),
                    Some(AdamWState {
                        momentum: AdaptiveMomentumState {
                            time: time - 1,
                            moment_1: tensor(moment_1),
                            moment_2: tensor(moment_2),
                            max_moment_2: max_moment_2.map(tensor),
                        },
                    }),
                ),
            });
        }
    };

    let (param, state) = restore(time, output);
    Ok((param, Some(state)))
}

fn restore<const D: usize>(
    time: usize,
    output: AdamWOutput<CubePrimitive>,
) -> (Tensor<D>, AdamWState<D>) {
    let param = Tensor::<D>::from_primitive::<burn_cubecl::Cube>(output.param);
    let moment_1 = Tensor::<D>::from_primitive::<burn_cubecl::Cube>(output.moment_1);
    let moment_2 = Tensor::<D>::from_primitive::<burn_cubecl::Cube>(output.moment_2);
    let max_moment_2 = output
        .max_moment_2
        .map(Tensor::<D>::from_primitive::<burn_cubecl::Cube>);

    let state = AdamWState {
        momentum: AdaptiveMomentumState {
            time,
            moment_1,
            moment_2,
            max_moment_2,
        },
    };

    (param, state)
}

/// Whether every input can be read by the fused kernel: `F32`, one shape, one device, and a
/// parameter with elements.
///
/// This rejects cheaply on the high-level tensors, before any handle clone. The primitive-level
/// `try_adamw` repeats the check because only primitive metadata exposes tiling and quantization
/// parameters. A future capability added to one side and not the other shows up as a fallback,
/// never as a wrong result.
fn eligible<const D: usize>(
    tensor: &Tensor<D>,
    grad: &Tensor<D>,
    state: Option<&AdamWState<D>>,
) -> bool {
    if tensor.dtype() != DType::F32
        || tensor.shape().num_elements() == 0
        || tensor.shape().num_dims() == 0
    {
        return false;
    }

    let matches = |other: &Tensor<D>| {
        other.dtype() == DType::F32
            && other.shape() == tensor.shape()
            && other.device() == tensor.device()
    };

    if !matches(grad) {
        return false;
    }

    match state {
        Some(state) => {
            matches(&state.momentum.moment_1)
                && matches(&state.momentum.moment_2)
                && state
                    .momentum
                    .max_moment_2
                    .as_ref()
                    .map(matches)
                    .unwrap_or(true)
        }
        None => true,
    }
}
