//! AdamW's private backend extension, with eager and deferred CubeCL execution.
//!
//! Unsupported high-level inputs stay on the portable tensor implementation. Fusion registers
//! the same operation as raw dispatch and resolves its buffers only when execution is needed;
//! the backend handles layout-dependent fallback after those buffers become available.

use alloc::{vec, vec::Vec};
use burn::backend::{
    Backend, Dispatch, DispatchDevice, ExtensionType, backend_extension, tensor::FloatTensor,
};
use burn::tensor::{DType, Tensor};
use burn_core as burn;
use burn_cubecl::kernel::optim::{AdamWHostScalars, AdamWInputs, AdamWOutput};
use burn_fusion::{Fusion, FusionBackend, FusionRuntime};

use super::{AdamWState, AdaptiveMomentumW, StepInputs};
use crate::{LearningRate, optim::AdaptiveMomentumState};

/// Only a transport value for the extension; the optimizer continues to own its existing state.
#[derive(ExtensionType)]
enum Moments<B: Backend> {
    Empty,
    State {
        first: FloatTensor<B>,
        second: FloatTensor<B>,
    },
    Maximum {
        first: FloatTensor<B>,
        second: FloatTensor<B>,
        maximum: FloatTensor<B>,
    },
}

impl<B: Backend> Moments<B> {
    fn state(
        first: FloatTensor<B>,
        second: FloatTensor<B>,
        maximum: Option<FloatTensor<B>>,
    ) -> Self {
        match maximum {
            Some(maximum) => Self::Maximum {
                first,
                second,
                maximum,
            },
            None => Self::State { first, second },
        }
    }

    fn into_inputs(
        self,
        param: FloatTensor<B>,
        grad: FloatTensor<B>,
    ) -> AdamWInputs<FloatTensor<B>> {
        match self {
            Self::Empty => AdamWInputs::Initial { param, grad },
            Self::State { first, second } => AdamWInputs::State {
                param,
                grad,
                moment_1: first,
                moment_2: second,
                max_moment_2: None,
            },
            Self::Maximum {
                first,
                second,
                maximum,
            } => AdamWInputs::State {
                param,
                grad,
                moment_1: first,
                moment_2: second,
                max_moment_2: Some(maximum),
            },
        }
    }
}

#[backend_extension(Cube)]
trait AdamWOps: Backend {
    fn adamw(
        param: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        #[extension_type] moments: Moments<Self>,
        scalars: AdamWHostScalars,
        cautious: bool,
        amsgrad: bool,
    ) -> (FloatTensor<Self>, Moments<Self>);
}

impl AdamWOps for burn_cubecl::CubeBackend {
    fn adamw(
        param: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        moments: Moments<Self>,
        scalars: AdamWHostScalars,
        cautious: bool,
        amsgrad: bool,
    ) -> (FloatTensor<Self>, Moments<Self>) {
        let AdamWOutput {
            param,
            moment_1,
            moment_2,
            max_moment_2,
        } = burn_cubecl::kernel::optim::adamw(
            scalars,
            cautious,
            amsgrad,
            moments.into_inputs(param, grad),
        );
        (param, Moments::state(moment_1, moment_2, max_moment_2))
    }
}

// Compile this implementation whenever the extension is available. A different crate can enable
// CubeCL's fusion feature, so a local `cfg(feature = "fusion")` would miss that configuration.
impl<B: FusionBackend + AdamWOps> AdamWOps for Fusion<B> {
    fn adamw(
        param: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        moments: Moments<Self>,
        scalars: AdamWHostScalars,
        cautious: bool,
        amsgrad: bool,
    ) -> (FloatTensor<Self>, Moments<Self>) {
        use burn_fusion::{
            ExecutionError,
            custom::{
                CustomOpIr, HandleContainer, Operation, OperationIr, ScalarIr, StreamId, TensorIr,
            },
        };

        #[derive(Debug)]
        struct Step<B> {
            desc: CustomOpIr,
            scalars: AdamWHostScalars,
            cautious: bool,
            amsgrad: bool,
            backend: core::marker::PhantomData<B>,
        }

        impl<B: FusionBackend + AdamWOps> Operation<B::FusionRuntime> for Step<B> {
            fn execute(
                &self,
                handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
            ) -> Result<(), ExecutionError> {
                let mut inputs = self.desc.inputs.iter();
                let param = handles.get_float_tensor::<B>(inputs.next().unwrap());
                let grad = handles.get_float_tensor::<B>(inputs.next().unwrap());
                let moments = match inputs.next() {
                    Some(first) => {
                        let first = handles.get_float_tensor::<B>(first);
                        let second = handles.get_float_tensor::<B>(inputs.next().unwrap());
                        let maximum = inputs.next().map(|t| handles.get_float_tensor::<B>(t));
                        Moments::state(first, second, maximum)
                    }
                    None => Moments::Empty,
                };
                let (param, moments) = B::adamw(
                    param,
                    grad,
                    moments,
                    self.scalars,
                    self.cautious,
                    self.amsgrad,
                );
                let mut outputs = self.desc.outputs.iter();
                handles.register_float_tensor::<B>(&outputs.next().unwrap().id, param);
                let (first, second, maximum) = match moments {
                    Moments::State { first, second } => (first, second, None),
                    Moments::Maximum {
                        first,
                        second,
                        maximum,
                    } => (first, second, Some(maximum)),
                    Moments::Empty => unreachable!("an AdamW step always produces moments"),
                };
                handles.register_float_tensor::<B>(&outputs.next().unwrap().id, first);
                handles.register_float_tensor::<B>(&outputs.next().unwrap().id, second);
                if let Some(maximum) = maximum {
                    handles.register_float_tensor::<B>(&outputs.next().unwrap().id, maximum);
                }
                Ok(())
            }
        }

        let client = param.client.clone();
        let has_maximum = amsgrad || matches!(&moments, Moments::Maximum { .. });
        let outputs: Vec<_> = (0..if has_maximum { 4 } else { 3 })
            .map(|_| {
                TensorIr::uninit(
                    client.create_empty_handle(),
                    param.shape.clone(),
                    param.dtype,
                )
            })
            .collect();
        let mut inputs = vec![param.into_ir(), grad.into_ir()];
        match moments {
            Moments::Empty => {}
            Moments::State { first, second } => inputs.extend([first.into_ir(), second.into_ir()]),
            Moments::Maximum {
                first,
                second,
                maximum,
            } => inputs.extend([first.into_ir(), second.into_ir(), maximum.into_ir()]),
        }
        let mut args: Vec<_> = scalars
            .to_scalars()
            .into_iter()
            .map(ScalarIr::from)
            .collect();
        args.extend([ScalarIr::Bool(cautious), ScalarIr::Bool(amsgrad)]);
        let desc = CustomOpIr::with_scalars("optim::adamw", &inputs, &outputs, args);
        let mut outputs = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                Step::<B> {
                    desc,
                    scalars,
                    cautious,
                    amsgrad,
                    backend: core::marker::PhantomData,
                },
            )
            .into_iter();
        let param = outputs.next().unwrap();
        let moments = Moments::state(
            outputs.next().unwrap(),
            outputs.next().unwrap(),
            outputs.next(),
        );
        (param, moments)
    }
}

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
    if !eligible(&tensor, &grad, state.as_ref()) {
        return Err((tensor, grad, state));
    }
    let time = state
        .as_ref()
        .map(|state| state.momentum.time + 1)
        .unwrap_or(1);
    let moments = match state {
        Some(state) => Moments::state(
            state.momentum.moment_1.into_dispatch(),
            state.momentum.moment_2.into_dispatch(),
            state.momentum.max_moment_2.map(Tensor::into_dispatch),
        ),
        None => Moments::Empty,
    };
    let scalars = AdamWHostScalars::new(
        lr,
        momentum.beta_1,
        momentum.beta_2,
        momentum.epsilon,
        weight_decay,
        time,
    );
    let (param, moments) = Dispatch::adamw(
        tensor.into_dispatch(),
        grad.into_dispatch(),
        moments,
        scalars,
        cautious_weight_decay,
        momentum.amsgrad,
    );
    let (first, second, maximum) = match moments {
        Moments::State { first, second } => (first, second, None),
        Moments::Maximum {
            first,
            second,
            maximum,
        } => (first, second, Some(maximum)),
        Moments::Empty => unreachable!("an AdamW step always produces moments"),
    };
    let state = AdamWState {
        momentum: AdaptiveMomentumState {
            time,
            moment_1: Tensor::from_dispatch(first),
            moment_2: Tensor::from_dispatch(second),
            max_moment_2: maximum.map(Tensor::from_dispatch),
        },
    };
    Ok((Tensor::from_dispatch(param), Some(state)))
}

/// Reject unsupported decorators before dispatch, whose extension contract is intentionally narrow.
fn eligible<const D: usize>(
    tensor: &Tensor<D>,
    grad: &Tensor<D>,
    state: Option<&AdamWState<D>>,
) -> bool {
    let shape = tensor.shape();
    let device = tensor.device();
    let supported = |tensor: &Tensor<D>| {
        tensor.dtype() == DType::F32
            && matches!(tensor.device().as_dispatch(), DispatchDevice::Cube(_))
            && tensor.gradient_checkpointing_strategy().is_none()
    };
    if !supported(tensor) || shape.num_elements() == 0 || shape.num_dims() == 0 {
        return false;
    }
    let matches =
        |other: &Tensor<D>| supported(other) && other.shape() == shape && other.device() == device;
    matches(grad)
        && state.is_none_or(|state| {
            matches(&state.momentum.moment_1)
                && matches(&state.momentum.moment_2)
                && state.momentum.max_moment_2.as_ref().is_none_or(matches)
        })
}
