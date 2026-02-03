#[cfg(feature = "autotune")]
use crate::optim::reduce::tune::fused_reduce_autotune;
use crate::{
    CubeFusionHandle, FallbackOperation,
    engine::{
        launch::FuseTraceLauncher,
        trace::{FuseTrace, TraceError, TuneOutput},
    },
    optim::{
        elemwise::{ElemwiseOptimization, ElemwiseOptimizationState},
        reduce::{ReduceOptimizationInfo, ReduceOptimizationState, ReduceOptimizationTuneArg},
        reduce_broadcasted::{
            launch::{FusedReduceBroadcastedLaunch, ReduceBrFuseBlock},
            tune::fused_broadcasted_reduce_autotune,
        },
    },
};
use burn_fusion::stream::Context;
use cubecl::{Runtime, prelude::*};
use cubek::reduce::launch::RoutineStrategy;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub struct ReduceBroadcastedOptimization<R: Runtime> {
    pub(crate) info: Arc<ReduceBroadcastedOptimizationInfo<R>>,
    pub(crate) num_ops: usize,
}

pub(crate) struct ReduceBroadcastedOptimizationInfo<R: Runtime> {
    pub(crate) fallbacks: Vec<ReduceBlockOptimInfo<R>>,
    pub(crate) info_br: Arc<ReduceBrInfo>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct ReduceBrInfo {
    pub(crate) blocks: Vec<ReduceBrFuseBlock>,
    pub(crate) trace: FuseTrace,
    pub(crate) reduce_axis: usize,
}

pub(crate) enum ReduceBlockOptimInfo<R: Runtime> {
    Reduce(Arc<ReduceOptimizationInfo<R>>),
    Elemwise(Arc<ElemwiseOptimization<R>>),
}

impl<R: Runtime> ReduceBlockOptimInfo<R> {
    pub fn from_state(device: &R::Device, state: ReduceBlockState) -> Self {
        match state {
            ReduceBlockState::Reduce(state) => {
                Self::Reduce(Arc::new(ReduceOptimizationInfo::from_state(device, state)))
            }
            ReduceBlockState::Elemwise(state) => {
                Self::Elemwise(Arc::new(ElemwiseOptimization::from_state(device, state)))
            }
        }
    }
    pub fn to_state(&self) -> ReduceBlockState {
        match self {
            Self::Reduce(info) => ReduceBlockState::Reduce(info.to_state()),
            Self::Elemwise(info) => ReduceBlockState::Elemwise(info.to_state()),
        }
    }
}

pub(crate) struct ReduceBroadcastedOptimizationTuneArg<R: Runtime> {
    pub(crate) fallbacks: Vec<ReduceBlockOptimArg<R>>,
    pub(crate) info_br: Arc<ReduceBrInfo>,
    pub(crate) client: ComputeClient<R>,
    pub(crate) device: R::Device,
}

pub(crate) enum ReduceBlockOptimArg<R: Runtime> {
    Reduce(ReduceOptimizationTuneArg<R>),
    Elemwise(Arc<ElemwiseOptimization<R>>),
}

impl<R: Runtime> ReduceBlockOptimArg<R> {
    pub fn execute_fallback<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Option<TuneOutput<R>> {
        match self {
            ReduceBlockOptimArg::Reduce(reduce) => {
                #[cfg(feature = "autotune")]
                {
                    fused_reduce_autotune::<R, BT>(reduce.clone(), context);
                    None
                }
                #[cfg(not(feature = "autotune"))]
                Some(reduce.execute_fallback::<BT>(context))
            }
            ReduceBlockOptimArg::Elemwise(elem) => {
                elem.execute::<BT>(context);
                None
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReduceBroadcastedOptimizationState {
    fallbacks: Vec<ReduceBlockState>,
    broadcasted: ReduceBrInfo,
    num_ops: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum ReduceBlockState {
    Reduce(ReduceOptimizationState),
    Elemwise(ElemwiseOptimizationState),
}

impl<R: Runtime> ReduceBroadcastedOptimizationTuneArg<R> {
    pub fn execute_fused<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        strategy: RoutineStrategy,
    ) -> Result<TuneOutput<R>, TraceError<String>> {
        println!("==== Execute Fused ====");
        println!("{}", self.info_br.trace);
        for b in self.info_br.blocks.iter() {
            println!("{} = {:?}({})", b.output, b.op, b.input);
        }
        let launch = FusedReduceBroadcastedLaunch::new(
            &self.info_br.blocks,
            self.info_br.reduce_axis,
            strategy,
        );
        let launcher = FuseTraceLauncher::new(&self.info_br.trace, &launch);

        launcher
            .launch::<BT>(&self.client, &self.device, context)
            .map_err(|err| TraceError::RunnerError(format!("{:?}", err)))
    }

    pub fn execute_fallback<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) {
        for fallback in self.fallbacks.iter() {
            fallback.execute_fallback::<BT>(context);
        }
    }
}

#[allow(clippy::too_many_arguments)]
impl<R: Runtime> ReduceBroadcastedOptimization<R> {
    /// Execute the optimization.
    pub fn execute<BT: CubeElement>(
        &mut self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        fallback: impl Fn(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        // println!("==== Execute broadcasted ====");
        let mut current_index = 0;
        let mut client = None;
        let mut device = None;

        let fallbacks = self
            .info
            .fallbacks
            .iter()
            .map(|info| {
                match info {
                    ReduceBlockOptimInfo::Reduce(info) => {
                        // The index of the fallback reduce is the number of ops fused as read.
                        let fallback = fallback(current_index + info.len_read);
                        client = Some(info.client.clone());
                        device = Some(info.device.clone());
                        let arg = ReduceOptimizationTuneArg {
                            info: info.clone(),
                            fallback: Arc::new(fallback),
                        };
                        current_index += info.len;
                        ReduceBlockOptimArg::Reduce(arg)
                    }
                    ReduceBlockOptimInfo::Elemwise(op) => ReduceBlockOptimArg::Elemwise(op.clone()),
                }
            })
            .collect();

        let arg = ReduceBroadcastedOptimizationTuneArg {
            fallbacks,
            client: client.unwrap(),
            device: device.unwrap(),
            info_br: self.info.info_br.clone(),
        };

        #[cfg(feature = "autotune")]
        fused_broadcasted_reduce_autotune::<R, BT>(arg, context);

        #[cfg(not(feature = "autotune"))]
        arg.execute_fallback::<BT>(context);
    }

    pub fn to_state(&self) -> ReduceBroadcastedOptimizationState {
        ReduceBroadcastedOptimizationState {
            fallbacks: self
                .info
                .fallbacks
                .iter()
                .map(|info| info.to_state())
                .collect(),
            broadcasted: self.info.info_br.as_ref().clone(),
            num_ops: self.num_ops,
        }
    }

    pub fn from_state(device: &R::Device, state: ReduceBroadcastedOptimizationState) -> Self {
        Self {
            info: Arc::new(ReduceBroadcastedOptimizationInfo {
                fallbacks: state
                    .fallbacks
                    .into_iter()
                    .map(|state| ReduceBlockOptimInfo::from_state(device, state))
                    .collect(),
                info_br: Arc::new(state.broadcasted),
            }),
            num_ops: state.num_ops,
        }
    }

    /// Returns the number of output buffers added by fusion.
    pub fn num_ops_fused(&self) -> usize {
        self.num_ops
    }
}
