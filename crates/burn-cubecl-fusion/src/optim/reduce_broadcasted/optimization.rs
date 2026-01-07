use crate::{
    CubeFusionHandle, FallbackOperation,
    engine::{
        codegen::ir::{FuseArg, FuseBlockConfig, FuseType, GlobalArgsLaunch},
        launch::runner::{TraceRunner, Vectorization},
        trace::{FuseTrace, TraceError, TuneOutput},
    },
    optim::reduce::ReduceOptimizationInfo,
};
use burn_fusion::stream::Context;
use burn_ir::ReduceDimOpIr;
use cubecl::{Runtime, client::ComputeClient, prelude::*};
use cubek::reduce::{ReduceError, launch::RoutineStrategy};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[cfg(not(feature = "autotune"))]
use cubek::reduce::routines::{BlueprintStrategy, unit::UnitStrategy};

pub struct ReduceBroadcastedOptimization<R: Runtime> {
    info: Arc<ReduceBroadcastedOptimizationInfo<R>>,
}

pub(crate) struct ReduceBroadcastedOptimizationInfo<R: Runtime> {
    pub(crate) infos: Vec<ReduceOptimizationInfo<R>>,
    pub(crate) client: ComputeClient<R>,
    pub(crate) device: R::Device,
}

pub(crate) struct ReduceBroadcastedOptimizationTuneArg<R: Runtime> {
    pub(crate) info: Arc<ReduceBroadcastedOptimizationInfo<R>>,
    pub(crate) fallback: Box<dyn FallbackOperation<R>>,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub enum ReduceInstruction {
    ArgMax,
    ArgMin,
    Mean,
    Prod,
    Sum,
    Max,
    Min,
    MaxAbs,
}

pub trait ReduceFallbackFn<R: Runtime>: Send + Sync {
    fn run(&self, context: &mut Context<'_, CubeFusionHandle<R>>);
}

#[derive(Serialize, Deserialize)]
pub struct ReduceBroadcastedOptimizationState {
    trace: FuseTrace,
    trace_read_fallback: FuseTrace,
    trace_write_fallback: FuseTrace,
    pub(crate) reduce: FusedReduce,
    len: usize,
    len_read: usize,
}

impl core::fmt::Debug for ReduceBroadcastedOptimizationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{{ len_read: {}, len_total: {} }}",
            self.len_read, self.len
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FusedReduce {
    pub(crate) input: FuseArg,
    pub(crate) output: FuseArg,
    pub(crate) acc: FuseType,
    pub(crate) axis: usize,
    pub(crate) op: ReduceDimOpIr,
    pub(crate) use_planes: bool,
    pub(crate) shared: bool,
    pub(crate) inst: ReduceInstruction,
}

#[derive(new)]
pub struct FusedReduceLaunch<'a> {
    reduce: &'a FusedReduce,
    strategy: RoutineStrategy,
}

#[derive(Debug)]
pub enum FusedReduceError {
    Reduce(ReduceError),
    InvalidSelection(Box<&'static str>),
    InvalidInput,
}

impl From<ReduceError> for FusedReduceError {
    fn from(value: ReduceError) -> Self {
        Self::Reduce(value)
    }
}

impl<R: Runtime> ReduceBroadcastedOptimizationTuneArg<R> {
    pub fn execute_fused<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        strategy: RoutineStrategy,
    ) -> Result<TuneOutput<R>, TraceError<FusedReduceError>> {
        todo!()
    }

    pub fn execute_fallback<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> TuneOutput<R> {
        todo!()
    }
}

#[allow(clippy::too_many_arguments)]
impl<R: Runtime> ReduceBroadcastedOptimization<R> {
    pub fn new(infos: Vec<ReduceOptimizationInfo<R>>) -> Self {
        todo!()
    }
    /// Execute the optimization.
    pub fn execute<BT: CubeElement>(
        &mut self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        fallback: impl FnOnce(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        todo!()
    }

    pub fn num_output_buffers(&self) -> usize {
        todo!()
    }

    pub fn to_state(&self) -> ReduceBroadcastedOptimizationState {
        todo!()
    }

    pub fn from_state(device: &R::Device, state: ReduceBroadcastedOptimizationState) -> Self {
        todo!()
    }

    /// Returns the number of output buffers added by fusion.
    pub fn num_ops_fused(&self) -> usize {
        todo!()
    }
}

// TODO: Implement better vectorization here.
impl<R: Runtime> Vectorization<R> for FusedReduceLaunch<'_> {}

impl<R: Runtime> TraceRunner<R> for FusedReduceLaunch<'_> {
    type Error = FusedReduceError;

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        configs: &'a [FuseBlockConfig],
    ) -> Result<(), FusedReduceError> {
        todo!()
    }
}
