use std::sync::Arc;

use burn_fusion::stream::Context;
use burn_ir::ReduceDimOpIr;
use burn_tensor::DType;
use cubecl::prelude::*;
use cubecl::reduce::{
    BoundChecksInner, ReduceFamily, ReduceParams, ReduceStrategy, init_tensors,
    reduce_kernel_virtual,
};
use cubecl::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    reduce::{LineMode, ReduceConfig, ReduceError},
};
use cubecl::{
    ir::StorageType,
    reduce::instructions::{ReduceFn, ReduceFnConfig},
};
use serde::{Deserialize, Serialize};

use crate::elemwise::optimization::ElemwiseRunner;
use crate::reduce::args::FusedReduceArgs;
use crate::shared::ir::{FusePrecision, RefLayout};
use crate::shared::trace::{TraceError, TraceRunner};
use crate::shared::trace::{TuneOutput, Vectorization};
use crate::shared::{
    ir::{Arg, FuseBlockConfig, GlobalArgsLaunch},
    trace::FuseTrace,
};
use crate::{CubeFusionHandle, FallbackOperation};

use super::args::{
    FusedReduceInput, FusedReduceInputLaunch, FusedReduceOutput, FusedReduceOutputLaunch,
};
use super::tune::fused_reduce_autotune;

pub struct ReduceOptimization<R: Runtime> {
    info: Arc<ReduceOptimizationInfo<R>>,
}

pub(crate) struct ReduceOptimizationInfo<R: Runtime> {
    pub(crate) trace: FuseTrace,
    trace_read_fallback: FuseTrace,
    trace_write_fallback: FuseTrace,
    pub(crate) client: ComputeClient<R::Server, R::Channel>,
    pub(crate) device: R::Device,
    pub(crate) len: usize,
    pub(crate) len_read: usize,
    pub(crate) reduce: FusedReduce,
    pub(crate) reduce_plane: FusedReduce,
    pub(crate) reduce_shared_plane: FusedReduce,
}

pub(crate) struct ReduceOptimizationTuneArg<R: Runtime> {
    pub(crate) info: Arc<ReduceOptimizationInfo<R>>,
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
pub struct ReduceOptimizationState {
    trace: FuseTrace,
    trace_read_fallback: FuseTrace,
    trace_write_fallback: FuseTrace,
    pub(crate) reduce: FusedReduce,
    pub(crate) reduce_plane: FusedReduce,
    pub(crate) reduce_shared_plane: FusedReduce,
    len: usize,
    len_read: usize,
}

impl core::fmt::Debug for ReduceOptimizationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{{ len_read: {}, len_total: {} }}",
            self.len_read, self.len
        ))
    }
}

#[derive(new, Clone, Serialize, Deserialize, Debug)]
pub struct FusedReduce {
    input: Arg,
    output: Arg,
    pub(crate) acc: FusePrecision,
    pub(crate) axis: usize,
    pub(crate) op: ReduceDimOpIr,
    strategy: ReduceStrategy,
    inst: ReduceInstruction,
}

impl FusedReduce {
    pub fn with_strategy(&self, strategy: ReduceStrategy) -> Self {
        Self {
            input: self.input.clone(),
            output: self.output.clone(),
            acc: self.acc,
            axis: self.axis,
            op: self.op.clone(),
            strategy,
            inst: self.inst,
        }
    }
}

#[derive(Debug)]
pub enum FusedReduceError {
    LaunchError(ReduceError),
    InvalidSelection(Box<&'static str>),
    InvalidInput,
}

impl<R: Runtime> ReduceOptimizationTuneArg<R> {
    pub fn execute_fused_reduce<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<TuneOutput<R>, TraceError<FusedReduceError>> {
        FuseTrace::run::<R, BT, FusedReduce>(
            &self.info.trace,
            &self.info.client,
            &self.info.device,
            context,
            &self.info.reduce,
        )
    }

    pub fn execute_fused_reduce_plane<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<TuneOutput<R>, TraceError<FusedReduceError>> {
        FuseTrace::run::<R, BT, FusedReduce>(
            &self.info.trace,
            &self.info.client,
            &self.info.device,
            context,
            &self.info.reduce_plane,
        )
    }

    pub fn execute_fused_reduce_shared_plane<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<TuneOutput<R>, TraceError<FusedReduceError>> {
        FuseTrace::run::<R, BT, FusedReduce>(
            &self.info.trace,
            &self.info.client,
            &self.info.device,
            context,
            &self.info.reduce_shared_plane,
        )
    }

    pub fn execute_fallback<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> TuneOutput<R> {
        #[allow(unused_mut)] // It is used when `autotune-checks` is activated.
        let mut output_read = self
            .info
            .trace_read_fallback
            .run::<R, BT, ElemwiseRunner>(
                &self.info.client,
                &self.info.device,
                context,
                &ElemwiseRunner,
            )
            .unwrap();

        self.fallback.run(context);

        #[cfg(feature = "autotune-checks")]
        if let TuneOutput::Checked { handles } = &mut output_read {
            let out_desc = context.tensors.get(&self.info.reduce.op.out.id).unwrap();
            let handle_out = context
                .handles
                .get_handle(&out_desc.id, &burn_ir::TensorStatus::ReadOnly);

            handles.insert(
                self.info.reduce.op.out.id,
                (out_desc.shape.clone(), handle_out.clone()),
            );
        }

        let output_write = self
            .info
            .trace_write_fallback
            .run::<R, BT, ElemwiseRunner>(
                &self.info.client,
                &self.info.device,
                context,
                &ElemwiseRunner,
            )
            .unwrap();

        output_read.merge(output_write)
    }
}

#[allow(clippy::too_many_arguments)]
impl<R: Runtime> ReduceOptimization<R> {
    pub fn new(
        trace: FuseTrace,
        trace_read_fallback: FuseTrace,
        trace_write_fallback: FuseTrace,
        client: ComputeClient<R::Server, R::Channel>,
        device: R::Device,
        len: usize,
        len_read: usize,
        reduce: FusedReduce,
    ) -> Self {
        let reduce_plane = reduce.with_strategy(ReduceStrategy {
            use_planes: true,
            shared: false,
        });
        let reduce_shared_plane = reduce.with_strategy(ReduceStrategy {
            use_planes: true,
            shared: true,
        });

        let info = ReduceOptimizationInfo {
            trace,
            trace_read_fallback,
            trace_write_fallback,
            client,
            device,
            len,
            len_read,
            reduce,
            reduce_plane,
            reduce_shared_plane,
        };

        Self {
            info: Arc::new(info),
        }
    }
    /// Execute the optimization.
    pub fn execute<BT: CubeElement>(
        &mut self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        fallback: impl FnOnce(usize) -> Box<dyn FallbackOperation<R>>,
    ) {
        // The index of the fallback reduce is the number of ops fused as read.
        let fallback = fallback(self.info.len_read);
        let arg = ReduceOptimizationTuneArg {
            info: self.info.clone(),
            fallback,
        };

        #[cfg(feature = "autotune")]
        fused_reduce_autotune::<R, BT>(arg, context);

        #[cfg(not(feature = "autotune"))]
        if arg.execute_fused_reduce::<BT>(context).is_err() {
            arg.execute_fallback::<BT>(context);
        }
    }

    pub fn num_output_buffers(&self) -> usize {
        self.info.trace_read_fallback.resources.outputs.len()
    }

    pub fn to_state(&self) -> ReduceOptimizationState {
        ReduceOptimizationState {
            trace: self.info.trace.clone(),
            trace_read_fallback: self.info.trace_read_fallback.clone(),
            trace_write_fallback: self.info.trace_write_fallback.clone(),
            reduce: self.info.reduce.clone(),
            reduce_plane: self.info.reduce_plane.clone(),
            reduce_shared_plane: self.info.reduce_shared_plane.clone(),
            len: self.info.len,
            len_read: self.info.len_read,
        }
    }

    pub fn from_state(device: &R::Device, state: ReduceOptimizationState) -> Self {
        let client = R::client(device);

        let info = ReduceOptimizationInfo {
            trace: state.trace,
            trace_read_fallback: state.trace_read_fallback,
            trace_write_fallback: state.trace_write_fallback,
            reduce: state.reduce,
            reduce_plane: state.reduce_plane,
            reduce_shared_plane: state.reduce_shared_plane,
            len: state.len,
            len_read: state.len_read,
            client,
            device: device.clone(),
        };

        Self {
            info: Arc::new(info),
        }
    }

    /// Returns the number of output buffers added by fusion.
    pub fn num_ops_fused(&self) -> usize {
        self.info.len
    }
}

impl<R: Runtime> Vectorization<R> for FusedReduce {}

impl<R: Runtime> TraceRunner<R> for FusedReduce {
    type Error = FusedReduceError;

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        configs: &'a [FuseBlockConfig],
    ) -> Result<(), FusedReduceError> {
        let [config_read, config_write] = [&configs[0], &configs[1]];
        self.strategy
            .validate::<R>(client)
            .map_err(FusedReduceError::LaunchError)?;

        let strategy = self.strategy;
        let shape = match &config_read.ref_layout {
            RefLayout::Concrete(Arg::Output(..)) => {
                outputs.shape_ref(&config_read.ref_layout, config_read.rank as usize)
            }
            _ => inputs.shape_ref(&config_read.ref_layout, config_read.rank as usize),
        };
        let reduce_count: u32 = shape
            .iter()
            .enumerate()
            .map(|(i, s)| if i == self.axis { 1 } else { *s as u32 })
            .product();

        let line_mode = match self.axis == config_read.rank as usize - 1 {
            true => LineMode::Parallel,
            false => LineMode::Perpendicular,
        };

        let config_reduce = ReduceConfig {
            cube_count: CubeCount::new_single(),
            cube_dim: CubeDim::new_single(),
            line_mode,
            line_size_input: config_read.width as u32,
            line_size_output: config_write.width as u32,
            bound_checks: false,
            bound_checks_inner: if strategy.use_planes {
                BoundChecksInner::Branch
            } else {
                BoundChecksInner::Mask
            },
        }
        .generate_cube_dim(client, strategy.use_planes)
        .generate_cube_count::<R>(reduce_count, &strategy);

        if let CubeCount::Static(x, y, z) = config_reduce.cube_count {
            let (max_x, max_y, max_z) = R::max_cube_count();
            if x > max_x || y > max_y || z > max_z {
                return Err(FusedReduceError::LaunchError(
                    ReduceError::CubeCountTooLarge,
                ));
            }
        }

        let kwargs = ReduceKwArgs {
            client,
            inputs,
            outputs,
            axis: self.axis as u32,
            strategy: &strategy,
            config_reduce,
            config_fuse_read: config_read.clone(),
            config_fuse_write: config_write.clone(),
            input: self.input.clone(),
            output: self.output.clone(),
        };
        launch_reduce_mixed_precision::<R>(
            kwargs,
            self.inst,
            self.op.input.dtype,
            self.op.out.dtype,
            DType::from(self.acc.into_elem()),
        );

        Ok(())
    }
}

struct ReduceKwArgs<'a, 'b, Run: Runtime> {
    client: &'b ComputeClient<Run::Server, Run::Channel>,
    inputs: GlobalArgsLaunch<'a, Run>,
    outputs: GlobalArgsLaunch<'a, Run>,
    axis: u32,
    strategy: &'b ReduceStrategy,
    config_reduce: ReduceConfig,
    config_fuse_read: FuseBlockConfig,
    config_fuse_write: FuseBlockConfig,
    input: Arg,
    output: Arg,
}

fn launch_reduce_mixed_precision<Run: Runtime>(
    kwargs: ReduceKwArgs<'_, '_, Run>,
    instruction: ReduceInstruction,
    dtype_input: DType,
    dtype_output: DType,
    dtype_acc: DType,
) {
    let config = match instruction {
        ReduceInstruction::ArgMax => ReduceFnConfig::ArgMax,
        ReduceInstruction::ArgMin => ReduceFnConfig::ArgMin,
        ReduceInstruction::Prod => ReduceFnConfig::Prod,
        ReduceInstruction::Mean => ReduceFnConfig::Mean,
        ReduceInstruction::Sum => ReduceFnConfig::Sum,
        ReduceInstruction::Max => ReduceFnConfig::Max,
        ReduceInstruction::Min => ReduceFnConfig::Min,
        ReduceInstruction::MaxAbs => ReduceFnConfig::MaxAbs,
    };
    launch_reduce::<Run, ReduceFn>(kwargs, config, dtype_input, dtype_output, dtype_acc)
}

fn launch_reduce<Run: Runtime, Rd: ReduceFamily>(
    kwargs: ReduceKwArgs<'_, '_, Run>,
    config: Rd::Config,
    dtype_input: DType,
    dtype_output: DType,
    dtype_acc: DType,
) {
    let settings = ReduceParams {
        shared: kwargs.strategy.shared.then(|| {
            if kwargs.strategy.use_planes {
                kwargs.config_reduce.cube_dim.y
            } else {
                kwargs.config_reduce.cube_dim.num_elems()
            }
        }),
        use_planes: kwargs.strategy.use_planes,
        line_size_input: kwargs.config_reduce.line_size_input,
        line_size_output: kwargs.config_reduce.line_size_output,
        line_mode: kwargs.config_reduce.line_mode,
        bound_checks: kwargs.config_reduce.bound_checks,
        bound_checks_inner: kwargs.config_reduce.bound_checks_inner,
    };

    unsafe {
        reduce_kernel::launch_unchecked::<Rd, Run>(
            kwargs.client,
            kwargs.config_reduce.cube_count,
            kwargs.config_reduce.cube_dim,
            FusedReduceInputLaunch::new(kwargs.inputs, kwargs.config_fuse_read, kwargs.input),
            FusedReduceOutputLaunch::new(kwargs.outputs, kwargs.config_fuse_write, kwargs.output),
            ScalarArg::new(kwargs.axis),
            settings,
            config,
            dtype_input.into(),
            dtype_output.into(),
            dtype_acc.into(),
        );
    }
}

const INPUT: u8 = 0;
const OUTPUT: u8 = 1;
const ACC: u8 = 2;

#[cube(launch_unchecked)]
pub fn reduce_kernel<R: ReduceFamily>(
    input: &FusedReduceInput,
    output: &mut FusedReduceOutput,
    axis_reduce: u32,
    #[comptime] params: ReduceParams,
    #[comptime] config: R::Config,
    #[comptime] elem_in: StorageType,
    #[comptime] elem_out: StorageType,
    #[comptime] elem_acc: StorageType,
) {
    set_polyfill::<NumericExpand<INPUT>>(elem_in);
    set_polyfill::<NumericExpand<OUTPUT>>(elem_out);
    set_polyfill::<NumericExpand<ACC>>(elem_acc);

    let (input, mut output) =
        init_tensors::<FusedReduceArgs, NumericExpand<INPUT>, NumericExpand<OUTPUT>>(input, output);

    reduce_kernel_virtual::<NumericExpand<INPUT>, NumericExpand<OUTPUT>, NumericExpand<ACC>, R>(
        &input,
        &mut output,
        axis_reduce,
        params,
        config,
    );
}
