use super::args::{
    FusedReduceInput, FusedReduceInputLaunch, FusedReduceOutput, FusedReduceOutputLaunch,
};
use super::tune::fused_reduce_autotune;
use crate::engine::launch::FuseTraceLauncher;
use crate::engine::launch::runner::{TraceRunner, Vectorization};
use crate::{
    CubeFusionHandle, FallbackOperation,
    engine::{
        codegen::ir::{FuseArg, FuseBlockConfig, FuseType, GlobalArgsLaunch, RefLayout},
        trace::{FuseTrace, TraceError, TuneOutput},
    },
    optim::{elemwise::ElemwiseRunner, reduce::args::FusedReduceArgs},
};
use burn_fusion::stream::Context;
use burn_ir::ReduceDimOpIr;
use burn_std::DType;
use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, ir::StorageType, prelude::*};
use cubek::reduce::{
    BoundChecksInner, LineMode, ReduceError,
    components::instructions::ReduceOperationConfig,
    init_tensors,
    launch::{ReduceLaunchInfo, ReduceStrategy},
    routines::{
        CubeReduceBlueprint, GlobalReduceBlueprint, PlaneReduceBlueprint, ReduceBlueprint,
        ReduceBlueprintKind, reduce_kernel_virtual,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub struct ReduceOptimization<R: Runtime> {
    info: Arc<ReduceOptimizationInfo<R>>,
}

pub(crate) struct ReduceOptimizationInfo<R: Runtime> {
    pub(crate) trace: FuseTrace,
    trace_read_fallback: FuseTrace,
    trace_write_fallback: FuseTrace,
    pub(crate) client: ComputeClient<R>,
    pub(crate) device: R::Device,
    pub(crate) len: usize,
    pub(crate) len_read: usize,
    pub(crate) reduce: FusedReduce,
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
    strategy: ReduceStrategy,
}

#[derive(Debug)]
pub enum FusedReduceError {
    Reduce(ReduceError),
    InvalidSelection(Box<&'static str>),
    InvalidInput,
}

impl<R: Runtime> ReduceOptimizationTuneArg<R> {
    pub fn execute_fused<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        strategy: ReduceStrategy,
    ) -> Result<TuneOutput<R>, TraceError<FusedReduceError>> {
        let launch = FusedReduceLaunch::new(&self.info.reduce, strategy);
        let launcher = FuseTraceLauncher::new(&self.info.trace, &launch);
        launcher.launch::<BT>(&self.info.client, &self.info.device, context)
    }

    pub fn execute_fallback<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> TuneOutput<R> {
        let launcher = FuseTraceLauncher::new(&self.info.trace_read_fallback, &ElemwiseRunner);

        #[allow(unused_mut)] // It is used when `autotune-checks` is activated.
        let mut output_read = launcher
            .launch::<BT>(&self.info.client, &self.info.device, context)
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
                (out_desc.shape.dims.clone(), handle_out.clone()),
            );
        }

        let launcher = FuseTraceLauncher::new(&self.info.trace_write_fallback, &ElemwiseRunner);

        let output_write = launcher
            .launch::<BT>(&self.info.client, &self.info.device, context)
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
        client: ComputeClient<R>,
        device: R::Device,
        len: usize,
        len_read: usize,
        reduce: FusedReduce,
    ) -> Self {
        let info = ReduceOptimizationInfo {
            trace,
            trace_read_fallback,
            trace_write_fallback,
            client,
            device,
            len,
            len_read,
            reduce,
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
        let [config_read, config_write] = [&configs[0], &configs[1]];
        self.strategy
            .validate(client)
            .map_err(FusedReduceError::Reduce)?;

        let strategy = self.strategy;
        let shape = match &config_read.ref_layout {
            RefLayout::Concrete(FuseArg::Output(..)) => {
                outputs.shape_ref(&config_read.ref_layout, config_read.rank as usize)
            }
            _ => inputs.shape_ref(&config_read.ref_layout, config_read.rank as usize),
        };
        let reduce_count: u32 = shape
            .iter()
            .enumerate()
            .map(|(i, s)| if i == self.reduce.axis { 1 } else { *s as u32 })
            .product();

        let line_mode = match self.reduce.axis == config_read.rank as usize - 1 {
            true => LineMode::Parallel,
            false => LineMode::Perpendicular,
        };

        let launch_info = ReduceLaunchInfo {
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

        if let CubeCount::Static(x, y, z) = launch_info.cube_count {
            let (max_x, max_y, max_z) = R::max_cube_count();
            if x > max_x || y > max_y || z > max_z {
                return Err(FusedReduceError::Reduce(ReduceError::CubeCountTooLarge));
            }
        }

        let kwargs = ReduceKwArgs {
            client,
            inputs,
            outputs,
            axis: self.reduce.axis as u32,
            strategy: &strategy,
            info: launch_info,
            config_fuse_read: config_read.clone(),
            config_fuse_write: config_write.clone(),
            input: self.reduce.input.clone(),
            output: self.reduce.output.clone(),
        };
        let result = launch_reduce_mixed_precision(
            kwargs,
            self.reduce.inst,
            self.reduce.op.input.dtype,
            self.reduce.op.out.dtype,
            DType::from(self.reduce.acc.into_elem()),
        );

        match result {
            Ok(_) => Ok(()),
            Err(err) => Err(FusedReduceError::Reduce(ReduceError::Launch(err))),
        }
    }
}

struct ReduceKwArgs<'a, 'b, Run: Runtime> {
    client: &'b ComputeClient<Run>,
    inputs: GlobalArgsLaunch<'a, Run>,
    outputs: GlobalArgsLaunch<'a, Run>,
    axis: u32,
    strategy: &'b ReduceStrategy,
    info: ReduceLaunchInfo,
    config_fuse_read: FuseBlockConfig,
    config_fuse_write: FuseBlockConfig,
    input: FuseArg,
    output: FuseArg,
}

fn launch_reduce_mixed_precision<Run: Runtime>(
    kwargs: ReduceKwArgs<'_, '_, Run>,
    instruction: ReduceInstruction,
    dtype_input: DType,
    dtype_output: DType,
    dtype_acc: DType,
) -> Result<(), LaunchError> {
    let config = match instruction {
        ReduceInstruction::ArgMax => ReduceOperationConfig::ArgMax,
        ReduceInstruction::ArgMin => ReduceOperationConfig::ArgMin,
        ReduceInstruction::Prod => ReduceOperationConfig::Prod,
        ReduceInstruction::Mean => ReduceOperationConfig::Mean,
        ReduceInstruction::Sum => ReduceOperationConfig::Sum,
        ReduceInstruction::Max => ReduceOperationConfig::Max,
        ReduceInstruction::Min => ReduceOperationConfig::Min,
        ReduceInstruction::MaxAbs => ReduceOperationConfig::MaxAbs,
    };
    launch_reduce::<Run>(kwargs, config, dtype_input, dtype_output, dtype_acc)
}

fn launch_reduce<Run: Runtime>(
    kwargs: ReduceKwArgs<'_, '_, Run>,
    inst: ReduceOperationConfig,
    dtype_input: DType,
    dtype_output: DType,
    dtype_acc: DType,
) -> Result<(), LaunchError> {
    let kind = match (kwargs.strategy.shared, kwargs.strategy.use_planes) {
        (true, true) => GlobalReduceBlueprint::Cube(CubeReduceBlueprint {
            accumulator_size: kwargs.info.cube_dim.y,
            bound_checks_inner: kwargs.info.bound_checks_inner,
            use_planes: true,
        }),
        (true, false) => ReduceBlueprintKind::Cube(CubeReduceBlueprint {
            accumulator_size: kwargs.info.cube_dim.num_elems(),
            bound_checks_inner: kwargs.info.bound_checks_inner,
            use_planes: false,
        }),
        (false, true) => ReduceBlueprintKind::Plane(PlaneReduceBlueprint {
            bound_checks_inner: kwargs.info.bound_checks_inner,
        }),
        (false, false) => ReduceBlueprintKind::Unit,
    };

    let blueprint = ReduceBlueprint {
        line_mode: kwargs.info.line_mode,
        bound_checks: kwargs.info.bound_checks,
        kind,
    };

    unsafe {
        reduce_kernel::launch_unchecked::<Run>(
            kwargs.client,
            kwargs.info.cube_count,
            kwargs.info.cube_dim,
            FusedReduceInputLaunch::new(kwargs.inputs, kwargs.config_fuse_read, kwargs.input),
            FusedReduceOutputLaunch::new(kwargs.outputs, kwargs.config_fuse_write, kwargs.output),
            ScalarArg::new(kwargs.axis),
            blueprint,
            inst,
            dtype_input.into(),
            dtype_output.into(),
            dtype_acc.into(),
        )
    }
}

#[cube(launch_unchecked)]
pub fn reduce_kernel<In: Numeric, Out: Numeric, Acc: Numeric>(
    input: &FusedReduceInput,
    output: &mut FusedReduceOutput,
    axis_reduce: u32,
    #[comptime] blueprint: ReduceBlueprint,
    #[comptime] config: ReduceOperationConfig,
    #[define(In)] _input_dtype: StorageType,
    #[define(Out)] _output_dtype: StorageType,
    #[define(Acc)] _acc_dtype: StorageType,
) {
    let (input, mut output) = init_tensors::<FusedReduceArgs, In, Out>(input, output);

    reduce_kernel_virtual::<In, Out, Acc>(&input, &mut output, axis_reduce, blueprint, config);
}
