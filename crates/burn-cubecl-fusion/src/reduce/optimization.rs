use std::sync::Arc;

use burn_fusion::stream::Context;
use burn_ir::{ReduceDimOpIr, TensorStatus};
use burn_tensor::DType;
use cubecl::prelude::*;
use cubecl::reduce::instructions::{ReduceFn, ReduceFnConfig};
use cubecl::reduce::{BoundChecksInner, ReduceFamily, ReduceParams, ReduceStrategy, reduce_kernel};
use cubecl::{
    CubeCount, CubeDim, Runtime,
    client::ComputeClient,
    reduce::{LineMode, ReduceConfig, ReduceError},
};
use serde::{Deserialize, Serialize};

use crate::CubeFusionHandle;
use crate::elemwise::optimization::ElemwiseRunner;
use crate::shared::ir::RefLayout;
use crate::shared::trace::{TraceError, TraceRunner};
use crate::shared::trace::{TuneOutput, Vectorization};
use crate::shared::{
    ir::{Arg, FuseBlockConfig, GlobalArgsLaunch},
    trace::FuseTrace,
};

use super::args::{FusedReduceArgs, FusedReduceInputLaunch, FusedReduceOutputLaunch};
use super::tune::fused_reduce_autotune;

pub struct ReduceOptimization<R: Runtime> {
    pub(crate) trace: FuseTrace,
    trace_read_fallback: FuseTrace,
    trace_write_fallback: FuseTrace,
    pub(crate) client: ComputeClient<R::Server, R::Channel>,
    pub(crate) device: R::Device,
    pub(crate) len: usize,
    pub(crate) reduce: FusedReduce,
    pub(crate) reduce_plane: FusedReduce,
    pub(crate) reduce_shared_plane: FusedReduce,
    fallback: Arc<dyn ReduceFallbackFn<R>>,
}

#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub enum ReduceInstruction {
    ArgMax,
    ArgMin,
    Mean,
    Prod,
    Sum,
}

pub trait ReduceFallbackFn<R: Runtime>: Send + Sync {
    fn run(
        &self,
        input_handle: CubeFusionHandle<R>,
        shape: &[usize],
        axis: usize,
        inst: &ReduceInstruction,
        dtype_out: &DType,
    ) -> CubeFusionHandle<R>;
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReduceOptimizationState {
    trace: FuseTrace,
    trace_read_fallback: FuseTrace,
    trace_write_fallback: FuseTrace,
    pub(crate) reduce: FusedReduce,
    pub(crate) reduce_plane: FusedReduce,
    pub(crate) reduce_shared_plane: FusedReduce,
    len: usize,
}

#[derive(new, Clone, Serialize, Deserialize, Debug)]
pub struct FusedReduce {
    input: Arg,
    output: Arg,
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

#[allow(clippy::too_many_arguments)]
impl<R: Runtime> ReduceOptimization<R> {
    pub fn new(
        trace: FuseTrace,
        trace_read_fallback: FuseTrace,
        trace_write_fallback: FuseTrace,
        client: ComputeClient<R::Server, R::Channel>,
        device: R::Device,
        len: usize,
        reduce: FusedReduce,
        fallback: Arc<dyn ReduceFallbackFn<R>>,
    ) -> Self {
        let reduce_plane = reduce.with_strategy(ReduceStrategy {
            use_planes: true,
            shared: false,
        });
        let reduce_shared_plane = reduce.with_strategy(ReduceStrategy {
            use_planes: true,
            shared: true,
        });
        Self {
            trace,
            trace_read_fallback,
            trace_write_fallback,
            client,
            device,
            len,
            reduce,
            reduce_plane,
            reduce_shared_plane,
            fallback,
        }
    }
    /// Execute the optimization.
    pub fn execute<BT: CubeElement>(&mut self, context: &mut Context<'_, CubeFusionHandle<R>>) {
        #[cfg(feature = "autotune")]
        fused_reduce_autotune::<R, BT>(self, context);

        #[cfg(not(feature = "autotune"))]
        if self.execute_fused_reduce::<BT>(context).is_err() {
            self.execute_fallback::<BT>(context);
        }
    }

    pub fn num_output_buffers(&self) -> usize {
        self.trace_read_fallback.resources.outputs.len()
    }

    pub fn to_state(&self) -> ReduceOptimizationState {
        ReduceOptimizationState {
            trace: self.trace.clone(),
            trace_read_fallback: self.trace_read_fallback.clone(),
            trace_write_fallback: self.trace_write_fallback.clone(),
            reduce: self.reduce.clone(),
            reduce_plane: self.reduce_plane.clone(),
            reduce_shared_plane: self.reduce_shared_plane.clone(),
            len: self.len,
        }
    }

    pub fn from_state(
        device: &R::Device,
        state: ReduceOptimizationState,
        fallback: Arc<dyn ReduceFallbackFn<R>>,
    ) -> Self {
        let client = R::client(device);

        Self {
            trace: state.trace,
            trace_read_fallback: state.trace_read_fallback,
            trace_write_fallback: state.trace_write_fallback,
            reduce: state.reduce,
            reduce_plane: state.reduce_plane,
            reduce_shared_plane: state.reduce_shared_plane,
            fallback,
            len: state.len,
            client,
            device: device.clone(),
        }
    }

    pub fn execute_fused_reduce<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<TuneOutput<R>, TraceError<FusedReduceError>> {
        FuseTrace::run::<R, BT, FusedReduce>(
            &self.trace,
            &self.client,
            &self.device,
            context,
            &self.reduce,
        )
    }

    pub fn execute_fused_reduce_plane<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<TuneOutput<R>, TraceError<FusedReduceError>> {
        FuseTrace::run::<R, BT, FusedReduce>(
            &self.trace,
            &self.client,
            &self.device,
            context,
            &self.reduce_plane,
        )
    }

    pub fn execute_fused_reduce_shared_plane<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<TuneOutput<R>, TraceError<FusedReduceError>> {
        FuseTrace::run::<R, BT, FusedReduce>(
            &self.trace,
            &self.client,
            &self.device,
            context,
            &self.reduce_shared_plane,
        )
    }

    pub fn execute_fallback<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> TuneOutput<R> {
        #[allow(unused_mut)] // It is used when #[cfg(test)] is true.
        let mut output_read = self
            .trace_read_fallback
            .run::<R, BT, ElemwiseRunner>(&self.client, &self.device, context, &ElemwiseRunner)
            .unwrap();

        let (out_tensor, out_desc) = {
            let input = context
                .tensors
                .get(&self.reduce.op.input.id)
                .unwrap()
                .clone();
            let out = context.tensors.get(&self.reduce.op.out.id).unwrap().clone();

            let input_handle = context
                .handles
                .get_handle(&input.id, &TensorStatus::ReadOnly);
            let out_handle = self.fallback.run(
                input_handle,
                &input.shape,
                self.reduce.op.axis,
                &self.reduce.inst,
                &self.reduce.op.out.dtype,
            );

            (out_handle, out)
        };
        #[cfg(feature = "autotune-checks")]
        if let TuneOutput::Checked { handles } = &mut output_read {
            handles.insert(
                self.reduce.op.out.id,
                (out_desc.shape.clone(), out_tensor.clone()),
            );
        }
        context.handles.register_handle(out_desc.id, out_tensor);
        let output_write = self
            .trace_write_fallback
            .run::<R, BT, ElemwiseRunner>(&self.client, &self.device, context, &ElemwiseRunner)
            .unwrap();

        output_read.merge(output_write)
    }
    /// Returns the number of output buffers added by fusion.
    pub fn num_ops_fused(&self) -> usize {
        self.len
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
            true => LineMode::Parallel, // axis de vectorization == axis de reduce.
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
            config_fuse_read: config_read,
            config_fuse_write: config_write,
            input: &self.input,
            output: &self.output,
        };
        launch_reduce_input_output_inst::<R>(
            kwargs,
            self.inst,
            self.op.input.dtype,
            self.op.out.dtype,
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
    config_fuse_read: &'a FuseBlockConfig,
    config_fuse_write: &'a FuseBlockConfig,
    input: &'a Arg,
    output: &'a Arg,
}

fn launch_reduce_input_output_inst<Run: Runtime>(
    kwargs: ReduceKwArgs<'_, '_, Run>,
    instruction: ReduceInstruction,
    dtype_input: DType,
    dtype_output: DType,
) {
    let config = match instruction {
        ReduceInstruction::ArgMax => ReduceFnConfig::ArgMax,
        ReduceInstruction::ArgMin => ReduceFnConfig::ArgMin,
        ReduceInstruction::Prod => ReduceFnConfig::Prod,
        ReduceInstruction::Mean => ReduceFnConfig::Mean,
        ReduceInstruction::Sum => ReduceFnConfig::Sum,
    };
    launch_reduce_input_output::<Run, ReduceFn>(kwargs, dtype_input, dtype_output, config)
}

fn launch_reduce_input_output<Run: Runtime, Rd: ReduceFamily>(
    kwargs: ReduceKwArgs<'_, '_, Run>,
    dtype_input: DType,
    dtype_output: DType,
    config: Rd::Config,
) {
    match dtype_input {
        DType::F64 => launch_reduce_output::<Run, f64, Rd>(kwargs, dtype_output, config),
        DType::F32 => launch_reduce_output::<Run, f32, Rd>(kwargs, dtype_output, config),
        DType::F16 => launch_reduce_output::<Run, half::f16, Rd>(kwargs, dtype_output, config),
        DType::BF16 => launch_reduce_output::<Run, half::bf16, Rd>(kwargs, dtype_output, config),
        DType::I64 => launch_reduce_output::<Run, i64, Rd>(kwargs, dtype_output, config),
        DType::I32 => launch_reduce_output::<Run, i32, Rd>(kwargs, dtype_output, config),
        DType::I16 => launch_reduce_output::<Run, i16, Rd>(kwargs, dtype_output, config),
        DType::I8 => launch_reduce_output::<Run, i8, Rd>(kwargs, dtype_output, config),
        DType::U64 => launch_reduce_output::<Run, u64, Rd>(kwargs, dtype_output, config),
        DType::U32 => launch_reduce_output::<Run, u32, Rd>(kwargs, dtype_output, config),
        DType::U16 => launch_reduce_output::<Run, u16, Rd>(kwargs, dtype_output, config),
        DType::U8 => launch_reduce_output::<Run, u8, Rd>(kwargs, dtype_output, config),
        _ => panic!("Unsupported"),
    }
}

fn launch_reduce_output<Run: Runtime, In: Numeric, Rd: ReduceFamily>(
    kwargs: ReduceKwArgs<'_, '_, Run>,
    dtype: DType,
    config: Rd::Config,
) {
    match dtype {
        DType::F64 => launch_reduce::<Run, In, f64, Rd>(kwargs, config),
        DType::F32 => launch_reduce::<Run, In, f32, Rd>(kwargs, config),
        DType::F16 => launch_reduce::<Run, In, half::f16, Rd>(kwargs, config),
        DType::BF16 => launch_reduce::<Run, In, half::bf16, Rd>(kwargs, config),
        DType::I64 => launch_reduce::<Run, In, i64, Rd>(kwargs, config),
        DType::I32 => launch_reduce::<Run, In, i32, Rd>(kwargs, config),
        DType::I16 => launch_reduce::<Run, In, i16, Rd>(kwargs, config),
        DType::I8 => launch_reduce::<Run, In, i8, Rd>(kwargs, config),
        DType::U64 => launch_reduce::<Run, In, u64, Rd>(kwargs, config),
        DType::U32 => launch_reduce::<Run, In, u32, Rd>(kwargs, config),
        DType::U16 => launch_reduce::<Run, In, u16, Rd>(kwargs, config),
        DType::U8 => launch_reduce::<Run, In, u8, Rd>(kwargs, config),
        _ => panic!("Unsupported"),
    }
}

fn launch_reduce<Run: Runtime, In: Numeric, Out: Numeric, Rd: ReduceFamily>(
    kwargs: ReduceKwArgs<'_, '_, Run>,
    config: Rd::Config,
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
        reduce_kernel::launch_unchecked::<In, Out, Rd, FusedReduceArgs, Run>(
            kwargs.client,
            kwargs.config_reduce.cube_count,
            kwargs.config_reduce.cube_dim,
            FusedReduceInputLaunch::new(kwargs.inputs, kwargs.config_fuse_read, kwargs.input),
            FusedReduceOutputLaunch::new(kwargs.outputs, kwargs.config_fuse_write, kwargs.output),
            ScalarArg::new(kwargs.axis),
            settings,
            config,
        );
    }
}
