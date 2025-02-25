use burn_fusion::stream::Context;
use burn_ir::ReduceDimOpIr;
use cubecl::prelude::*;
use cubecl::reduce::{reduce_kernel, Reduce, ReduceParams, ReduceStrategy};
use cubecl::{
    client::ComputeClient,
    reduce::{LineMode, ReduceConfig, ReduceError},
    CubeCount, CubeDim, Runtime,
};
use serde::{Deserialize, Serialize};

use crate::shared::trace::{MultiTraceRunner, Vectorization};
use crate::shared::{
    ir::{Arg, ElemwiseConfig, GlobalArgsLaunch},
    trace::FuseTrace,
};
use crate::CubeFusionHandle;

use super::args::{FusedReduceArgs, FusedReduceInputLaunch, FusedReduceOutputLaunch};

#[derive(new)]
pub struct ReduceOptimization<R: Runtime> {
    trace_read: FuseTrace,
    trace_write: FuseTrace,
    trace_read_fallback: FuseTrace,
    trace_write_fallback: FuseTrace,
    pub(crate) client: ComputeClient<R::Server, R::Channel>,
    pub(crate) device: R::Device,
    pub(crate) len: usize,
    pub(crate) fuse: FusedReduce,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReduceOptimizationState {
    trace_read: FuseTrace,
    trace_write: FuseTrace,
    trace_read_fallback: FuseTrace,
    trace_write_fallback: FuseTrace,
    pub(crate) fuse: FusedReduce,
    len: usize,
}

#[derive(new, Clone, Serialize, Deserialize, Debug)]
pub struct FusedReduce {
    input: Arg,
    output: Arg,
    axis: usize,
    pub(crate) op: ReduceDimOpIr,
}
#[derive(Debug)]
pub enum FusedReduceError {
    LaunchError(ReduceError),
    InvalidInput,
}
impl<R: Runtime> ReduceOptimization<R> {
    /// Execute the optimization.
    pub fn execute<BT: CubeElement>(&mut self, context: &mut Context<'_, CubeFusionHandle<R>>) {
        if self.execute_fused::<BT>(context).is_err() {
            // self.execute_fallback::<BT>(context);
            panic!("NOOOOOOOOOOOOOo");
        }
    }

    pub fn execute_fused<BT: CubeElement>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
    ) -> Result<(), FusedReduceError> {
        FuseTrace::run_multi::<R, BT, FusedReduce>(
            (&self.trace_read, &self.trace_write),
            &self.client,
            &self.device,
            context,
            &self.fuse,
        )
    }
    /// Returns the number of output buffers added by fusion.
    pub fn num_ops_fused(&self) -> usize {
        self.len
    }
}
impl<R: Runtime> Vectorization<R> for FusedReduce {}
impl<R: Runtime> MultiTraceRunner<R> for FusedReduce {
    type Error = FusedReduceError;

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config_read: &'a ElemwiseConfig,
        config_write: &'a ElemwiseConfig,
    ) -> Result<(), FusedReduceError> {
        let strategy = ReduceStrategy::new::<R>(client, true);
        let reduce_count: u32 = inputs
            .shape(&config_read.ref_layout)
            .iter()
            .enumerate()
            .map(|(i, s)| if i == self.axis { 1 } else { *s as u32 })
            .product();
        let config_reduce = ReduceConfig {
            cube_count: CubeCount::new_single(),
            cube_dim: CubeDim::new_single(),
            line_mode: LineMode::Parallel,
            line_size: inputs.line_size(&config_read.ref_layout) as u32,
            bound_checks: false,
        }
        .generate_cube_dim(client, strategy.use_planes)
        .generate_cube_count::<R>(reduce_count as u32, &strategy);

        if let CubeCount::Static(x, y, z) = config_reduce.cube_count {
            let (max_x, max_y, max_z) = R::max_cube_count();
            if x > max_x || y > max_y || z > max_z {
                return Err(FusedReduceError::LaunchError(
                    ReduceError::CubeCountTooLarge,
                ));
            }
        }

        launch_reduce::<R, f32, f32, cubecl::reduce::instructions::Sum>(
            client,
            inputs,
            outputs,
            self.axis as u32,
            &ReduceStrategy::new::<R>(client, true),
            config_reduce,
            config_read,
            config_write,
            &self.input,
            &self.output,
        );

        Ok(())
    }
}

fn launch_reduce<'a, Run: Runtime, In: Numeric, Out: Numeric, Rd: Reduce>(
    client: &ComputeClient<Run::Server, Run::Channel>,
    inputs: GlobalArgsLaunch<'a, Run>,
    outputs: GlobalArgsLaunch<'a, Run>,
    axis: u32,
    strategy: &ReduceStrategy,
    config_reduce: ReduceConfig,
    config_fuse_read: &'a ElemwiseConfig,
    config_fuse_write: &'a ElemwiseConfig,
    input: &'a Arg,
    output: &'a Arg,
) {
    let settings = ReduceParams {
        shared: strategy.shared.then(|| {
            if strategy.use_planes {
                config_reduce.cube_dim.y
            } else {
                config_reduce.cube_dim.num_elems()
            }
        }),
        use_planes: strategy.use_planes,
        line_size: config_reduce.line_size,
        line_mode: config_reduce.line_mode,
        bound_checks: config_reduce.bound_checks,
    };

    unsafe {
        reduce_kernel::launch_unchecked::<In, Out, Rd, FusedReduceArgs, Run>(
            client,
            config_reduce.cube_count,
            config_reduce.cube_dim,
            FusedReduceInputLaunch::new(inputs, &config_fuse_read, input),
            FusedReduceOutputLaunch::new(outputs, &config_fuse_write, output),
            ScalarArg::new(axis),
            settings,
        );
    }
}
