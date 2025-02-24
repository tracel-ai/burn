use burn_ir::BinaryOpIr;
use cubecl::prelude::*;
use cubecl::reduce::{reduce_kernel, Reduce, ReduceParams, ReduceStrategy};
use cubecl::{
    client::ComputeClient,
    reduce::{LineMode, ReduceConfig, ReduceError},
    CubeCount, CubeDim, Runtime,
};
use serde::{Deserialize, Serialize};

use crate::shared::{
    ir::{Arg, ElemwiseConfig, GlobalArgsLaunch},
    trace::{FuseTrace, TraceRunner},
};

use super::args::{
    FusedReduceArgs, FusedReduceInput, FusedReduceInputLaunch, FusedReduceOutput,
    FusedReduceOutputLaunch,
};

pub struct ReduceOptimization<R: Runtime> {
    trace_read: FuseTrace,
    trace_write: FuseTrace,
    pub(crate) client: ComputeClient<R::Server, R::Channel>,
    pub(crate) device: R::Device,
    pub(crate) len: usize,
    pub(crate) fuse: FusedReduce,
}

#[derive(new, Clone, Serialize, Deserialize, Debug)]
pub struct FusedReduce {
    lhs: Arg,
    rhs: Arg,
    out: Arg,
    pub(crate) op: BinaryOpIr,
}
#[derive(Debug)]
pub enum FusedReduceError {
    LaunchError(ReduceError),
    InvalidInput,
}

impl<R: Runtime> TraceRunner<R> for FusedReduce {
    type Error = FusedReduceError;

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: &'a ElemwiseConfig,
    ) -> Result<(), FusedReduceError> {
        let config = ReduceConfig {
            cube_count: CubeCount::new_single(),
            cube_dim: CubeDim::new_single(),
            line_mode: LineMode::Parallel,
            line_size: 1,
            bound_checks: true,
        };
        if let CubeCount::Static(x, y, z) = config.cube_count {
            let (max_x, max_y, max_z) = R::max_cube_count();
            if x > max_x || y > max_y || z > max_z {
                return Err(FusedReduceError::LaunchError(
                    ReduceError::CubeCountTooLarge,
                ));
            }
        }

        // reduce_kernel(input, output, axis_reduce, config);

        Ok(())
    }
}

fn launch_reduce<'a, Run: Runtime, In: Numeric, Out: Numeric, Rd: Reduce>(
    client: &ComputeClient<Run::Server, Run::Channel>,
    inputs: GlobalArgsLaunch<'a, Run>,
    outputs: GlobalArgsLaunch<'a, Run>,
    axis: u32,
    strategy: ReduceStrategy,
    config_reduce: ReduceConfig,
    config_fuse: ElemwiseConfig,
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
            FusedReduceInputLaunch::new(inputs, &config_fuse, todo!()),
            FusedReduceOutputLaunch::new(outputs, &config_fuse, todo!()),
            ScalarArg::new(axis),
            settings,
        );
    }
}
