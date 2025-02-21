use burn_ir::BinaryOpIr;
use cubecl::{client::ComputeClient, Runtime};
use serde::{Deserialize, Serialize};

use crate::shared::{ir::{Arg, ElemwiseConfig, GlobalArgsLaunch}, trace::{FuseTrace, TraceRunner}};

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
    LaunchError(String),
    InvalidInput,
}

impl<R: Runtime> TraceRunner<R> for FusedReduce {
    type Error = String;

    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        config: &'a ElemwiseConfig,
    ) -> Result<(), String> {
        Ok(())
    }
}
