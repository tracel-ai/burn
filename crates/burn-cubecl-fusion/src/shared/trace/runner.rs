use super::{
    super::ir::{FuseBlockConfig, GlobalArgsLaunch},
    vectorization::{Vect, vectorization_default},
};
use crate::CubeFusionHandle;
use burn_fusion::stream::Context;
use burn_ir::{TensorId, TensorIr};
use cubecl::{ir::Elem, prelude::*};
use std::collections::BTreeMap;

/// A trace runner is responsible for determining the vectorization factor as well as launching
/// a kernel based on global [inputs](GlobalArgsLaunch) and [outputs](GlobalArgsLaunch)
/// with a provided [element wise config](ElemwiseConfig).
pub trait TraceRunner<R: Runtime>: Vectorization<R> {
    /// The error that might happen while running the trace.
    type Error;

    /// Run the trace with the given inputs and outputs.
    ///
    /// There is one [fuse config](FuseBlockConfig) for each [block](super::block::FuseBlock) registered
    /// in the [optimization builder](burn_fusion::OptimizationBuilder).
    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R::Server, R::Channel>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        configs: &'a [FuseBlockConfig],
    ) -> Result<(), Self::Error>;
}

pub enum VectorizationHandle<'a, R: Runtime> {
    NormalInput(&'a CubeFusionHandle<R>, &'a TensorIr),
    QuantData(&'a CubeFusionHandle<R>, &'a TensorIr),
    QuantScales(&'a CubeFusionHandle<R>),
}

impl<'a, R: Runtime> VectorizationHandle<'a, R> {
    pub fn from_tensor(&self, id: TensorId) -> bool {
        match self {
            VectorizationHandle::NormalInput(_, tensor_ir) => tensor_ir.id == id,
            VectorizationHandle::QuantData(_, tensor_ir) => tensor_ir.id == id,
            VectorizationHandle::QuantScales(_) => false,
        }
    }
}

pub trait Vectorization<R: Runtime> {
    fn axis(&self) -> Option<usize> {
        None
    }
    /// The vectorization factor for all inputs and outputs.
    #[allow(clippy::too_many_arguments)]
    fn vectorization<'a>(
        &self,
        _context: &Context<'_, CubeFusionHandle<R>>,
        vectorizations: &mut BTreeMap<TensorId, Vect>,
        inputs: impl Iterator<Item = VectorizationHandle<'a, R>>,
        outputs: impl Iterator<Item = &'a TensorIr>,
        reshaped: impl Iterator<Item = (&'a TensorIr, &'a TensorIr, bool)>,
        swapped: impl Iterator<Item = (&'a TensorIr, &'a TensorIr, bool, &'a (u32, u32))>,
        ref_elem: &Elem,
        max: u8,
        axis: Option<usize>,
    ) {
        vectorization_default(
            vectorizations,
            inputs,
            outputs,
            reshaped,
            swapped,
            ref_elem,
            &Default::default(),
            max,
            axis,
        )
    }
}
