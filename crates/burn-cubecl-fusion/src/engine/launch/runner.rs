use super::super::codegen::ir::{FuseBlockConfig, GlobalArgsLaunch};
use crate::{
    CubeFusionHandle,
    engine::launch::{
        LaunchPlan,
        vectorization::{Vect, vectorization_default},
    },
};
use burn_fusion::stream::Context;
use burn_ir::{TensorId, TensorIr};
use cubecl::prelude::*;
use std::collections::{BTreeMap, HashMap};

/// A trace runner is responsible for determining the vectorization factor as well as launching
/// a kernel based on global [inputs](GlobalArgsLaunch) and [outputs](GlobalArgsLaunch)
/// with provided [fuse block configs](FuseBlockConfig).
pub trait TraceRunner<R: Runtime>: Vectorization<R> {
    /// The error that might happen while running the trace.
    type Error;

    /// Run the trace with the given inputs and outputs.
    ///
    /// There is one [fuse config](FuseBlockConfig) for each [block](super::block::FuseBlock) registered
    /// in the [optimization builder](burn_fusion::OptimizationBuilder).
    fn run<'a>(
        &'a self,
        client: &'a ComputeClient<R>,
        inputs: GlobalArgsLaunch<'a, R>,
        outputs: GlobalArgsLaunch<'a, R>,
        configs: &'a [FuseBlockConfig],
    ) -> Result<(), Self::Error>;
}

pub enum VectorizationHandle<'a, R: Runtime> {
    NormalInput(&'a CubeFusionHandle<R>, &'a TensorIr),
    QuantValues(&'a CubeFusionHandle<R>, &'a TensorIr),
    QuantParams,
}

impl<'a, R: Runtime> VectorizationHandle<'a, R> {
    /// Returns if the current vectorization handle is from the given tensor id.
    pub fn is_from_tensor(&self, id: TensorId) -> bool {
        match self {
            VectorizationHandle::NormalInput(_, tensor_ir) => tensor_ir.id == id,
            VectorizationHandle::QuantValues(_, tensor_ir) => tensor_ir.id == id,
            VectorizationHandle::QuantParams => false,
        }
    }
}

#[derive(Default)]
pub struct VectorizationAxis {
    axis: HashMap<TensorId, usize>,
}

impl VectorizationAxis {
    pub fn get<F: FnOnce() -> usize>(&self, id: TensorId, default: F) -> usize {
        self.axis.get(&id).copied().unwrap_or_else(default)
    }
    pub fn insert(&mut self, id: TensorId, axis: usize) {
        self.axis.insert(id, axis);
    }
}

pub trait Vectorization<R: Runtime> {
    /// Returns the vectorization options.
    fn axis(&self, _plan: &LaunchPlan<'_, R>) -> VectorizationAxis {
        VectorizationAxis::default()
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
        line_sizes: &[u8],
        max: u8,
        axis: VectorizationAxis,
    ) {
        vectorization_default(
            vectorizations,
            inputs,
            outputs,
            reshaped,
            swapped,
            line_sizes,
            &Default::default(),
            max,
            &axis,
        )
    }
}
