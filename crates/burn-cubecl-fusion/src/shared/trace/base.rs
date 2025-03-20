use crate::{
    CubeFusionHandle,
    shared::ir::{Arg, FusePrecision},
};

use super::{
    HandleInput, HandleOutput, LaunchPlan, TraceRunner, block::FuseBlock,
    executor::LaunchPlanExecutor, input::InputPlanner, output::OutputPlanner,
    vectorization::VectorizationPlanner,
};
use burn_fusion::stream::Context;
use burn_ir::{TensorId, TensorIr};
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Serialize, Deserialize, Debug)]
/// A trace constains all [blocks](FuseBlock) and the [resources](KernelResources) used by the
/// kernel.
pub struct FuseTrace {
    pub blocks: Vec<FuseBlock>,
    pub resources: FuseResources,
}

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
/// Declare all resources used by the kernel, and potentially multiple [blocks](FuseBlock).
///
/// # Notes
///
/// Each block can't contain their own resources, since they are shared between blocks. The
/// vectorization factor of one input tensor must be the same for all blocks.
pub struct FuseResources {
    pub outputs: RegisteredTensors,
    pub inputs: RegisteredTensors,
    pub scalars: Vec<(FusePrecision, u32)>,
    pub views: Vec<TensorView>,
    pub indexed: BTreeMap<TensorId, Arg>,
    pub inputs_unhandled: Vec<TensorId>,
    pub outputs_unhandled: Vec<Arg>,
    pub num_reshaped: usize,
}

#[derive(Debug)]
pub enum TraceError<Err> {
    ReferenceNotFound,
    RunnerError(Err),
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum TensorView {
    Reshape {
        reshaped: TensorId,
        original: TensorId,
        reshape_pos: u32,
        shape_relative: Vec<usize>,
    },
    SwapDims {
        swapped: TensorId,
        original: TensorId,
        dims: (u32, u32),
    },
}

impl FuseTrace {
    /// Run a trace with the given [runner](TraceRunner).
    pub fn run<R: Runtime, BT: CubeElement, Runner: TraceRunner<R>>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        runner: &Runner,
    ) -> Result<(), TraceError<Runner::Error>> {
        let mut plan = LaunchPlan::<R>::new(&self.blocks);

        InputPlanner::<R>::new(&self.resources, &self.blocks).run(context, &mut plan);

        OutputPlanner::<R>::new(&self.resources).run::<BT>(client, device, context, &mut plan);

        VectorizationPlanner::<R>::new(&self.resources, &self.blocks)
            .run(runner, context, &mut plan);

        match LaunchPlanExecutor::<R>::new(&self.resources, &self.blocks)
            .execute::<_, BT>(client, runner, context, plan)
        {
            Err(err) => {
                self.rollback(context, err.handles_input, err.handles_output);
                Err(err.error)
            }
            Ok(val) => Ok(val),
        }
    }

    fn rollback<R: Runtime>(
        &self,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        handle_inputs: Vec<HandleInput<R>>,
        handle_outputs: Vec<HandleOutput<R>>,
    ) {
        for input in handle_inputs {
            context
                .handles
                .register_handle(input.global_id, input.handle);
        }
        for output in handle_outputs {
            if let HandleOutput::Owned {
                global_id, handle, ..
            } = output
            {
                context.handles.register_handle(global_id, handle);
            }
        }
    }
}

#[derive(Default, Clone, Serialize, Deserialize, Debug)]
pub struct RegisteredTensors {
    tensors: Vec<(TensorIr, FusePrecision)>,
}

impl RegisteredTensors {
    pub fn iter(&self) -> impl Iterator<Item = &(TensorIr, FusePrecision)> {
        self.tensors.iter()
    }
    pub fn into_iter(self) -> impl Iterator<Item = (TensorIr, FusePrecision)> {
        self.tensors.into_iter()
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn get_index(&self, tensor_id: TensorId) -> Option<u32> {
        self.tensors
            .iter()
            .enumerate()
            .find(|(_pos, (tensor, _))| tensor.id == tensor_id)
            .map(|(pos, (_, _))| pos as u32)
    }

    pub fn get(&self, tensor_id: TensorId) -> Option<&(TensorIr, FusePrecision)> {
        self.tensors
            .iter()
            .find(|(tensor, _)| tensor.id == tensor_id)
    }

    pub fn insert(&mut self, precision: FusePrecision, tensor: TensorIr) -> u32 {
        let value = (tensor, precision);
        if let Some(old) = self
            .tensors
            .iter()
            .enumerate()
            .find(|(_, val)| *val == &value)
        {
            return old.0 as u32;
        }

        let pos = self.len();
        self.tensors.push(value);
        pos as u32
    }

    pub fn update(&mut self, tensor: &TensorIr) {
        if let Some((tensor_old, _)) = self
            .tensors
            .iter_mut()
            .find(|(tensor_old, _)| tensor_old.id == tensor.id)
        {
            tensor_old.status = tensor.status.clone();
        }
    }
}
