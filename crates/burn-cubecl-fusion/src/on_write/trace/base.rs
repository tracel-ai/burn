use crate::CubeFusionHandle;

use super::{
    super::{
        ir::{ElemwiseOp, ElemwisePrecision},
        settings::FuseSettings,
    },
    executor::LaunchPlanExecutor,
    input::InputPlanner,
    output::OutputPlanner,
    vectorization::VectorizationPlanner,
    HandleInput, HandleOutput, LaunchPlan, TraceRunner,
};
use burn_fusion::stream::Context;
use burn_ir::{TensorId, TensorIr};
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Serialize, Deserialize, Debug)]
/// Trace containing all element wise operations as well as reads and writes.
pub struct FuseOnWriteTrace {
    pub outputs: RegisteredTensors,
    pub inputs: RegisteredTensors,
    pub settings: FuseSettings,
    pub scalars: Vec<(ElemwisePrecision, u32)>,
    pub views: Vec<TensorView>,
    pub indexed: BTreeSet<TensorId>,
    pub shape_ref: Vec<usize>,
    pub ops: Vec<ElemwiseOp>,
    pub reads: BTreeMap<TensorId, Vec<ElemwiseOp>>,
    pub writes: BTreeMap<TensorId, ElemwiseOp>,
    pub inputs_unhandled: Vec<TensorId>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum TensorView {
    Reshape {
        reshaped: TensorId,
        original: TensorId,
    },
    SwapDims {
        swapped: TensorId,
        original: TensorId,
        dims: (u32, u32),
    },
}

impl FuseOnWriteTrace {
    /// Run a trace with the given [runner](TraceRunner).
    pub fn run<R: Runtime, BT: CubeElement, Runner: TraceRunner<R>>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        runner: &Runner,
    ) -> Result<(), Runner::Error> {
        let mut plan = LaunchPlan::new(&self.reads, &self.writes, self.shape_ref.len());

        InputPlanner::<R>::new(
            &self.inputs,
            &self.inputs_unhandled,
            &self.views,
            &self.shape_ref,
            &self.settings,
        )
        .run(context, &mut plan);

        OutputPlanner::<R>::new(&self.inputs, &self.outputs, &self.views)
            .run::<BT>(client, device, context, &mut plan);

        VectorizationPlanner::<R>::new(&self.views, &self.reads, &self.indexed)
            .run::<Runner>(context, &mut plan);

        match LaunchPlanExecutor::<R>::new(&self.scalars, &self.views, &self.ops)
            .execute::<_, BT>(client, runner, context, plan)
        {
            Err(err) => {
                self.rollback(context, err.handles_input, err.handles_output);
                Err(err.runner_error)
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
    tensors: Vec<(TensorIr, ElemwisePrecision)>,
}

impl RegisteredTensors {
    pub fn iter(&self) -> impl Iterator<Item = &(TensorIr, ElemwisePrecision)> {
        self.tensors.iter()
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

    pub fn get(&self, tensor_id: TensorId) -> Option<&(TensorIr, ElemwisePrecision)> {
        self.tensors
            .iter()
            .find(|(tensor, _)| tensor.id == tensor_id)
    }

    pub fn insert(&mut self, precision: ElemwisePrecision, tensor: TensorIr) -> u32 {
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
