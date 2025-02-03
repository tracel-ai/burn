use crate::{fusion::JitFusionHandle, BoolElement, JitRuntime};

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
use burn_tensor::repr::{TensorDescription, TensorId};
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Serialize, Deserialize, Debug)]
/// Trace containing all element wise operations as well as reads and writes.
pub struct FuseOnWriteTrace {
    pub outputs: RegisteredTensors,
    pub inputs: RegisteredTensors,
    pub settings: FuseSettings,
    pub scalars: BTreeMap<ElemwisePrecision, u32>,
    pub reshapes: Vec<Reshape>,
    pub shape_ref: Vec<usize>,
    pub ops: Vec<ElemwiseOp>,
    pub reads: BTreeMap<TensorId, Vec<ElemwiseOp>>,
    pub writes: BTreeMap<TensorId, ElemwiseOp>,
    pub inputs_unhandled: Vec<TensorId>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Reshape {
    pub reshaped: TensorId,
    pub original: TensorId,
}

impl FuseOnWriteTrace {
    /// Run a trace with the given [runner](TraceRunner).
    pub fn run<R: JitRuntime, BT: BoolElement, Runner: TraceRunner<R>>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'_, JitFusionHandle<R>>,
        runner: &Runner,
    ) -> Result<(), Runner::Error> {
        let mut plan = LaunchPlan::new(&self.reads, &self.writes, self.shape_ref.len());

        InputPlanner::<R>::new(
            &self.inputs,
            &self.inputs_unhandled,
            &self.reshapes,
            &self.shape_ref,
            &self.settings,
        )
        .run(context, &mut plan);

        OutputPlanner::<R>::new(&self.inputs, &self.outputs, &self.reshapes)
            .run::<BT>(client, device, context, &mut plan);

        VectorizationPlanner::<R>::new(&self.reshapes, &self.reads, &self.settings)
            .run::<Runner>(context, &mut plan);

        match LaunchPlanExecutor::<R>::new(&self.scalars, &self.reshapes, &self.ops)
            .execute::<_, BT>(client, runner, context, plan)
        {
            Err(err) => {
                self.rollback(context, err.handles_input, err.handles_output);
                Err(err.runner_error)
            }
            Ok(val) => Ok(val),
        }
    }

    fn rollback<R: JitRuntime>(
        &self,
        context: &mut Context<'_, JitFusionHandle<R>>,
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
    tensors: BTreeMap<ElemwisePrecision, Vec<TensorDescription>>,
}

impl RegisteredTensors {
    pub fn iter(&self) -> impl Iterator<Item = (ElemwisePrecision, &TensorDescription)> {
        self.tensors.iter().flat_map(|(precision, descriptions)| {
            descriptions.iter().map(|desc| (*precision, desc))
        })
    }

    pub fn len(&self) -> usize {
        self.tensors.values().map(|v| v.len()).sum()
    }

    pub fn get_index(&self, precision: ElemwisePrecision, tensor_id: TensorId) -> Option<usize> {
        self.tensors.get(&precision).and_then(|items| {
            items
                .iter()
                .enumerate()
                .find(|(_pos, tensor)| tensor.id == tensor_id)
                .map(|(pos, _)| pos)
        })
    }

    pub fn get_all(&self, precision: ElemwisePrecision) -> &[TensorDescription] {
        self.tensors
            .get(&precision)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn get(
        &self,
        precision: ElemwisePrecision,
        tensor_id: TensorId,
    ) -> Option<&TensorDescription> {
        self.get_all(precision)
            .iter()
            .find(|desc| desc.id == tensor_id)
    }

    pub fn insert(&mut self, precision: ElemwisePrecision, tensor: TensorDescription) -> u32 {
        if let Some(tensors) = self.tensors.get_mut(&precision) {
            let position = tensors.len() as u32;
            tensors.push(tensor);
            position
        } else {
            self.tensors.insert(precision, vec![tensor]);
            0
        }
    }

    pub fn update(&mut self, precision: ElemwisePrecision, tensor: &TensorDescription) {
        if let Some(tensors) = self.tensors.get_mut(&precision) {
            if let Some(tensor_old) = tensors
                .iter_mut()
                .find(|tensor_old| tensor_old.id == tensor.id)
            {
                tensor_old.status = tensor.status.clone();
            }
        }
    }
}
