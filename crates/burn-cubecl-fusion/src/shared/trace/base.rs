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
use std::{collections::BTreeMap, marker::PhantomData};

#[cfg(feature = "autotune-checks")]
use burn_tensor::TensorData;
#[cfg(feature = "autotune-checks")]
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize, Debug)]
/// A trace contains all [blocks](FuseBlock) and the [resources](KernelResources) used by the
/// kernel.
pub struct FuseTrace {
    pub blocks: Vec<FuseBlock>,
    pub resources: FuseResources,
}

pub enum TuneOutput<R: Runtime> {
    UnChecked(PhantomData<R>),
    #[cfg(feature = "autotune-checks")]
    Checked {
        handles: HashMap<TensorId, (Vec<usize>, CubeFusionHandle<R>)>,
    },
}

impl<R: Runtime> TuneOutput<R> {
    #[allow(unused_variables)]
    pub fn merge(self, other: Self) -> Self {
        let mut result = self;

        match &mut result {
            TuneOutput::UnChecked(..) => {}
            #[cfg(feature = "autotune-checks")]
            TuneOutput::Checked { handles } => match other {
                TuneOutput::UnChecked(..) => {}
                TuneOutput::Checked { handles: o } => {
                    for (k, v) in o.into_iter() {
                        handles.insert(k, v);
                    }
                }
            },
        }

        result
    }
}

impl<R: Runtime> cubecl::tune::AutotuneOutput for TuneOutput<R> {
    #[cfg(feature = "autotune-checks")]
    fn check_equivalence(&self, other: Self) {
        if let (
            TuneOutput::Checked {
                handles: handles_ref,
            },
            TuneOutput::Checked { handles },
        ) = (self, &other)
        {
            let mut num_checked = 0;
            for (id, (shape, handle)) in handles_ref.iter() {
                if let Some((shape_other, other)) = handles.get(id) {
                    assert_eq!(
                        handle.strides, other.strides,
                        "TODO: It should be OK, we simply need to call `into_contiguous` before the assertion."
                    );

                    let data_ref = handle.client.read_one(handle.handle.clone().binding());
                    let data_other = other.client.read_one(other.handle.clone().binding());
                    let data_ref = TensorData::from_bytes(data_ref, shape.clone(), handle.dtype);
                    let data_other =
                        TensorData::from_bytes(data_other, shape_other.clone(), handle.dtype);

                    data_ref.assert_approx_eq(&data_other, 2);
                    num_checked += 1;
                } else {
                    // Debug info for the tests.
                    println!("No tensor found for {id:?}=>{shape:?}");
                }
            }
            // At least one check is needed per output.
            //
            // Some optimizations might write more outputs than needed, so it might be fined if
            // the number of handles is different, but at least one is required.
            assert!(num_checked >= 1);
        }
    }
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
    ) -> Result<TuneOutput<R>, TraceError<Runner::Error>> {
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
