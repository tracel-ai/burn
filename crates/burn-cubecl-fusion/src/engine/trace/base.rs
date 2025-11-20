use crate::{
    CubeFusionHandle,
    engine::{
        codegen::ir::{FuseArg, FuseType},
        launch::{
            HandleInput, HandleOutput, LaunchPlan, executor::LaunchPlanExecutor,
            input::InputPlanner, output::OutputPlanner, runner::TraceRunner,
            vectorization::VectorizationPlanner,
        },
        trace::block::FuseBlock,
    },
};
use burn_fusion::stream::Context;
use burn_ir::{TensorId, TensorIr};
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashSet},
    marker::PhantomData,
};

#[cfg(feature = "autotune-checks")]
use burn_tensor::TensorData;
#[cfg(feature = "autotune-checks")]
use std::collections::HashMap;

#[derive(Clone, Serialize, Deserialize, Debug)]
/// A trace contains all [blocks](FuseBlock) and the [resources](FuseResources) used by the
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
        use burn_tensor::{DType, Tolerance};

        if let (
            TuneOutput::Checked {
                handles: handles_ref,
            },
            TuneOutput::Checked { handles },
        ) = (self, &other)
        {
            let mut num_checked = 0;
            let mut num_handles = 0;
            for (id, (shape, handle)) in handles_ref.iter() {
                num_handles += 1;
                if let Some((shape_other, other)) = handles.get(id) {
                    assert_eq!(
                        handle.strides, other.strides,
                        "TODO: It should be OK, we simply need to call `into_contiguous` before the assertion."
                    );

                    let data_ref = handle.client.read_one(handle.handle.clone());
                    let data_other = other.client.read_one(other.handle.clone());
                    let data_ref = TensorData::from_bytes(data_ref, shape.clone(), handle.dtype);
                    let data_other =
                        TensorData::from_bytes(data_other, shape_other.clone(), handle.dtype);

                    match handle.dtype {
                        DType::F64 => {
                            data_ref.assert_approx_eq::<f64>(&data_other, Tolerance::permissive())
                        }
                        DType::F32 => {
                            data_ref.assert_approx_eq::<f32>(&data_other, Tolerance::permissive())
                        }
                        DType::F16 => data_ref
                            .assert_approx_eq::<half::f16>(&data_other, Tolerance::permissive()),
                        DType::BF16 => data_ref
                            .assert_approx_eq::<half::bf16>(&data_other, Tolerance::permissive()),
                        _ => data_ref.assert_eq(&data_other, true),
                    }
                    num_checked += 1;
                } else {
                    // Debug info for the tests.
                    println!("No tensor found for {id:?}=>{shape:?}");
                }
            }

            // At least one check is needed per output when there is an output.
            //
            // Some optimizations might write more outputs than needed, so it might be fined if
            // the number of handles is different, but at least one is required.
            //
            // An optimization might not create outputs if its dead code detection is triggered,
            // therefore avoiding useless computation.
            if num_handles > 0 {
                assert!(num_checked >= 1);
            }
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
    pub scalars: Vec<(FuseType, u64)>,
    pub views: Vec<TensorView>,
    pub indexed: BTreeMap<TensorId, FuseArg>,
    pub inputs_unhandled: Vec<TensorId>,
    pub outputs_unhandled: Vec<FuseArg>,
    pub num_reshaped: usize,
    pub dropped: HashSet<TensorId>,
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
        client: &ComputeClient<R::Server>,
        device: &R::Device,
        context: &mut Context<'_, CubeFusionHandle<R>>,
        runner: &Runner,
    ) -> Result<TuneOutput<R>, TraceError<Runner::Error>> {
        let mut plan = LaunchPlan::<R>::new(&self.blocks);

        InputPlanner::<R>::new(&self.resources, &self.blocks).run(context, &mut plan);

        OutputPlanner::<R>::new(&self.resources, &self.blocks)
            .run::<BT>(client, device, context, &mut plan);

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
            match input {
                HandleInput::Normal(input) => {
                    context
                        .handles
                        .register_handle(input.global_ir.id, input.handle_rollback());
                }
                HandleInput::QuantValues(input) => {
                    context
                        .handles
                        .register_handle(input.global_ir.id, input.handle);
                }
                HandleInput::QuantParams(_) => {
                    // The scales are part of the quant data handle.
                }
            };
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
    tensors: Vec<RegisterTensor>,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub enum RegisterTensor {
    Normal(TensorIr, FuseType),
    QuantValues(TensorIr),
    QuantParams(TensorId),
}

impl RegisterTensor {
    pub fn as_normal_tensor(&self) -> Option<(&TensorIr, &FuseType)> {
        match self {
            RegisterTensor::Normal(tensor_ir, precision) => Some((tensor_ir, precision)),
            RegisterTensor::QuantValues(_) => None,
            RegisterTensor::QuantParams(_) => None,
        }
    }
}

impl RegisteredTensors {
    /// Iterate over all the registered tensors.
    pub fn iter(&self) -> impl Iterator<Item = &RegisterTensor> {
        self.tensors.iter()
    }

    /// Consumes and iterate over all the registered tensors.
    pub fn into_iter(self) -> impl Iterator<Item = RegisterTensor> {
        self.tensors.into_iter()
    }

    /// Returns the number of tensors registered.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Retrieve the [tensor id](TensorId) at the given index.
    pub fn get_id(&self, index: usize) -> Option<TensorId> {
        self.tensors.get(index).map(|entry| match entry {
            RegisterTensor::Normal(tensor_ir, _) => tensor_ir.id,
            RegisterTensor::QuantValues(tensor_ir) => tensor_ir.id,
            RegisterTensor::QuantParams(tensor_id) => *tensor_id,
        })
    }

    /// Doesn't return quantized tensor.
    pub fn get_index(&self, tensor_id: TensorId) -> Option<u32> {
        self.tensors
            .iter()
            .enumerate()
            .find(|(_pos, entry)| match entry {
                RegisterTensor::Normal(tensor_ir, _) => tensor_ir.id == tensor_id,
                RegisterTensor::QuantValues(_) => false,
                RegisterTensor::QuantParams(_) => false,
            })
            .map(|(pos, _)| pos as u32)
    }

    /// Get the index of a quantized tensor.
    pub fn get_index_quant(&self, tensor_id: TensorId) -> Option<u32> {
        self.tensors
            .iter()
            .enumerate()
            .find(|(_pos, entry)| match entry {
                RegisterTensor::Normal(..) => false,
                RegisterTensor::QuantValues(tensor_ir) => tensor_ir.id == tensor_id,
                RegisterTensor::QuantParams(_) => false,
            })
            .map(|(pos, _)| pos as u32)
    }

    /// Doesn't return quantized tensor.
    pub fn get(&self, tensor_id: TensorId) -> Option<(&TensorIr, &FuseType)> {
        self.tensors
            .iter()
            .find(|entry| match entry {
                RegisterTensor::Normal(tensor_ir, _) => tensor_ir.id == tensor_id,
                RegisterTensor::QuantValues(_) => false,
                RegisterTensor::QuantParams(_) => false,
            })
            .and_then(|entry| match entry {
                RegisterTensor::Normal(tensor_ir, fuse_precision) => {
                    Some((tensor_ir, fuse_precision))
                }
                RegisterTensor::QuantValues(_) => None,
                RegisterTensor::QuantParams(_) => None,
            })
    }

    /// Insert a quantized tensor.
    ///
    /// It will return the positions for both the value tensor and param tensor.
    pub fn insert_quant(&mut self, tensor: TensorIr) -> (u32, u32) {
        if let Some(old) = self.tensors.iter().enumerate().find(|(_, val)| match &val {
            RegisterTensor::QuantValues(tensor_ir) => tensor_ir == &tensor,
            _ => false,
        }) {
            let values = old.0 as u32;
            let params = values + 1;
            return (values, params);
        }

        let params = RegisterTensor::QuantParams(tensor.id);
        let values = RegisterTensor::QuantValues(tensor);
        let pos_values = self.len();
        self.tensors.push(values);

        let pos_params = self.len();
        self.tensors.push(params);

        (pos_values as u32, pos_params as u32)
    }

    /// Insert a normal tensor with the given [precision](FusePrecision) in the current block.
    pub fn insert(&mut self, precision: FuseType, tensor: TensorIr) -> u32 {
        if let Some(old) = self.tensors.iter().enumerate().find(|(_, val)| match &val {
            RegisterTensor::Normal(tensor_ir, _) => tensor_ir == &tensor,
            _ => false,
        }) {
            return old.0 as u32;
        }

        let value = RegisterTensor::Normal(tensor, precision);
        let pos = self.len();

        self.tensors.push(value);

        pos as u32
    }

    /// Update the already registered tensor with the given [tensor ir](TensorIr).
    ///
    /// # Notes
    ///
    /// This function only works with normal tensors, not quantized tensors.
    pub fn update(&mut self, tensor: &TensorIr) {
        if let Some(entry) = self.tensors.iter_mut().find(|entry| match entry {
            RegisterTensor::Normal(tensor_ir, _) => tensor_ir.id == tensor.id,
            _ => false,
        }) && let RegisterTensor::Normal(tensor_ir, _) = entry
        {
            tensor_ir.status = tensor.status
        }
    }
}
