use crate::{
    fusion::{strides_dyn_rank, JitFusionHandle},
    JitRuntime,
};

use super::ir::{Arg, ElemwiseOp, FusionArgsLaunch, FusionConfig, OpPrecision, RefLayout};
use burn_fusion::stream::Context;
use burn_tensor::{
    repr::{TensorDescription, TensorId, TensorStatus},
    DType,
};
use cubecl::{ir::Elem, prelude::*};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Serialize, Deserialize)]
pub struct FuseOnWriteTrace {
    pub outputs: RegisteredTensors,
    pub inputs: RegisteredTensors,
    pub scalars: BTreeMap<OpPrecision, u32>,
    pub ops: Sequence<ElemwiseOp>,
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Tensor2Index {
    tensors: BTreeMap<OpPrecision, BTreeMap<TensorId, u32>>,
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct RegisteredTensors {
    tensors: BTreeMap<OpPrecision, Vec<TensorDescription>>,
}

impl RegisteredTensors {
    pub fn iter(&self) -> impl Iterator<Item = (OpPrecision, &TensorDescription)> {
        self.tensors
            .iter()
            .map(|(precision, descriptions)| descriptions.iter().map(|desc| (*precision, desc)))
            .flatten()
    }

    pub fn get_index(&self, precision: OpPrecision, tensor_id: TensorId) -> Option<usize> {
        self.tensors
            .get(&precision)
            .map(|items| {
                items
                    .iter()
                    .enumerate()
                    .find(|(_pos, tensor)| tensor.id == tensor_id)
                    .map(|(pos, _)| pos)
            })
            .flatten()
    }

    pub fn get_all(&self, precision: OpPrecision) -> &[TensorDescription] {
        self.tensors
            .get(&precision)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
    pub fn get(&self, precision: OpPrecision, tensor_id: TensorId) -> Option<&TensorDescription> {
        self.get_all(precision)
            .iter()
            .find(|desc| desc.id == tensor_id)
    }

    pub fn insert(&mut self, precision: OpPrecision, tensor: TensorDescription) -> u32 {
        if let Some(tensors) = self.tensors.get_mut(&precision) {
            let position = tensors.len() as u32;
            tensors.push(tensor);
            position
        } else {
            self.tensors.insert(precision, vec![tensor]);
            0
        }
    }
    pub fn update(&mut self, precision: OpPrecision, tensor: &TensorDescription) -> bool {
        if let Some(tensors) = self.tensors.get_mut(&precision) {
            if let Some(tensor_old) = tensors
                .iter_mut()
                .find(|tensor_old| tensor_old.id == tensor.id)
            {
                tensor_old.status = tensor.status.clone();
                return true;
            }
        }

        false
    }
}

impl Tensor2Index {
    pub fn get(&self, precision: OpPrecision, tensor_id: TensorId) -> Option<u32> {
        if let Some(indexes) = self.tensors.get(&precision) {
            if let Some(index) = indexes.get(&tensor_id) {
                return Some(*index);
            }
        }

        None
    }

    pub fn get_any_precision(&self, tensor_id: TensorId) -> Option<(OpPrecision, u32)> {
        for (precision, indexes) in self.tensors.iter() {
            if let Some(index) = indexes.get(&tensor_id) {
                return Some((*precision, *index));
            }
        }

        None
    }

    pub fn find(&self, precision: OpPrecision, position: u32) -> Option<TensorId> {
        if let Some(indexes) = self.tensors.get(&precision) {
            indexes
                .iter()
                .find(|(_id, index)| **index == position)
                .map(|(id, _index)| *id)
        } else {
            None
        }
    }

    pub fn new_index(&mut self, precision: OpPrecision, tensor_id: TensorId) -> u32 {
        if let Some(indexes) = self.tensors.get_mut(&precision) {
            let new_index = indexes.len() as u32;
            indexes.insert(tensor_id, new_index);
            return new_index;
        }

        let new_index = 0;
        self.tensors
            .insert(precision, BTreeMap::from_iter([(tensor_id, new_index)]));
        new_index
    }
}

pub trait RunTrace<R: JitRuntime> {
    fn run<'a>(
        client: &ComputeClient<R::Server, R::Channel>,
        inputs: FusionArgsLaunch<'a, R>,
        outputs: FusionArgsLaunch<'a, R>,
        config: FusionConfig,
    );
    /// The vectorization factor for all inputs and outputs.
    fn vectorization<'a>(
        handles_inputs: impl Iterator<Item = &'a JitFusionHandle<R>>,
        inputs: impl Iterator<Item = &'a TensorDescription>,
        outputs: impl Iterator<Item = &'a TensorDescription>,
    ) -> u8;
}

impl FuseOnWriteTrace {
    pub fn run<'a, R: JitRuntime, L: RunTrace<R>>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        context: &mut Context<'a, JitFusionHandle<R>>,
    ) {
        #[derive(Debug)]
        enum HandleOutput<R: JitRuntime> {
            Alias(usize, OpPrecision),
            Owned(OpPrecision, JitFusionHandle<R>, Vec<usize>),
        }
        let mut handles_inputs = Vec::new();
        let mut handles_outputs = Vec::new();

        let mut inputs_desc = Vec::new();
        let mut outputs_desc = Vec::new();
        let mut potential_inplaces = Vec::new();

        let mut ref_layout = None;
        let mut rank = 1;

        let mut inputs = FusionArgsLaunch::new(
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
        );
        let mut outputs = FusionArgsLaunch::new(
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
            SequenceArg::new(),
        );

        for (i, (precision, tensor_relative)) in self.inputs.iter().enumerate() {
            let tensor_global = context.tensors.get(&tensor_relative.id).unwrap();
            // Important to take the status of the relative graph and not
            // the global graph, since the status of the global graph
            // might be of a later operation on the same tensor id.
            let status = &tensor_relative.status;
            let handle = context.handles.get_handle(&tensor_global.id, status);

            if status == &TensorStatus::ReadWrite && handle.handle.can_mut() && false {
                potential_inplaces.push((tensor_relative, handle.strides.clone(), i));
            }

            inputs_desc.push(tensor_global);
            rank = usize::max(tensor_global.shape.len(), rank);
            handles_inputs.push((precision, handle, tensor_global.shape.clone()));
        }

        for (precision, tensor_relative) in self.outputs.iter() {
            let tensor_global = context.tensors.get(&tensor_relative.id).unwrap();
            let strides = strides_dyn_rank(&tensor_global.shape);
            outputs_desc.push(tensor_global);

            if let Some(index) = potential_inplaces
                .iter()
                .enumerate()
                .find(|(_pos, (input_relative, input_strides, _))| {
                    input_relative.dtype == tensor_global.dtype
                        && input_relative.shape == tensor_relative.shape
                        && input_strides == &strides
                })
                .map(|(pos, _)| pos)
            {
                let (tensor_relative_input, _strides, handle_index) =
                    potential_inplaces.remove(index);
                let (_, handle, _) = handles_inputs.get(handle_index).unwrap();

                if ref_layout.is_none() {
                    let index_input = self
                        .inputs
                        .get_index(precision, tensor_relative_input.id)
                        .unwrap();
                    ref_layout = Some(RefLayout {
                        arg: Arg::Input(index_input as u32, precision),
                    });
                }

                context
                    .handles
                    .register_handle(tensor_global.id, handle.clone());
                handles_outputs.push(HandleOutput::Alias(handle_index, precision));
            } else {
                if ref_layout.is_none() {
                    ref_layout = Some(RefLayout {
                        arg: Arg::Output(0, precision),
                    });
                }

                // We encode bool tensors as u32.
                let dtype = match tensor_global.dtype {
                    DType::Bool => DType::U32,
                    _ => tensor_global.dtype,
                };
                let size = tensor_global.shape.iter().product::<usize>() * Elem::from(dtype).size();
                let handle = JitFusionHandle {
                    client: client.clone(),
                    handle: client.empty(size),
                    device: device.clone(),
                    strides,
                };

                rank = usize::max(tensor_global.shape.len(), rank);
                context
                    .handles
                    .register_handle(tensor_global.id, handle.clone());
                handles_outputs.push(HandleOutput::Owned(
                    precision,
                    handle,
                    tensor_global.shape.clone(),
                ));
            }
        }

        for (precision, count) in self.scalars.iter() {
            for i in 0..(*count as usize) {
                match precision {
                    OpPrecision::F32 => inputs.s_f32.push(ScalarArg::new(context.scalar_f32[i])),
                    OpPrecision::F16 => inputs.s_f16.push(ScalarArg::new(context.scalar_f16[i])),
                    OpPrecision::I32 => inputs.s_i32.push(ScalarArg::new(context.scalar_ints[i])),
                    OpPrecision::U32 => todo!(),
                    _ => todo!(),
                }
            }
        }

        let vectorization = L::vectorization(
            handles_inputs.iter().map(|(_, handle, _)| handle),
            inputs_desc.iter().map(|desc| *desc),
            outputs_desc.iter().map(|desc| *desc),
        );

        let config = FusionConfig {
            rank: rank as u32,
            ref_layout: ref_layout.expect("An output should exist for the fused kernel"),
            ops: self.ops.clone(),
        };

        // Register everything
        for (precision, handle, shape) in handles_inputs.iter() {
            let arg = handle.as_tensor_arg(shape, vectorization);

            match precision {
                OpPrecision::F32 => inputs.t_f32.push(arg),
                OpPrecision::F16 => inputs.t_f16.push(arg),
                OpPrecision::I32 => inputs.t_i32.push(arg),
                OpPrecision::U32 => inputs.t_u32.push(arg),
                _ => todo!(),
            };
        }
        for item in handles_outputs.iter() {
            match item {
                HandleOutput::Alias(index, precision) => match precision {
                    OpPrecision::F32 => outputs.t_f32.push(TensorArg::alias(*index)),
                    OpPrecision::F16 => outputs.t_f16.push(TensorArg::alias(*index)),
                    OpPrecision::I32 => outputs.t_i32.push(TensorArg::alias(*index)),
                    OpPrecision::U32 => outputs.t_u32.push(TensorArg::alias(*index)),
                    _ => todo!(),
                },
                HandleOutput::Owned(precision, handle, shape) => {
                    let arg = handle.as_tensor_arg(shape, vectorization);

                    match precision {
                        OpPrecision::F32 => outputs.t_f32.push(arg),
                        OpPrecision::F16 => outputs.t_f16.push(arg),
                        OpPrecision::I32 => outputs.t_i32.push(arg),
                        OpPrecision::U32 => outputs.t_u32.push(arg),
                        // Bools are encoded as u32.
                        OpPrecision::Bool => outputs.t_u32.push(arg),
                        _ => todo!(),
                    };
                }
            }
        }

        L::run(client, inputs, outputs, config)
    }
}
