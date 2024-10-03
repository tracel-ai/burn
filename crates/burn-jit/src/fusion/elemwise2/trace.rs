use crate::{
    fusion::{strides_dyn_rank, JitFusionHandle},
    JitRuntime,
};

use super::{
    builder::InputHandles,
    ir::{
        Arg, BinaryElemwiseOp, ElemwiseOp, FusionArgsLaunch, FusionConfig, OpPrecision,
        UnaryElemwiseOp,
    },
};
use burn_fusion::stream::Context;
use burn_tensor::{
    repr::{TensorDescription, TensorId, TensorStatus},
    DType, Element,
};
use cubecl::{ir::Elem, prelude::*};
use std::collections::BTreeMap;

#[derive(Clone)]
pub struct Tracel2Builder {
    pub locals: Tensor2Index,
    pub outputs: Index2Tensor,
    pub inputs: Index2Tensor,
    pub scalars: BTreeMap<OpPrecision, u32>,
    pub ops: Sequence<ElemwiseOp>,
}

#[derive(Clone)]
pub struct Trace2 {
    pub outputs: Index2Tensor,
    pub inputs: Index2Tensor,
    pub scalars: BTreeMap<OpPrecision, u32>,
    pub ops: Sequence<ElemwiseOp>,
}

#[derive(Default, Clone)]
pub struct Tensor2Index {
    tensors: BTreeMap<OpPrecision, BTreeMap<TensorId, u32>>,
}

#[derive(Default, Clone)]
pub struct Index2Tensor {
    tensors: BTreeMap<OpPrecision, Vec<TensorDescription>>,
}

impl Index2Tensor {
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
    pub fn insert(&mut self, precision: OpPrecision, tensor: TensorDescription) {
        if let Some(tensors) = self.tensors.get_mut(&precision) {
            tensors.push(tensor);
        } else {
            self.tensors.insert(precision, vec![tensor]);
        }
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

impl Tracel2Builder {
    pub fn new() -> Self {
        Self {
            locals: Tensor2Index::default(),
            outputs: Index2Tensor::default(),
            inputs: Index2Tensor::default(),
            scalars: BTreeMap::default(),
            ops: Sequence::new(),
        }
    }

    pub fn register_operation(&mut self, op: ElemwiseOp) {
        self.ops.push(op);
    }

    pub fn input(&mut self, tensor: &TensorDescription) -> Arg {
        let precision = tensor.dtype.into();

        match self.locals.get(precision, tensor.id) {
            Some(val) => {
                // Update since the status can change for inplace.
                Arg::Local(val, precision)
            }
            None => {
                let new_input = self
                    .inputs
                    .tensors
                    .get(&precision)
                    .map(|val| val.len() as u32)
                    .unwrap_or(0);

                let new_local = self.locals.new_index(precision, tensor.id);
                let input = Arg::Input(new_input, precision);
                let out = Arg::Local(new_local, precision);

                self.ops
                    .push(ElemwiseOp::Assign(UnaryElemwiseOp { input, out }));

                self.inputs.insert(precision, tensor.clone());

                out
            }
        }
    }

    pub fn output(&mut self, tensor: &TensorDescription) -> Arg {
        let precision = tensor.dtype.into();

        match self.locals.get(precision, tensor.id) {
            Some(val) => Arg::Local(val, precision),
            None => {
                let new_local = self.locals.new_index(precision, tensor.id);
                let out = Arg::Local(new_local, precision);

                self.outputs.insert(precision, tensor.clone());

                out
            }
        }
    }

    pub fn scalar<E: Element>(&mut self, _: &E, dtype: DType) -> Arg {
        let precision = dtype.into();
        let new_index = self.scalars.get(&precision).map(|a| *a).unwrap_or(0);
        self.scalars.insert(precision, new_index);
        Arg::Scalar(new_index, precision)
    }

    pub fn build(&self) -> Trace2 {
        let outputs = self.output_tensors();
        let mut ops = self.ops.clone();

        for (precision, tensor) in outputs.iter() {
            let local_index = self.locals.get(precision, tensor.id).unwrap();
            let out_index = outputs.get_index(precision, tensor.id).unwrap();

            ops.push(ElemwiseOp::Assign(UnaryElemwiseOp {
                input: Arg::Local(local_index, precision),
                out: Arg::Output(out_index as u32, precision),
            }))
        }

        Trace2 {
            outputs,
            inputs: self.inputs.clone(),
            scalars: self.scalars.clone(),
            ops,
        }
    }

    fn output_tensors(&self) -> Index2Tensor {
        let mut result = Index2Tensor::default();

        let mut local_tensor_ids_input = Vec::new();
        let mut local_tensor_ids_output = Vec::new();

        // Mark a variable to the provided list of tensor ids using the variable list.
        //
        // Only local variables can become outputs.
        let mark = |var: &Arg, list: &mut Vec<(TensorId, OpPrecision)>| {
            if let Arg::Local(index, precision) = var {
                if let Some(tensor_id) = self.locals.find(*precision, *index) {
                    let entry = (tensor_id, *precision);
                    if !list.contains(&entry) {
                        list.push(entry);
                    }
                }
            }
        };

        let mark_binary = |op: &BinaryElemwiseOp,
                           inputs: &mut Vec<(TensorId, OpPrecision)>,
                           outputs: &mut Vec<(TensorId, OpPrecision)>| {
            mark(&op.lhs, inputs);
            mark(&op.rhs, inputs);
            mark(&op.out, outputs);
        };
        let mark_unary = |op: &UnaryElemwiseOp,
                          inputs: &mut Vec<(TensorId, OpPrecision)>,
                          outputs: &mut Vec<(TensorId, OpPrecision)>| {
            mark(&op.input, inputs);
            mark(&op.out, outputs);
        };

        // For all operators, mark their local tensor id in the proper set.
        for index in 0..self.ops.len() {
            let op = self.ops.index(index);

            match op {
                ElemwiseOp::Add(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Sub(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Mul(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Div(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Powf(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Abs(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Exp(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Log(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Log1p(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Cos(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Sin(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Tanh(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Erf(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Recip(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Assign(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::ToLayout(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::ConditionalAssign {
                    cond,
                    lhs,
                    rhs,
                    out,
                } => {
                    mark(&cond, &mut local_tensor_ids_input);
                    mark(&lhs, &mut local_tensor_ids_input);
                    mark(&rhs, &mut local_tensor_ids_input);
                    mark(&out, &mut local_tensor_ids_output);
                }
                ElemwiseOp::Equal(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Lower(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::Greater(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::LowerEqual(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                ElemwiseOp::GreaterEqual(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
            }
        }

        // All output tensors that are never read by a following operation should be written to
        // since they are essentially the "logical" output of the shader.
        for entry in local_tensor_ids_output {
            let is_read = local_tensor_ids_input.contains(&entry);

            if !is_read {
                let (tensor_id, precision) = entry;
                let tensor = self.outputs.get(precision, tensor_id).unwrap();
                result.insert(precision, tensor.clone());
            }
        }

        // All tensors where their latest description is read only should be written to since they
        // are going to be used after the fused kernel by other operations.
        for (precision, tensor) in self.outputs.iter() {
            if let TensorStatus::ReadOnly = tensor.status {
                if self.locals.get(precision, tensor.id).is_some() {
                    result.insert(precision, tensor.clone());
                }
            }
        }

        result
    }
}

pub trait Launch<R: JitRuntime> {
    fn run<'a>(
        client: &ComputeClient<R::Server, R::Channel>,
        inputs: FusionArgsLaunch<'a, R>,
        outputs: FusionArgsLaunch<'a, R>,
        config: FusionConfig,
    );
}

impl Trace2 {
    pub fn run<'a, R: JitRuntime, L: Launch<R>>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
        vectorization: u8,
        context: &mut Context<'a, JitFusionHandle<R>>,
    ) {
        let mut handles_inputs = Vec::new();
        let mut handles_outputs = Vec::new();
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
        );

        for (precision, tensor) in self.inputs.iter() {
            let tensor = context.tensors.get(&tensor.id).unwrap();
            let handle = context.handles.get_handle(&tensor.id, &tensor.status);

            rank = usize::max(tensor.shape.len(), rank);
            handles_inputs.push((precision, handle, tensor.shape.clone()));
        }

        for (precision, tensor) in self.outputs.iter() {
            let tensor = context.tensors.get(&tensor.id).unwrap();
            let size = tensor.shape.iter().product::<usize>() * Elem::from(tensor.dtype).size();
            let handle = JitFusionHandle {
                client: client.clone(),
                handle: client.empty(size),
                device: device.clone(),
                strides: strides_dyn_rank(&tensor.shape),
            };

            rank = usize::max(tensor.shape.len(), rank);
            context.handles.register_handle(tensor.id, handle.clone());
            handles_outputs.push((precision, handle, tensor.shape.clone()));
        }

        for (precision, count) in self.scalars.iter() {
            for i in 0..(*count as usize + 1) {
                match precision {
                    OpPrecision::F32 => inputs.s_f32.push(ScalarArg::new(context.scalar_f32[i])),
                    OpPrecision::F16 => inputs.s_f16.push(ScalarArg::new(context.scalar_f16[i])),
                    OpPrecision::I32 => inputs.s_i32.push(ScalarArg::new(context.scalar_ints[i])),
                    OpPrecision::U32 => todo!(),
                    _ => todo!(),
                }
            }
        }

        let config = FusionConfig {
            rank: rank as u32,
            ref_layout: super::ir::RefLayout {
                arg: Arg::Output(0, OpPrecision::F32),
            },
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
        for (precision, handle, shape) in handles_outputs.iter() {
            let arg = handle.as_tensor_arg(shape, vectorization);

            match precision {
                OpPrecision::F32 => outputs.t_f32.push(arg),
                OpPrecision::F16 => outputs.t_f16.push(arg),
                OpPrecision::I32 => outputs.t_i32.push(arg),
                OpPrecision::U32 => outputs.t_u32.push(arg),
                _ => todo!(),
            };
        }

        L::run(client, inputs, outputs, config)
    }
}

fn can_vectorize<R: JitRuntime>(
    handles_inputs: &[JitFusionHandle<R>],
    inputs: &[&TensorDescription],
    outputs: &[&TensorDescription],
    factor: usize,
) -> bool {
    let is_unavailable_input = |handle: &JitFusionHandle<R>, desc: &TensorDescription| {
        let rank = handle.strides.len();

        // Last dimension strides should be 1, otherwise vecX won't be contiguous.
        if handle.strides[rank - 1] != 1 {
            return true;
        }

        // The last dimension should be a multiple of the vector size.
        desc.shape[rank - 1] % factor != 0
    };
    let is_unavailable_output = |desc: &TensorDescription| {
        let rank = desc.shape.len();

        // The last dimension should be a multiple of the vector size.
        desc.shape[rank - 1] % factor != 0
    };

    for (handle, tensor) in handles_inputs.iter().zip(inputs.iter()) {
        if is_unavailable_input(handle, tensor) {
            return false;
        }
    }

    // Only need to check when there is no input.
    if handles_inputs.is_empty() {
        for tensor in outputs.iter() {
            if is_unavailable_output(tensor) {
                return false;
            }
        }
    }

    true
}
