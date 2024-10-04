use super::{
    ir::{Arg, BinaryElemwiseOp, ElemwiseOp, OpPrecision, UnaryElemwiseOp},
    trace::{FuseOnWriteTrace, RegisteredTensors, Tensor2Index},
};
use burn_tensor::{
    repr::{TensorDescription, TensorId, TensorStatus},
    DType, Element,
};
use cubecl::prelude::Sequence;
use std::collections::BTreeMap;

#[derive(Clone)]
pub struct FuseOnWriteTraceBuilder {
    pub locals: Tensor2Index,
    pub outputs: RegisteredTensors,
    pub inputs: RegisteredTensors,
    pub scalars: BTreeMap<OpPrecision, u32>,
    pub ops: Sequence<ElemwiseOp>,
}

impl FuseOnWriteTraceBuilder {
    pub fn new() -> Self {
        Self {
            locals: Tensor2Index::default(),
            outputs: RegisteredTensors::default(),
            inputs: RegisteredTensors::default(),
            scalars: BTreeMap::default(),
            ops: Sequence::new(),
        }
    }

    pub fn register_operation(&mut self, op: ElemwiseOp) {
        // println!("Op EW {op:?}\n");
        self.ops.push(op);
    }

    pub fn input(&mut self, tensor: &TensorDescription) -> Arg {
        // println!("EW INPUT {tensor:?}\n");
        let precision = tensor.dtype.into();

        // Bool tensors are encoded as u32.
        let precision_input = match precision {
            OpPrecision::Bool => OpPrecision::U32,
            _ => precision,
        };

        match self.locals.get(precision, tensor.id) {
            Some(val) => {
                self.inputs.update(precision_input, &tensor);
                // An input can be an output of a previously fused operation.
                // We need to flag the new status for the tensor.
                self.outputs.update(precision_input, &tensor);

                Arg::Local(val, precision)
            }
            None => {
                let new_input = self.inputs.insert(precision_input, tensor.clone());
                let new_local = self.locals.new_index(precision, tensor.id);

                let input = Arg::Input(new_input, precision_input);
                let out = Arg::Local(new_local, precision);

                self.ops
                    .push(ElemwiseOp::Assign(UnaryElemwiseOp { input, out }));

                out
            }
        }
    }

    pub fn output(&mut self, tensor: &TensorDescription) -> Arg {
        let precision = tensor.dtype.into();

        // Bool tensors are encoded as u32.
        let precision_output = match precision {
            OpPrecision::Bool => OpPrecision::U32,
            _ => precision,
        };

        match self.locals.get(precision, tensor.id) {
            Some(val) => Arg::Local(val, precision),
            None => {
                let new_local = self.locals.new_index(precision, tensor.id);
                let out = Arg::Local(new_local, precision);

                self.outputs.insert(precision_output, tensor.clone());

                out
            }
        }
    }

    pub fn clear(&mut self) {
        *self = Self::new();
    }

    pub fn scalar<E: Element>(&mut self, _: &E, dtype: DType) -> Arg {
        let precision = dtype.into();

        // Bool scalars are encoded as u32.
        let precision = match precision {
            OpPrecision::Bool => OpPrecision::U32,
            _ => precision,
        };
        let new_index = self
            .scalars
            .get(&precision)
            .map(|num_scalars| *num_scalars)
            .unwrap_or(0);

        let num_scalars = new_index + 1;

        self.scalars.insert(precision, num_scalars);
        Arg::Scalar(new_index, precision)
    }

    pub fn build(&self) -> FuseOnWriteTrace {
        let outputs = self.output_tensors();
        let mut ops = self.ops.clone();

        for (precision, tensor) in outputs.iter() {
            let (local_precision, local_index) = self.locals.get_any_precision(tensor.id).unwrap();
            let out_index = outputs.get_index(precision, tensor.id).unwrap();

            ops.push(ElemwiseOp::Assign(UnaryElemwiseOp {
                input: Arg::Local(local_index, local_precision),
                out: Arg::Output(out_index as u32, precision),
            }))
        }

        FuseOnWriteTrace {
            outputs,
            inputs: self.inputs.clone(),
            scalars: self.scalars.clone(),
            ops,
        }
    }

    fn output_tensors(&self) -> RegisteredTensors {
        let mut result = RegisteredTensors::default();

        let mut local_tensor_ids_input = Vec::new();
        let mut local_tensor_ids_output = Vec::new();

        // Mark a variable to the provided list of tensor ids using the variable list.
        //
        // Only local variables can become outputs.
        let mark = |var: &Arg, list: &mut Vec<(TensorId, OpPrecision)>| {
            if let Arg::Local(index, precision) = var {
                if let Some(tensor_id) = self.locals.find(*precision, *index) {
                    // Input and outputs tensors are using u32 for booleans.
                    let precision = match precision {
                        OpPrecision::Bool => OpPrecision::U32,
                        _ => *precision,
                    };

                    let entry = (tensor_id, precision);
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

        println!("Input {local_tensor_ids_input:?}");
        println!("Output {local_tensor_ids_output:?}");
        // All output tensors that are never read by a following operation should be written to
        // since they are essentially the "logical" output of the shader.
        for entry in local_tensor_ids_output {
            let is_read = local_tensor_ids_input.contains(&entry);

            if !is_read {
                let (tensor_id, precision) = entry;
                let tensor = self.outputs.get(precision, tensor_id).unwrap();
                result.insert(precision, tensor.clone());
                println!("Include output 1 {tensor:?}");
            } else {
                println!("Don't include output 1 {entry:?}");
            }
        }

        // All tensors where their latest description is read only should be written to since they
        // are going to be used after the fused kernel by other operations.
        for (_precision, tensor) in self.outputs.iter() {
            if let TensorStatus::ReadOnly = tensor.status {
                let (precision, _) = self.locals.get_any_precision(tensor.id).unwrap();
                println!("Include output 2 {tensor:?}");
                result.insert(precision, tensor.clone());
            } else {
                println!("Don't include output 2 {tensor:?}");
            }
        }

        result
    }
}
