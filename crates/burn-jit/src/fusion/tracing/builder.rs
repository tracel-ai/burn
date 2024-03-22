use super::{trace::Trace, Scalars};
use crate::codegen::dialect::gpu::{self, Operation, Variable};
use burn_fusion::{TensorDescription, TensorId};
use burn_tensor::Element;
use hashbrown::HashMap;

/// Type facilitating building a [trace](Trace) by doing most of the conversions between the
/// operations provided in [burn_fusion] and the [gpu dialect](gpu).
#[derive(Clone)]
pub struct TraceBuilder {
    // Input tensor descriptions with the variables created after reading from global memory.
    inputs: Vec<(TensorDescription, Variable)>,
    // Each output tensor id with the output variable index created by the operation.
    output_to_local: HashMap<TensorId, u16>,
    tensors: HashMap<TensorId, (TensorDescription, gpu::Elem)>,
    scalars: Scalars,
    scope: gpu::Scope,
}

impl TraceBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            output_to_local: HashMap::new(),
            tensors: HashMap::new(),
            scalars: Scalars::default(),
            scope: gpu::Scope::root(),
        }
    }

    /// Register a [gpu operation](gpu::Operation).
    pub fn register_operation<T: Into<gpu::Operation>>(&mut self, value: T) {
        self.scope.register(value)
    }

    /// Create a variable from an input [tensor description](TensorDescription).
    pub fn input(&mut self, tensor: &TensorDescription, elem: gpu::Elem) -> gpu::Variable {
        let already_exists = self.tensors.contains_key(&tensor.id);

        let variable = match already_exists {
            false => {
                // New input
                let index = self.inputs.len() as u16;
                let item = gpu::Item::Scalar(elem);

                let local = self.scope.read_array(index, item);
                self.inputs.push((tensor.clone(), local));
                local
            }
            true => match self.output_to_local.get(&tensor.id) {
                // Is a local variable.
                Some(local_index) => {
                    gpu::Variable::Local(*local_index, gpu::Item::Scalar(elem), self.scope.depth)
                }
                // Isn't an operation output variable, so must be an existing input.
                None => self
                    .inputs
                    .iter()
                    .find(|(input, _local)| input.id == tensor.id)
                    .map(|(_, local)| *local)
                    .unwrap(),
            },
        };

        // Update the tensor description with the new version.
        self.tensors.insert(tensor.id, (tensor.clone(), elem));

        variable
    }

    /// Create a variable from an output [tensor description](TensorDescription).
    pub fn output(&mut self, tensor: &TensorDescription, elem: gpu::Elem) -> gpu::Variable {
        // Update the tensor description to the new version.
        self.tensors.insert(tensor.id, (tensor.clone(), elem));

        // Output already registered as a local variable.
        if let Some(index) = self.output_to_local.get(&tensor.id) {
            return gpu::Variable::Local(*index, gpu::Item::Scalar(elem), self.scope.depth);
        }

        let variable = self.scope.create_local(gpu::Item::Scalar(elem));
        let local_index = variable.index().unwrap();
        self.output_to_local.insert(tensor.id, local_index);
        variable
    }

    /// Create a variable from an input [scalar](Element).
    pub fn scalar<E: Element>(&mut self, _value: &E, elem_type: gpu::Elem) -> gpu::Variable {
        match elem_type {
            gpu::Elem::Float => {
                let var = self
                    .scope
                    .read_scalar(self.scalars.num_float as u16, elem_type);
                self.scalars.num_float += 1;
                var
            }
            gpu::Elem::Int => {
                let var = self
                    .scope
                    .read_scalar(self.scalars.num_int as u16, elem_type);
                self.scalars.num_int += 1;
                var
            }
            gpu::Elem::UInt => {
                let var = self
                    .scope
                    .read_scalar(self.scalars.num_uint as u16, elem_type);
                self.scalars.num_uint += 1;
                var
            }
            gpu::Elem::Bool => {
                let var = self
                    .scope
                    .read_scalar(self.scalars.num_bool as u16, elem_type);
                self.scalars.num_bool += 1;
                var
            }
        }
    }

    /// Build the [trace](Trace).
    pub fn build(self) -> Trace {
        let inputs = self.input_descriptions();
        let outputs = self.output_descriptions();
        let locals = outputs
            .iter()
            .map(|out| *self.output_to_local.get(&out.0.id).unwrap())
            .collect::<Vec<_>>();

        Trace::new(inputs, outputs, locals, self.scalars, self.scope)
    }

    fn input_descriptions(&self) -> Vec<(TensorDescription, gpu::Elem)> {
        self.inputs
            .iter()
            .map(|(input, _local)| {
                let updated_tensor = self.tensors.get(&input.id).unwrap();
                updated_tensor.clone()
            })
            .collect::<Vec<_>>()
    }

    fn output_descriptions(&self) -> Vec<(TensorDescription, gpu::Elem)> {
        let mut outputs = Vec::new();
        let mut local_tensor_ids_input = Vec::new();
        let mut local_tensor_ids_output = Vec::new();

        // Mark a variable to the provided list of tensor ids using the variable list.
        //
        // Only local variables can become outputs.
        let mark = |var: &gpu::Variable, list: &mut Vec<TensorId>| {
            if let gpu::Variable::Local(index, _, _) = var {
                if let Some((id, _)) = self
                    .output_to_local
                    .iter()
                    .find(|(_id, position)| *position == index)
                {
                    if !list.contains(id) {
                        list.push(*id);
                    }
                }
            }
        };
        let mark_binary =
            |op: &gpu::BinaryOperator, inputs: &mut Vec<TensorId>, outputs: &mut Vec<TensorId>| {
                mark(&op.lhs, inputs);
                mark(&op.rhs, inputs);
                mark(&op.out, outputs);
            };
        let mark_unary =
            |op: &gpu::UnaryOperator, inputs: &mut Vec<TensorId>, outputs: &mut Vec<TensorId>| {
                mark(&op.input, inputs);
                mark(&op.out, outputs);
            };

        // For all operators, mark their local tensor id in the proper set.
        for op in self.scope.operations.iter() {
            match op {
                Operation::Operator(op) => match op {
                    gpu::Operator::Assign(op) => {
                        mark(&op.out, &mut local_tensor_ids_output);
                    }
                    gpu::Operator::Add(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Max(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),

                    gpu::Operator::Min(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::And(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Or(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Not(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Index(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Sub(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Mul(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Div(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Exp(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Abs(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Erf(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Ceil(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Log(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Log1p(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Cos(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Sin(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Tanh(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Clamp(op) => {
                        mark(&op.input, &mut local_tensor_ids_input);
                        mark(&op.out, &mut local_tensor_ids_output);
                    }
                    gpu::Operator::Powf(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Recip(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Lower(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Greater(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::LowerEqual(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::GreaterEqual(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Equal(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::NotEqual(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Sqrt(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::Modulo(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::IndexAssign(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::BitwiseAnd(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::BitwiseXor(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::ShiftLeft(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    gpu::Operator::ShiftRight(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                },
                Operation::Procedure(proc) => {
                    match proc {
                        gpu::Procedure::ReadGlobalWithLayout(_) => {
                            // Nothing to do here.
                        }
                        gpu::Procedure::ReadGlobal(_) => {
                            // Nothing to do here.
                        }
                        gpu::Procedure::WriteGlobal(_) => {
                            // Nothing to do here.
                        }
                        gpu::Procedure::ConditionalAssign(proc) => {
                            mark(&proc.cond, &mut local_tensor_ids_input);
                            mark(&proc.lhs, &mut local_tensor_ids_input);
                            mark(&proc.rhs, &mut local_tensor_ids_input);
                            mark(&proc.out, &mut local_tensor_ids_output);
                        }
                        gpu::Procedure::IndexOffsetGlobalWithLayout(_) => {
                            // Nothing to do here.
                        }
                    }
                }
                Operation::Metadata(_) => {
                    // Nothing to do, should never impact read-write access to bindings.
                }
                Operation::Branch(_) => {
                    // Nothing to do, should never impact read-write access to bindings.
                }
                Operation::Synchronization(_) => {
                    // Nothing to do, should never impact read-write access to bindings.
                }
            }
        }

        // All output tensors that are never read by a following operation should be written to
        // since they are essentially the "logical" output of the shader.
        for out in local_tensor_ids_output {
            let is_read = local_tensor_ids_input.contains(&out);

            if !is_read {
                outputs.push(self.tensors.get(&out).unwrap().clone());
            }
        }

        // All tensors where their latest description is read only should be written to since they
        // are going to be used after the fused kernel by other operations.
        for entry in self.tensors.values() {
            let (tensor, _) = &entry;
            if let burn_fusion::TensorStatus::ReadOnly = tensor.status {
                if self.output_to_local.contains_key(&tensor.id) {
                    outputs.push(entry.clone());
                }
            }
        }

        outputs
    }
}
