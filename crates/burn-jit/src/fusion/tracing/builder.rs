use super::{trace::Trace, Scalars};
use burn_tensor::{
    repr::{TensorDescription, TensorId, TensorStatus},
    Element,
};
use cubecl::ir::{
    BinaryOperator, Elem, Item, Operation, Operator, Procedure, Scope, Subcube, UnaryOperator,
    Variable,
};
use hashbrown::HashMap;

/// Type facilitating building a [trace](Trace) by doing most of the conversions between the
/// operations provided in [burn_fusion] and the [gpu ir](gpu).
#[derive(Clone)]
pub struct TraceBuilder {
    // Input tensor descriptions with the variables created after reading from global memory.
    inputs: Vec<(TensorDescription, Variable)>,
    // Each output tensor id with the output variable index created by the operation.
    output_to_local: HashMap<TensorId, u16>,
    tensors: HashMap<TensorId, (TensorDescription, Elem, Variable)>,
    scalars: Scalars,
    scope: Scope,
}

impl TraceBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            output_to_local: HashMap::new(),
            tensors: HashMap::new(),
            scalars: Scalars::default(),
            scope: Scope::root(),
        }
    }

    /// Register a [gpu operation](Operation).
    pub fn register_operation<T: Into<Operation>>(&mut self, value: T) {
        self.scope.register(value)
    }

    /// Create a variable from an input [tensor description](TensorDescription).
    pub fn input(&mut self, tensor: &TensorDescription, position: Variable) -> Variable {
        let already_exists = self.tensors.contains_key(&tensor.id);
        let elem = tensor.dtype.into();

        let variable = match already_exists {
            false => {
                // New input
                let index = self.inputs.len() as u16;
                let item = Item::new(elem);

                let local = self.scope.read_array(index, item, position);
                self.inputs.push((tensor.clone(), local));
                local
            }
            true => match self.output_to_local.get(&tensor.id) {
                // Is a local variable.
                Some(local_index) => Variable::Local {
                    id: *local_index,
                    item: Item::new(elem),
                    depth: self.scope.depth,
                },
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
        self.tensors
            .insert(tensor.id, (tensor.clone(), elem, position));

        variable
    }

    /// Create a variable from an output [tensor description](TensorDescription).
    pub fn output(&mut self, tensor: &TensorDescription, position: Variable) -> Variable {
        let elem = tensor.dtype.into();
        // Update the tensor description to the new version.
        self.tensors
            .insert(tensor.id, (tensor.clone(), elem, position));

        // Output already registered as a local variable.
        if let Some(index) = self.output_to_local.get(&tensor.id) {
            return Variable::Local {
                id: *index,
                item: Item::new(elem),
                depth: self.scope.depth,
            };
        }

        let variable = self.scope.create_local(Item::new(elem));
        let local_index = variable.index().unwrap();
        self.output_to_local.insert(tensor.id, local_index);
        variable
    }

    /// Create a variable from an input [scalar](Element).
    pub fn scalar<E: Element>(&mut self, _value: &E, elem_type: Elem) -> Variable {
        match elem_type {
            Elem::Float(kind) => match kind {
                cubecl::ir::FloatKind::F16 => {
                    let var = self
                        .scope
                        .read_scalar(self.scalars.num_f16 as u16, elem_type);

                    self.scalars.num_f16 += 1;
                    var
                }
                cubecl::ir::FloatKind::F32 => {
                    let var = self
                        .scope
                        .read_scalar(self.scalars.num_f32 as u16, elem_type);

                    self.scalars.num_f32 += 1;
                    var
                }
                cubecl::ir::FloatKind::BF16 => {
                    let var = self
                        .scope
                        .read_scalar(self.scalars.num_bf16 as u16, elem_type);

                    self.scalars.num_bf16 += 1;
                    var
                }
                cubecl::ir::FloatKind::F64 => todo!(),
            },
            Elem::Int(_) => {
                let var = self
                    .scope
                    .read_scalar(self.scalars.num_int as u16, elem_type);
                self.scalars.num_int += 1;
                var
            }
            Elem::UInt => {
                let var = self
                    .scope
                    .read_scalar(self.scalars.num_uint as u16, elem_type);
                self.scalars.num_uint += 1;
                var
            }
            Elem::Bool => {
                let var = self
                    .scope
                    .read_scalar(self.scalars.num_bool as u16, elem_type);
                self.scalars.num_bool += 1;
                var
            }
            Elem::AtomicInt(_kind) => {
                let var = self
                    .scope
                    .read_scalar(self.scalars.num_int as u16, elem_type);
                self.scalars.num_int += 1;
                var
            }
            Elem::AtomicUInt => {
                let var = self
                    .scope
                    .read_scalar(self.scalars.num_uint as u16, elem_type);
                self.scalars.num_int += 1;
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

    fn input_descriptions(&self) -> Vec<(TensorDescription, Elem, Variable)> {
        self.inputs
            .iter()
            .map(|(input, _local)| {
                let updated_tensor = self.tensors.get(&input.id).unwrap();
                updated_tensor.clone()
            })
            .collect::<Vec<_>>()
    }

    fn output_descriptions(&self) -> Vec<(TensorDescription, Elem, Variable)> {
        let mut outputs = Vec::new();
        let mut local_tensor_ids_input = Vec::new();
        let mut local_tensor_ids_output = Vec::new();

        // Mark a variable to the provided list of tensor ids using the variable list.
        //
        // Only local variables can become outputs.
        let mark = |var: &Variable, list: &mut Vec<TensorId>| {
            if let Variable::Local { id: id_local, .. } = var {
                if let Some((id, _)) = self
                    .output_to_local
                    .iter()
                    .find(|(_tensor_id, position)| *position == id_local)
                {
                    if !list.contains(id) {
                        list.push(*id);
                    }
                }
            }
        };
        let mark_binary =
            |op: &BinaryOperator, inputs: &mut Vec<TensorId>, outputs: &mut Vec<TensorId>| {
                mark(&op.lhs, inputs);
                mark(&op.rhs, inputs);
                mark(&op.out, outputs);
            };
        let mark_unary =
            |op: &UnaryOperator, inputs: &mut Vec<TensorId>, outputs: &mut Vec<TensorId>| {
                mark(&op.input, inputs);
                mark(&op.out, outputs);
            };

        // For all operators, mark their local tensor id in the proper set.
        for op in self.scope.operations.iter() {
            match op {
                Operation::Operator(op) => match op {
                    Operator::Assign(op) => {
                        mark(&op.out, &mut local_tensor_ids_output);
                    }
                    Operator::Add(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Fma(op) => {
                        mark(&op.a, &mut local_tensor_ids_input);
                        mark(&op.b, &mut local_tensor_ids_input);
                        mark(&op.c, &mut local_tensor_ids_input);
                        mark(&op.out, &mut local_tensor_ids_output);
                    }
                    Operator::Slice(op) => {
                        mark(&op.input, &mut local_tensor_ids_input);
                        mark(&op.start, &mut local_tensor_ids_input);
                        mark(&op.out, &mut local_tensor_ids_output);
                    }
                    Operator::Max(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),

                    Operator::Min(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::And(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Or(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Not(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Neg(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Index(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::UncheckedIndex(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Sub(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Mul(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Div(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Exp(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Abs(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Round(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Erf(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Log(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Log1p(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Cos(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Sin(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Tanh(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Clamp(op) => {
                        mark(&op.input, &mut local_tensor_ids_input);
                        mark(&op.out, &mut local_tensor_ids_output);
                    }
                    Operator::Powf(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Recip(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Lower(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Greater(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::LowerEqual(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::GreaterEqual(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Equal(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::NotEqual(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Sqrt(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Floor(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Ceil(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Modulo(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::IndexAssign(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::UncheckedIndexAssign(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::BitwiseOr(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::BitwiseAnd(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::BitwiseXor(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::ShiftLeft(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::ShiftRight(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Remainder(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Bitcast(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::AtomicLoad(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::AtomicStore(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::AtomicSwap(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::AtomicAdd(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::AtomicSub(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::AtomicMax(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::AtomicMin(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::AtomicAnd(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::AtomicOr(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::AtomicXor(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Normalize(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::Magnitude(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Operator::AtomicCompareAndSwap(_op) => {
                        // Nothing to do.
                    }
                },
                Operation::Procedure(proc) => {
                    match proc {
                        Procedure::ReadGlobalWithLayout(_) => {
                            // Nothing to do here.
                        }
                        Procedure::ReadGlobal(_) => {
                            // Nothing to do here.
                        }
                        Procedure::WriteGlobal(_) => {
                            // Nothing to do here.
                        }
                        Procedure::CheckedIndex(_) => {
                            // Nothing to do here.
                        }
                        Procedure::CheckedIndexAssign(_) => {
                            // Nothing to do here.
                        }
                        Procedure::ConditionalAssign(proc) => {
                            mark(&proc.cond, &mut local_tensor_ids_input);
                            mark(&proc.lhs, &mut local_tensor_ids_input);
                            mark(&proc.rhs, &mut local_tensor_ids_input);
                            mark(&proc.out, &mut local_tensor_ids_output);
                        }
                        Procedure::IndexOffsetGlobalWithLayout(_) => {
                            // Nothing to do here.
                        }
                        Procedure::EarlyReturn(_) => {
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
                Operation::Subcube(op) => match op {
                    Subcube::Elect(op) => {
                        mark(&op.out, &mut local_tensor_ids_output);
                    }
                    Subcube::All(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Subcube::Any(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Subcube::Broadcast(op) => mark_binary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Subcube::Sum(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Subcube::Prod(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Subcube::Min(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                    Subcube::Max(op) => mark_unary(
                        op,
                        &mut local_tensor_ids_input,
                        &mut local_tensor_ids_output,
                    ),
                },
                Operation::CoopMma(_) => {
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
            let (tensor, _, _position) = &entry;
            if let TensorStatus::ReadOnly = tensor.status {
                if self.output_to_local.contains_key(&tensor.id) {
                    outputs.push(entry.clone());
                }
            }
        }

        outputs
    }
}
