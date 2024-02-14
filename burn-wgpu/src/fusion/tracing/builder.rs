use super::{trace::Trace, Scalars};
use crate::codegen::dialect::gpu::{self, Operation};
use burn_fusion::{TensorDescription, TensorId};
use burn_tensor::Element;
use hashbrown::HashMap;

/// A tracer is the captured computation that should be done.
///
/// It seves as an intermediary step between the high level fusion description based representation
/// and the lower level [gpu dialect](gpu).
#[derive(Clone)]
pub struct TraceBuilder {
    inputs: Vec<TensorDescription>,
    locals: HashMap<TensorId, u16>,
    tensors: HashMap<TensorId, (TensorDescription, gpu::Elem)>,
    scalars: Scalars,
    scope: gpu::Scope,
}

impl TraceBuilder {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            locals: HashMap::new(),
            tensors: HashMap::new(),
            scalars: Scalars::default(),
            scope: gpu::Scope::root(),
        }
    }

    pub fn register_operation<T: Into<gpu::Operation>>(&mut self, value: T) {
        self.scope.register(value)
    }

    pub fn input_to_var(&mut self, tensor: &TensorDescription, elem: gpu::Elem) -> gpu::Variable {
        let already_exists = self.tensors.contains_key(&tensor.id);

        let variable = match already_exists {
            false => {
                // New input
                let var = gpu::Variable::Input(self.inputs.len() as u16, gpu::Item::Scalar(elem));
                self.inputs.push(tensor.clone());
                var
            }
            true => match self.locals.get(&tensor.id) {
                // Is a local variable.
                Some(local_index) => {
                    gpu::Variable::Local(*local_index, gpu::Item::Scalar(elem), self.scope.depth)
                }
                // Isn't a local variable, so must be an existing input.
                None => {
                    let input = self
                        .inputs
                        .iter()
                        .enumerate()
                        .find(|(_, input)| input.id == tensor.id)
                        .unwrap();
                    let input_index = input.0;
                    gpu::Variable::Input(input_index as u16, gpu::Item::Scalar(elem))
                }
            },
        };

        // Update the tensor description with the new version.
        self.tensors.insert(tensor.id, (tensor.clone(), elem));

        variable
    }

    pub fn output_to_var(&mut self, tensor: &TensorDescription, elem: gpu::Elem) -> gpu::Variable {
        // Update the tensor description to the new version.
        self.tensors.insert(tensor.id, (tensor.clone(), elem));

        // Output already registered as a local variable.
        if let Some(index) = self.locals.get(&tensor.id) {
            return gpu::Variable::Local(*index, gpu::Item::Scalar(elem), self.scope.depth);
        }

        let variable = self.scope.create_local(gpu::Item::Scalar(elem));
        let local_index = variable.index().unwrap();
        self.locals.insert(tensor.id, local_index);
        variable
    }

    pub fn scalar_to_var<E: Element>(&mut self, _value: &E, elem_type: gpu::Elem) -> gpu::Variable {
        match elem_type {
            gpu::Elem::Float => {
                self.scalars.num_float += 1;
                gpu::Variable::Scalar(
                    self.scalars.num_float as u16 - 1,
                    gpu::Item::Scalar(gpu::Elem::Float),
                )
            }
            gpu::Elem::Int => {
                self.scalars.num_int += 1;
                gpu::Variable::Scalar(
                    self.scalars.num_int as u16 - 1,
                    gpu::Item::Scalar(gpu::Elem::Int),
                )
            }
            gpu::Elem::UInt => {
                self.scalars.num_uint += 1;
                gpu::Variable::Scalar(
                    self.scalars.num_uint as u16 - 1,
                    gpu::Item::Scalar(gpu::Elem::UInt),
                )
            }
            gpu::Elem::Bool => {
                self.scalars.num_bool += 1;
                gpu::Variable::Scalar(
                    self.scalars.num_bool as u16 - 1,
                    gpu::Item::Scalar(gpu::Elem::Bool),
                )
            }
        }
    }

    pub fn build(self) -> Trace {
        let inputs = self.input_descriptions();
        let outputs = self.output_descriptions();
        let locals = outputs
            .iter()
            .map(|out| *self.locals.get(&out.0.id).unwrap())
            .collect::<Vec<_>>();

        Trace::new(inputs, outputs, locals, self.scalars, self.scope)
    }

    fn input_descriptions(&self) -> Vec<(TensorDescription, gpu::Elem)> {
        self.inputs
            .iter()
            .map(|input| {
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
                    .locals
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
                Operation::Operator(op) => {
                    match op {
                        gpu::Operator::AssignGlobal(_) => {
                            // Nothing to do here.
                        }
                        gpu::Operator::AssignLocal(op) => {
                            mark(&op.out, &mut local_tensor_ids_output);
                        }
                        gpu::Operator::ReadGlobalWithLayout(_) => {
                            // Nothing to do here.
                        }
                        gpu::Operator::ReadGlobal(_) => {
                            // Nothing to do here.
                        }
                        gpu::Operator::Add(op) => mark_binary(
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
                        gpu::Operator::ConditionalAssign(op) => {
                            mark(&op.cond, &mut local_tensor_ids_input);
                            mark(&op.lhs, &mut local_tensor_ids_input);
                            mark(&op.rhs, &mut local_tensor_ids_input);
                            mark(&op.out, &mut local_tensor_ids_output);
                        }
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
                    }
                }
                Operation::Algorithm(_) => {}
                Operation::Metadata(_) => {
                    // Nothing to do, should never impact read-write access to bindings.
                }
                Operation::Loop(_) => {
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
                if self.locals.contains_key(&tensor.id) {
                    outputs.push(entry.clone());
                }
            }
        }

        outputs
    }
}
