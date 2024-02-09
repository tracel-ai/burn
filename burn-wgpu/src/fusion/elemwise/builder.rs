use super::{optimization::ElementWise, CompilationPhase, Scalars};
use crate::{
    codegen::{
        dialect::gpu::{
            BinaryOperation, ConditionalAssignOperation, Elem, Item, Operation, UnaryOperation,
            Variable,
        },
        Compiler,
    },
    element::WgpuElement,
    fusion::WgpuOptimization,
    GpuBackend, Runtime,
};
use burn_fusion::{
    stream::{
        BaseOperationDescription, BinaryOperationDescription, FloatOperationDescription,
        NumericOperationDescription, OperationDescription, ScalarOperationDescription,
        UnaryOperationDescription,
    },
    OptimizationBuilder, OptimizationProperties, OptimizationStatus, TensorDescription, TensorId,
};
use burn_tensor::{
    ops::{FloatElem, IntElem},
    Device, Element,
};
use hashbrown::HashMap;

/// Fused element wise operations that are normally memory bound.
pub(crate) struct ElementWiseBuilder<R: Runtime> {
    pub(crate) inputs: Vec<TensorDescription>,
    pub(crate) locals: HashMap<TensorId, u16>,
    pub(crate) tensors: HashMap<TensorId, (TensorDescription, Elem)>,
    pub(crate) scalars_float: usize,
    pub(crate) scalars_int: usize,
    pub(crate) scalars_uint: usize,
    pub(crate) booleans: usize,
    pub(crate) operators: Vec<Operation>,
    pub(crate) current_output_shape: Vec<usize>,
    pub(crate) status: OptimizationStatus,
    pub(crate) device: R::Device,
}

impl<R: Runtime> OptimizationBuilder<WgpuOptimization<R>> for ElementWiseBuilder<R> {
    fn register(&mut self, ops: &OperationDescription) {
        if let OptimizationStatus::Closed = self.status {
            return;
        }

        match ops {
            OperationDescription::BaseFloat(ops) => {
                if !self.register_base::<FloatElem<GpuBackend<R>>>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::BaseInt(ops) => {
                if !self.register_base::<IntElem<GpuBackend<R>>>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::Float(ops) => {
                if !self.register_float::<FloatElem<GpuBackend<R>>>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::NumericFloat(ops) => {
                if !self.register_numeric::<FloatElem<GpuBackend<R>>, _>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            OperationDescription::NumericInt(ops) => {
                if !self.register_numeric::<IntElem<GpuBackend<R>>, _>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            _ => {
                self.status = OptimizationStatus::Closed;
                return;
            }
        };

        self.status = OptimizationStatus::Open;
    }

    fn build(&self) -> WgpuOptimization<R> {
        let inputs = self.input_descriptions();
        let outputs = self.output_descriptions();
        let locals = outputs
            .iter()
            .map(|out| *self.locals.get(&out.0.id).unwrap())
            .collect::<Vec<_>>();

        let op = ElementWise::new(
            inputs,
            outputs,
            locals,
            Scalars::new(self.scalars_float, self.scalars_uint, self.scalars_int),
            self.operators.clone(),
            self.device.clone(),
            CompilationPhase,
        );

        WgpuOptimization::ElementWise(op.compile())
    }

    fn len(&self) -> usize {
        self.operators.len()
    }

    fn reset(&mut self) {
        self.inputs.clear();
        self.locals.drain();
        self.tensors.clear();
        self.scalars_float = 0;
        self.scalars_int = 0;
        self.scalars_uint = 0;
        self.booleans = 0;
        self.operators.clear();
        self.status = OptimizationStatus::Open;
        self.current_output_shape.clear();
    }

    fn status(&self) -> OptimizationStatus {
        self.status
    }

    fn properties(&self) -> OptimizationProperties {
        let ready = !self.operators.is_empty();

        OptimizationProperties {
            ready,
            score: self.operators.len() as u64,
        }
    }
}

impl<R: Runtime> ElementWiseBuilder<R> {
    pub fn new(device: Device<GpuBackend<R>>) -> Self {
        Self {
            inputs: Vec::new(),
            locals: HashMap::new(),
            tensors: HashMap::new(),
            scalars_float: 0,
            scalars_int: 0,
            scalars_uint: 0,
            booleans: 0,
            operators: Vec::new(),
            current_output_shape: Vec::new(),
            status: OptimizationStatus::Open,
            device,
        }
    }

    fn input_descriptions(&self) -> Vec<(TensorDescription, Elem)> {
        self.inputs
            .iter()
            .map(|input| {
                let updated_tensor = self.tensors.get(&input.id).unwrap();
                updated_tensor.clone()
            })
            .collect::<Vec<_>>()
    }

    fn output_descriptions(&self) -> Vec<(TensorDescription, Elem)> {
        let mut outputs = Vec::new();
        let mut local_tensor_ids_input = Vec::new();
        let mut local_tensor_ids_output = Vec::new();

        // Mark a variable to the provided list of tensor ids using the variable list.
        //
        // Only local variables can become outputs.
        let mark = |var: &Variable, list: &mut Vec<TensorId>| {
            if let Variable::Local(index, _) = var {
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
            |op: &BinaryOperation, inputs: &mut Vec<TensorId>, outputs: &mut Vec<TensorId>| {
                mark(&op.lhs, inputs);
                mark(&op.rhs, inputs);
                mark(&op.out, outputs);
            };
        let mark_unary =
            |op: &UnaryOperation, inputs: &mut Vec<TensorId>, outputs: &mut Vec<TensorId>| {
                mark(&op.input, inputs);
                mark(&op.out, outputs);
            };

        // For all operators, mark their local tensor id in the proper set.
        for ops in self.operators.iter() {
            match ops {
                Operation::AssignGlobal(_) => {
                    // Nothing to do here.
                }
                Operation::AssignLocal(op) => {
                    mark(&op.out, &mut local_tensor_ids_output);
                }
                Operation::ReadGlobalWithLayout(_) => {
                    // Nothing to do here.
                }
                Operation::ReadGlobal(_) => {
                    // Nothing to do here.
                }
                Operation::Add(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Sub(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Mul(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Div(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Exp(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Abs(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Erf(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Log(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Log1p(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Cos(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Sin(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Tanh(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Clamp(op) => {
                    mark(&op.input, &mut local_tensor_ids_input);
                    mark(&op.out, &mut local_tensor_ids_output);
                }
                Operation::Powf(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Recip(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Lower(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Greater(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::LowerEqual(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::GreaterEqual(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::Equal(op) => mark_binary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
                Operation::ConditionalAssign(op) => {
                    mark(&op.cond, &mut local_tensor_ids_input);
                    mark(&op.lhs, &mut local_tensor_ids_input);
                    mark(&op.rhs, &mut local_tensor_ids_input);
                    mark(&op.out, &mut local_tensor_ids_output);
                }
                Operation::Sqrt(op) => mark_unary(
                    op,
                    &mut local_tensor_ids_input,
                    &mut local_tensor_ids_output,
                ),
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

    fn input_to_var(&mut self, tensor: &TensorDescription, elem: Elem) -> Variable {
        let already_exists = self.tensors.contains_key(&tensor.id);

        let variable = match already_exists {
            false => {
                // New input
                let var = Variable::Input(self.inputs.len() as u16, Item::Scalar(elem));
                self.inputs.push(tensor.clone());
                var
            }
            true => match self.locals.get(&tensor.id) {
                // Is a local variable.
                Some(local_index) => Variable::Local(*local_index, Item::Scalar(elem)),
                // Isn't a local variable, so must be an existing input.
                None => {
                    let input = self
                        .inputs
                        .iter()
                        .enumerate()
                        .find(|(_, input)| input.id == tensor.id)
                        .unwrap();
                    let input_index = input.0;
                    Variable::Input(input_index as u16, Item::Scalar(elem))
                }
            },
        };

        // Update the tensor description with the new version.
        self.tensors.insert(tensor.id, (tensor.clone(), elem));

        variable
    }

    fn output_to_var(&mut self, tensor: &TensorDescription, elem: Elem) -> Variable {
        // Update the tensor description to the new version.
        self.tensors.insert(tensor.id, (tensor.clone(), elem));

        // Output already registered as a local variable.
        if let Some(index) = self.locals.get(&tensor.id) {
            return Variable::Local(*index, Item::Scalar(elem));
        }

        // New local variable.
        let local_index = self.locals.len() as u16;
        self.locals.insert(tensor.id, local_index);
        Variable::Local(local_index, Item::Scalar(elem))
    }

    fn register_base<E: WgpuElement>(&mut self, ops: &BaseOperationDescription) -> bool {
        match ops {
            BaseOperationDescription::Equal(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::Equal(BinaryOperation { lhs, rhs, out }),
            ),
            _ => false,
        }
    }

    fn register_float<E: WgpuElement>(&mut self, ops: &FloatOperationDescription) -> bool {
        match ops {
            FloatOperationDescription::Exp(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Exp(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::Log(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Log(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::Log1p(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Log1p(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::Cos(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Cos(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::Sin(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Sin(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::PowfScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Powf(BinaryOperation { lhs, rhs, out }),
            ),
            FloatOperationDescription::Tanh(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Tanh(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::Erf(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Erf(UnaryOperation { input, out })
                })
            }
            FloatOperationDescription::Recip(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Recip(UnaryOperation { input, out })
                })
            }
            _ => false,
        }
    }

    fn register_numeric<E: WgpuElement, EDesc: WgpuElement>(
        &mut self,
        ops: &NumericOperationDescription<EDesc>,
    ) -> bool {
        match ops {
            NumericOperationDescription::Add(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Add(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::AddScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Add(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::Sub(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Sub(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::SubScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Sub(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::Mul(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Mul(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::MulScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Mul(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::Div(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Div(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::DivScalar(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), E::gpu_elem()),
                |lhs, rhs, out| Operation::Div(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::Abs(desc) => {
                self.register_unary_ops(desc, (E::gpu_elem(), E::gpu_elem()), |input, out| {
                    Operation::Abs(UnaryOperation { input, out })
                })
            }
            NumericOperationDescription::Lower(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::Lower(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::LowerElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::Lower(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::Greater(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::Greater(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::GreaterElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::Greater(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::LowerEqual(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::LowerEqual(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::LowerEqualElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::LowerEqual(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::GreaterEqual(desc) => self.register_binary_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::GreaterEqual(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::GreaterEqualElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::GreaterEqual(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::EqualElem(desc) => self.register_scalar_ops(
                desc,
                (E::gpu_elem(), E::gpu_elem(), Elem::Bool),
                |lhs, rhs, out| Operation::Equal(BinaryOperation { lhs, rhs, out }),
            ),
            NumericOperationDescription::MaskWhere(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                let cond = self.input_to_var(&desc.mask, Elem::Bool);
                let lhs = self.input_to_var(&desc.value, E::gpu_elem());
                let rhs = self.input_to_var(&desc.tensor, E::gpu_elem());
                let out = self.output_to_var(&desc.out, E::gpu_elem());

                let ops = Operation::ConditionalAssign(ConditionalAssignOperation {
                    cond,
                    lhs,
                    rhs,
                    out,
                });
                self.operators.push(ops);

                true
            }
            NumericOperationDescription::MaskFill(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                let cond = self.input_to_var(&desc.mask, Elem::Bool);
                let lhs = self.scalar_to_var(&desc.value, E::gpu_elem());
                let rhs = self.input_to_var(&desc.tensor, E::gpu_elem());
                let out = self.output_to_var(&desc.out, E::gpu_elem());

                self.operators
                    .push(Operation::ConditionalAssign(ConditionalAssignOperation {
                        cond,
                        lhs,
                        rhs,
                        out,
                    }));

                true
            }
            NumericOperationDescription::Ones(desc) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let input = Variable::Constant(1.0, Item::Scalar(E::gpu_elem()));
                let out = self.output_to_var(desc, E::gpu_elem());

                self.operators
                    .push(Operation::AssignLocal(UnaryOperation { input, out }));

                true
            }
            NumericOperationDescription::Zeros(desc) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let input = Variable::Constant(0.0, Item::Scalar(E::gpu_elem()));
                let out = self.output_to_var(desc, E::gpu_elem());

                self.operators
                    .push(Operation::AssignLocal(UnaryOperation { input, out }));

                true
            }
            NumericOperationDescription::Full((desc, elem)) => {
                if !self.output_is_compatible(desc) {
                    return false;
                }

                let input = self.scalar_to_var(elem, E::gpu_elem());
                let out = self.output_to_var(desc, E::gpu_elem());

                self.operators
                    .push(Operation::AssignLocal(UnaryOperation { input, out }));

                true
            }
            _ => false,
        }
    }

    fn register_binary_ops<Func>(
        &mut self,
        desc: &BinaryOperationDescription,
        (elem_lhs, elem_rhs, elem_out): (Elem, Elem, Elem),
        func: Func,
    ) -> bool
    where
        Func: Fn(Variable, Variable, Variable) -> Operation,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let lhs = self.input_to_var(&desc.lhs, elem_lhs);
        let rhs = self.input_to_var(&desc.rhs, elem_rhs);
        let out = self.output_to_var(&desc.out, elem_out);

        self.operators.push(func(lhs, rhs, out));

        true
    }

    fn register_unary_ops<Func>(
        &mut self,
        desc: &UnaryOperationDescription,
        (elem_input, elem_out): (Elem, Elem),
        func: Func,
    ) -> bool
    where
        Func: Fn(Variable, Variable) -> Operation,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let input = self.input_to_var(&desc.input, elem_input);
        let out = self.output_to_var(&desc.out, elem_out);

        self.operators.push(func(input, out));

        true
    }

    fn register_scalar_ops<Func, E: Element>(
        &mut self,
        desc: &ScalarOperationDescription<E>,
        (elem_lhs, elem_rhs, elem_out): (Elem, Elem, Elem),
        func: Func,
    ) -> bool
    where
        Func: Fn(Variable, Variable, Variable) -> Operation,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let lhs = self.input_to_var(&desc.lhs, elem_lhs);
        let rhs = self.scalar_to_var(&desc.rhs, elem_rhs);
        let out = self.output_to_var(&desc.out, elem_out);

        self.operators.push(func(lhs, rhs, out));

        true
    }

    fn scalar_to_var<E: Element>(&mut self, _value: &E, elem_type: Elem) -> Variable {
        match elem_type {
            Elem::Float => {
                self.scalars_float += 1;
                Variable::Scalar(self.scalars_float as u16 - 1, Item::Scalar(Elem::Float))
            }
            Elem::Int => {
                self.scalars_int += 1;
                Variable::Scalar(self.scalars_int as u16 - 1, Item::Scalar(Elem::Int))
            }
            Elem::UInt => {
                self.scalars_uint += 1;
                Variable::Scalar(self.scalars_uint as u16 - 1, Item::Scalar(Elem::UInt))
            }
            Elem::Bool => {
                panic!("Bool scalars not supported")
            }
        }
    }

    fn output_is_compatible(&mut self, out: &TensorDescription) -> bool {
        if self.current_output_shape.is_empty() {
            self.current_output_shape = out.shape.clone();
        } else if self.current_output_shape != out.shape {
            return false;
        }

        true
    }
}
