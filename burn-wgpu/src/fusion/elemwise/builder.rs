use crate::{
    codegen::{Elem, Operator, Variable},
    element::WgpuElement,
    fusion::cache::KernelCompilationCache,
    FloatElement, GraphicsApi, IntElement, Wgpu,
};
use burn_common::id::IdGenerator;
use burn_fusion::{
    graph::{
        BaseOpsDescription, BinaryOpsDescription, FloatOpsDescription, NumericOpsDescription,
        ScalarOpsDescription, TensorOpsDescription, UnaryOpsDescription,
    },
    Optimization, OptimizationBuilder, OptimizationProperties, OptimizationStatus,
    TensorDescription, TensorId,
};
use burn_tensor::{Device, Element};
use hashbrown::HashMap;

use super::optimization::ElementWise;

/// Fused element wise operations that are normally memory bound.
pub(crate) struct FloatElementWiseBuilder<G, F, I>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    pub(crate) inputs: Vec<TensorDescription>,
    pub(crate) locals: HashMap<TensorId, u16>,
    pub(crate) tensors: HashMap<TensorId, (TensorDescription, Elem)>,
    pub(crate) scalars_f32: usize,
    pub(crate) scalars_i32: usize,
    pub(crate) scalars_u32: usize,
    pub(crate) booleans: usize,
    pub(crate) operators: Vec<Operator>,
    pub(crate) current_output_shape: Vec<usize>,
    pub(crate) status: OptimizationStatus,
    pub(crate) device: Device<Wgpu<G, F, I>>,
}

impl<G, F, I> OptimizationBuilder<Wgpu<G, F, I>> for FloatElementWiseBuilder<G, F, I>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    fn register(&mut self, ops: &TensorOpsDescription) {
        if let OptimizationStatus::Closed = self.status {
            return;
        }

        match ops {
            TensorOpsDescription::BaseOpsFloat(ops) => {
                if !self.register_base::<F>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            TensorOpsDescription::BaseOpsInt(ops) => {
                if !self.register_base::<I>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            TensorOpsDescription::FloatOps(ops) => {
                if !self.register_float::<F>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            TensorOpsDescription::NumericOpsFloat(ops) => {
                if !self.register_numeric::<F, _>(ops) {
                    self.status = OptimizationStatus::Closed;
                    return;
                }
            }
            TensorOpsDescription::NumericOpsInt(ops) => {
                if !self.register_numeric::<I, _>(ops) {
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

    fn build(&self) -> Box<dyn Optimization<Wgpu<G, F, I>>> {
        let inputs = self.input_descriptions();
        let outputs = self.output_descriptions();
        let locals = outputs
            .iter()
            .map(|out| *self.locals.get(&out.0.id).unwrap())
            .collect::<Vec<_>>();

        Box::new(ElementWise {
            id: IdGenerator::generate(),
            inputs,
            outputs,
            locals,
            operators: self.operators.clone(),
            scalars_f32: self.scalars_f32,
            scalars_u32: self.scalars_u32,
            scalars_i32: self.scalars_i32,
            device: self.device.clone(),
            cache: KernelCompilationCache::default(),
        })
    }

    fn reset(&mut self) {
        self.inputs.clear();
        self.locals.drain();
        self.tensors.clear();
        self.scalars_f32 = 0;
        self.scalars_i32 = 0;
        self.scalars_u32 = 0;
        self.booleans = 0;
        self.operators.clear();
        self.status = OptimizationStatus::Open;
        self.current_output_shape.clear();
    }

    fn status(&self) -> OptimizationStatus {
        self.status
    }

    fn properties(&self) -> OptimizationProperties {
        let ready = self.operators.len() > 1;

        OptimizationProperties {
            ready,
            score: self.operators.len() as u64,
        }
    }
}

impl<G, F, I> FloatElementWiseBuilder<G, F, I>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    pub fn new(device: Device<Wgpu<G, F, I>>) -> Self {
        Self {
            inputs: Vec::new(),
            locals: HashMap::new(),
            tensors: HashMap::new(),
            scalars_f32: 0,
            scalars_i32: 0,
            scalars_u32: 0,
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
                        list.push(id.clone());
                    }
                }
            }
        };

        // For all operators, mark their local tensor id in the proper set.
        for ops in self.operators.iter() {
            match ops {
                Operator::AssignGlobal { input: _, out: _ } => {
                    // Nothing to do here.
                }
                Operator::AssignLocal { input: _, out: _ } => {
                    // Nothing to do here.
                }
                Operator::ReadGlobalWithLayout {
                    variable: _,
                    tensor_read_pos: _,
                    tensor_layout_pos: _,
                } => {
                    // Nothing to do here.
                }
                Operator::ReadGlobal { variable: _ } => {
                    // Nothing to do here.
                }
                Operator::Add { lhs, rhs, out } => {
                    mark(lhs, &mut local_tensor_ids_input);
                    mark(rhs, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Sub { lhs, rhs, out } => {
                    mark(lhs, &mut local_tensor_ids_input);
                    mark(rhs, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Mul { lhs, rhs, out } => {
                    mark(lhs, &mut local_tensor_ids_input);
                    mark(rhs, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Div { lhs, rhs, out } => {
                    mark(lhs, &mut local_tensor_ids_input);
                    mark(rhs, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Exp { input, out } => {
                    mark(input, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Abs { input, out } => {
                    mark(input, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Erf { input, out } => {
                    mark(input, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Log { input, out } => {
                    mark(input, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Log1p { input, out } => {
                    mark(input, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Cos { input, out } => {
                    mark(input, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Sin { input, out } => {
                    mark(input, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Tanh { input, out } => {
                    mark(input, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Clamp {
                    input,
                    min_value: _,
                    max_value: _,
                    out,
                } => {
                    mark(input, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Powf { lhs, rhs, out } => {
                    mark(lhs, &mut local_tensor_ids_input);
                    mark(rhs, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Recip { input, out } => {
                    mark(input, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Lower { lhs, rhs, out } => {
                    mark(lhs, &mut local_tensor_ids_input);
                    mark(rhs, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Greater { lhs, rhs, out } => {
                    mark(lhs, &mut local_tensor_ids_input);
                    mark(rhs, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::LowerEqual { lhs, rhs, out } => {
                    mark(lhs, &mut local_tensor_ids_input);
                    mark(rhs, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::GreaterEqual { lhs, rhs, out } => {
                    mark(lhs, &mut local_tensor_ids_input);
                    mark(rhs, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Equal { lhs, rhs, out } => {
                    mark(lhs, &mut local_tensor_ids_input);
                    mark(rhs, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::ConditionalAssign {
                    cond,
                    lhs,
                    rhs,
                    out,
                } => {
                    mark(cond, &mut local_tensor_ids_input);
                    mark(lhs, &mut local_tensor_ids_input);
                    mark(rhs, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
                Operator::Sqrt { input, out } => {
                    mark(input, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
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

    fn input_to_var(&mut self, tensor: &TensorDescription, elem: Elem) -> Variable {
        let already_exists = self.tensors.contains_key(&tensor.id);

        let variable = match already_exists {
            false => {
                // New input
                let var = Variable::Input(self.inputs.len() as u16, elem);
                self.inputs.push(tensor.clone());
                var
            }
            true => match self.locals.get(&tensor.id) {
                // Is a local variable.
                Some(local_index) => Variable::Local(*local_index, elem),
                // Isn't a local variable, so must be an existing input.
                None => {
                    let input = self
                        .inputs
                        .iter()
                        .enumerate()
                        .find(|(_, input)| input.id == tensor.id)
                        .unwrap();
                    let input_index = input.0;
                    Variable::Input(input_index as u16, elem)
                }
            },
        };

        // Update the tensor description with the new version.
        self.tensors
            .insert(tensor.id.clone(), (tensor.clone(), elem));

        variable
    }

    fn output_to_var(&mut self, tensor: &TensorDescription, elem: Elem) -> Variable {
        // Update the tensor description to the new version.
        self.tensors
            .insert(tensor.id.clone(), (tensor.clone(), elem));

        // Output already registered as a local variable.
        if let Some(index) = self.locals.get(&tensor.id) {
            return Variable::Local(*index, elem);
        }

        // New local variable.
        let local_index = self.locals.len() as u16;
        self.locals.insert(tensor.id.clone(), local_index);
        Variable::Local(local_index, elem)
    }

    fn register_base<E: WgpuElement>(&mut self, ops: &BaseOpsDescription) -> bool {
        match ops {
            BaseOpsDescription::Equal(desc) => self.register_binary_ops(
                desc,
                (E::elem_type(), E::elem_type(), Elem::Bool),
                |lhs, rhs, out| Operator::Equal { lhs, rhs, out },
            ),
            _ => false,
        }
    }

    fn register_float<E: WgpuElement>(&mut self, ops: &FloatOpsDescription) -> bool {
        match ops {
            FloatOpsDescription::Exp(desc) => {
                self.register_unary_ops(desc, (E::elem_type(), E::elem_type()), |input, out| {
                    Operator::Exp { input, out }
                })
            }
            FloatOpsDescription::Log(desc) => {
                self.register_unary_ops(desc, (E::elem_type(), E::elem_type()), |input, out| {
                    Operator::Log { input, out }
                })
            }
            FloatOpsDescription::Log1p(desc) => {
                self.register_unary_ops(desc, (E::elem_type(), E::elem_type()), |input, out| {
                    Operator::Log1p { input, out }
                })
            }
            FloatOpsDescription::Cos(desc) => {
                self.register_unary_ops(desc, (E::elem_type(), E::elem_type()), |input, out| {
                    Operator::Cos { input, out }
                })
            }
            FloatOpsDescription::Sin(desc) => {
                self.register_unary_ops(desc, (E::elem_type(), E::elem_type()), |input, out| {
                    Operator::Sin { input, out }
                })
            }
            FloatOpsDescription::Powf(desc) => self.register_scalar_ops(
                desc,
                (E::elem_type(), E::elem_type(), E::elem_type()),
                |lhs, rhs, out| Operator::Powf { lhs, rhs, out },
            ),
            FloatOpsDescription::Tanh(desc) => {
                self.register_unary_ops(desc, (E::elem_type(), E::elem_type()), |input, out| {
                    Operator::Tanh { input, out }
                })
            }
            FloatOpsDescription::Erf(desc) => {
                self.register_unary_ops(desc, (E::elem_type(), E::elem_type()), |input, out| {
                    Operator::Erf { input, out }
                })
            }
            FloatOpsDescription::Recip(desc) => {
                self.register_unary_ops(desc, (E::elem_type(), E::elem_type()), |input, out| {
                    Operator::Recip { input, out }
                })
            }
            _ => false,
        }
    }

    fn register_numeric<E: WgpuElement, EDesc: WgpuElement>(
        &mut self,
        ops: &NumericOpsDescription<EDesc>,
    ) -> bool {
        match ops {
            NumericOpsDescription::Add(desc) => self.register_binary_ops(
                desc,
                (E::elem_type(), E::elem_type(), E::elem_type()),
                |lhs, rhs, out| Operator::Add { lhs, rhs, out },
            ),
            NumericOpsDescription::AddScalar(desc) => self.register_scalar_ops(
                desc,
                (E::elem_type(), E::elem_type(), E::elem_type()),
                |lhs, rhs, out| Operator::Add { lhs, rhs, out },
            ),
            NumericOpsDescription::Sub(desc) => self.register_binary_ops(
                desc,
                (E::elem_type(), E::elem_type(), E::elem_type()),
                |lhs, rhs, out| Operator::Sub { lhs, rhs, out },
            ),
            NumericOpsDescription::SubScalar(desc) => self.register_scalar_ops(
                desc,
                (E::elem_type(), E::elem_type(), E::elem_type()),
                |lhs, rhs, out| Operator::Sub { lhs, rhs, out },
            ),
            NumericOpsDescription::Mul(desc) => self.register_binary_ops(
                desc,
                (E::elem_type(), E::elem_type(), E::elem_type()),
                |lhs, rhs, out| Operator::Mul { lhs, rhs, out },
            ),
            NumericOpsDescription::MulScalar(desc) => self.register_scalar_ops(
                desc,
                (E::elem_type(), E::elem_type(), E::elem_type()),
                |lhs, rhs, out| Operator::Mul { lhs, rhs, out },
            ),
            NumericOpsDescription::Div(desc) => self.register_binary_ops(
                desc,
                (E::elem_type(), E::elem_type(), E::elem_type()),
                |lhs, rhs, out| Operator::Div { lhs, rhs, out },
            ),
            NumericOpsDescription::DivScalar(desc) => self.register_scalar_ops(
                desc,
                (E::elem_type(), E::elem_type(), E::elem_type()),
                |lhs, rhs, out| Operator::Div { lhs, rhs, out },
            ),
            NumericOpsDescription::Abs(desc) => {
                self.register_unary_ops(desc, (E::elem_type(), E::elem_type()), |input, out| {
                    Operator::Abs { input, out }
                })
            }
            NumericOpsDescription::Lower(desc) => self.register_binary_ops(
                desc,
                (E::elem_type(), E::elem_type(), Elem::Bool),
                |lhs, rhs, out| Operator::Lower { lhs, rhs, out },
            ),
            NumericOpsDescription::LowerElem(desc) => self.register_scalar_ops(
                desc,
                (E::elem_type(), E::elem_type(), Elem::Bool),
                |lhs, rhs, out| Operator::Lower { lhs, rhs, out },
            ),
            NumericOpsDescription::Greater(desc) => self.register_binary_ops(
                desc,
                (E::elem_type(), E::elem_type(), Elem::Bool),
                |lhs, rhs, out| Operator::Greater { lhs, rhs, out },
            ),
            NumericOpsDescription::GreaterElem(desc) => self.register_scalar_ops(
                desc,
                (E::elem_type(), E::elem_type(), Elem::Bool),
                |lhs, rhs, out| Operator::Greater { lhs, rhs, out },
            ),
            NumericOpsDescription::LowerEqual(desc) => self.register_binary_ops(
                desc,
                (E::elem_type(), E::elem_type(), Elem::Bool),
                |lhs, rhs, out| Operator::LowerEqual { lhs, rhs, out },
            ),
            NumericOpsDescription::LowerEqualElem(desc) => self.register_scalar_ops(
                desc,
                (E::elem_type(), E::elem_type(), Elem::Bool),
                |lhs, rhs, out| Operator::LowerEqual { lhs, rhs, out },
            ),
            NumericOpsDescription::GreaterEqual(desc) => self.register_binary_ops(
                desc,
                (E::elem_type(), E::elem_type(), Elem::Bool),
                |lhs, rhs, out| Operator::GreaterEqual { lhs, rhs, out },
            ),
            NumericOpsDescription::GreaterEqualElem(desc) => self.register_scalar_ops(
                desc,
                (E::elem_type(), E::elem_type(), Elem::Bool),
                |lhs, rhs, out| Operator::GreaterEqual { lhs, rhs, out },
            ),
            NumericOpsDescription::EqualElem(desc) => self.register_scalar_ops(
                desc,
                (E::elem_type(), E::elem_type(), Elem::Bool),
                |lhs, rhs, out| Operator::Equal { lhs, rhs, out },
            ),
            NumericOpsDescription::MaskWhere(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                let cond = self.input_to_var(&desc.mask, Elem::Bool);
                let lhs = self.input_to_var(&desc.value, E::elem_type());
                let rhs = self.input_to_var(&desc.tensor, E::elem_type());
                let out = self.output_to_var(&desc.out, E::elem_type());

                let ops = Operator::ConditionalAssign {
                    cond,
                    lhs,
                    rhs,
                    out,
                };
                self.operators.push(ops);

                true
            }
            NumericOpsDescription::MaskFill(desc) => {
                if !self.output_is_compatible(&desc.out) {
                    return false;
                }

                let cond = self.input_to_var(&desc.mask, Elem::Bool);
                let lhs = self.scalar_to_var(&desc.value, E::elem_type());
                let rhs = self.input_to_var(&desc.tensor, E::elem_type());
                let out = self.output_to_var(&desc.out, E::elem_type());

                self.operators.push(Operator::ConditionalAssign {
                    cond,
                    lhs,
                    rhs,
                    out,
                });

                true
            }
            _ => false,
        }
    }

    fn register_binary_ops<Func>(
        &mut self,
        desc: &BinaryOpsDescription,
        (elem_lhs, elem_rhs, elem_out): (Elem, Elem, Elem),
        func: Func,
    ) -> bool
    where
        Func: Fn(Variable, Variable, Variable) -> Operator,
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
        desc: &UnaryOpsDescription,
        (elem_input, elem_out): (Elem, Elem),
        func: Func,
    ) -> bool
    where
        Func: Fn(Variable, Variable) -> Operator,
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
        desc: &ScalarOpsDescription<E>,
        (elem_lhs, elem_rhs, elem_out): (Elem, Elem, Elem),
        func: Func,
    ) -> bool
    where
        Func: Fn(Variable, Variable, Variable) -> Operator,
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
            Elem::F32 => {
                self.scalars_f32 += 1;
                Variable::Scalar(self.scalars_f32 as u16 - 1, Elem::F32)
            }
            Elem::I32 => {
                self.scalars_i32 += 1;
                Variable::Scalar(self.scalars_i32 as u16 - 1, Elem::I32)
            }
            Elem::U32 => {
                self.scalars_u32 += 1;
                Variable::Scalar(self.scalars_u32 as u16 - 1, Elem::U32)
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
