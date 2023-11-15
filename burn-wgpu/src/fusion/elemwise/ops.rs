use crate::{
    fusion::codegen::{Elem, Operator, Variable},
    fusion::kernel::FusionKernel,
    FloatElement, GraphicsApi, IntElement, Wgpu,
};
use burn_fusion::{
    graph::{
        BinaryOpsDescription, FloatOpsDescription, NumericOpsDescription, ScalarOpsDescription,
        TensorOpsDescription, UnaryOpsDescription,
    },
    FusionBackend, FusionOps, FusionProperties, FusionStatus, HandleContainer, TensorDescription,
    TensorId,
};
use burn_tensor::{Device, Element};
use hashbrown::HashMap;
use std::sync::Arc;

/// Fused element wise operations that are normally memory bound.
pub struct FloatElementWiseFusionOps<G, F, I>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    pub(crate) inputs: Vec<TensorDescription>,
    pub(crate) locals: HashMap<TensorId, u16>,
    pub(crate) tensors: HashMap<TensorId, TensorDescription>,
    pub(crate) scalars_f32: Vec<f32>,
    pub(crate) operators: Vec<Operator>,
    pub(crate) properties: FusionProperties,
    pub(crate) current_output_shape: Vec<usize>,
    device: Device<Wgpu<G, F, I>>,
}

impl<G: GraphicsApi + 'static, F: FloatElement, I: IntElement> FusionOps<Wgpu<G, F, I>>
    for FloatElementWiseFusionOps<G, F, I>
{
    fn register(&mut self, ops: Arc<TensorOpsDescription<Wgpu<G, F, I>>>) -> FusionStatus {
        match ops.as_ref() {
            TensorOpsDescription::FloatOps(ops) => {
                if !self.register_float(ops) {
                    return FusionStatus::Closed(self.properties);
                }
            }
            TensorOpsDescription::NumericOpsFloat(ops) => {
                if !self.register_numeric(ops) {
                    return FusionStatus::Closed(self.properties);
                }
            }
            _ => {
                return FusionStatus::Closed(self.properties);
            }
        };

        self.properties.score += 1;
        self.properties.ready = self.operators.len() > 1;

        FusionStatus::Open(self.properties)
    }

    fn execute(&mut self, handles: &mut HandleContainer<Wgpu<G, F, I>>) {
        let inputs = self.input_descriptions();
        let outputs = self.output_descriptions();
        let locals = outputs
            .iter()
            .map(|out| *self.locals.get(&out.id).unwrap())
            .collect::<Vec<_>>();

        FusionKernel::new(&self.device)
            .inputs(&inputs, &self.scalars_f32)
            .body(&self.operators)
            .outputs(&outputs, &locals)
            .execute(handles);
    }

    fn reset(&mut self) {
        self.inputs.clear();
        self.locals.drain();
        self.tensors.clear();
        self.scalars_f32.clear();
        self.operators.clear();
        self.properties = FusionProperties::default();
        self.current_output_shape.clear();
    }

    fn len(&self) -> usize {
        self.operators.len()
    }
}

impl<G, F, I> FloatElementWiseFusionOps<G, F, I>
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
            scalars_f32: Vec::new(),
            operators: Vec::new(),
            current_output_shape: Vec::new(),
            properties: FusionProperties::default(),
            device,
        }
    }

    fn input_descriptions(&self) -> Vec<&TensorDescription> {
        self.inputs
            .iter()
            .map(|input| {
                let updated_tensor = self.tensors.get(&input.id).unwrap();
                updated_tensor
            })
            .collect::<Vec<_>>()
    }

    fn output_descriptions(&self) -> Vec<&TensorDescription> {
        let mut outputs = Vec::new();
        let mut local_tensor_ids_input = Vec::new();
        let mut local_tensor_ids_output = Vec::new();

        // Mark a variable to the provided list of tensor ids using the variable list.
        //
        // Only local variables can become outputs.
        let mark = |var: &Variable, list: &mut Vec<TensorId>| {
            if let Variable::Local(index) = var {
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
                Operator::ReadGlobal {
                    variable: _,
                    position: _,
                    position_out: _,
                } => {
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
                Operator::Powf { lhs, rhs, out } => {
                    mark(lhs, &mut local_tensor_ids_input);
                    mark(rhs, &mut local_tensor_ids_input);
                    mark(out, &mut local_tensor_ids_output);
                }
            }
        }

        // All output tensors that are never read by a following operation should be written to
        // since they are essentially the "logical" output of the shader.
        for out in local_tensor_ids_output {
            let is_read = local_tensor_ids_input.contains(&out);

            if !is_read {
                outputs.push(self.tensors.get(&out).unwrap());
            }
        }

        // All tensors where their latest description is read only should be written to since they
        // are going to be used after the fused kernel by other operations.
        for tensor in self.tensors.values() {
            if let burn_fusion::TensorStatus::ReadOnly = tensor.status {
                if self.locals.contains_key(&tensor.id) {
                    outputs.push(tensor);
                }
            }
        }

        outputs
    }

    fn input_to_var(&mut self, tensor: &TensorDescription) -> Variable {
        let already_exists = self.tensors.contains_key(&tensor.id);

        let variable = match already_exists {
            false => {
                // New input
                let var = Variable::Input(self.inputs.len() as u16);
                self.inputs.push(tensor.clone());
                var
            }
            true => match self.locals.get(&tensor.id) {
                // Is a local variable.
                Some(local_index) => Variable::Local(*local_index),
                // Isn't a local variable, so must be an existing input.
                None => {
                    let input = self
                        .inputs
                        .iter()
                        .enumerate()
                        .find(|(_, input)| input.id == tensor.id)
                        .unwrap();
                    let input_index = input.0;
                    Variable::Input(input_index as u16)
                }
            },
        };

        // Update the tensor description with the new version.
        self.tensors.insert(tensor.id.clone(), tensor.clone());

        variable
    }

    fn output_to_var(&mut self, tensor: &TensorDescription) -> Variable {
        // Update the tensor description to the new version.
        self.tensors.insert(tensor.id.clone(), tensor.clone());

        // Output already registered as a local variable.
        if let Some(index) = self.locals.get(&tensor.id) {
            return Variable::Local(*index);
        }

        // New local variable.
        let local_index = self.locals.len() as u16;
        self.locals.insert(tensor.id.clone(), local_index);
        Variable::Local(local_index)
    }

    fn register_float<B: FusionBackend>(&mut self, ops: &FloatOpsDescription<B>) -> bool {
        match ops {
            FloatOpsDescription::Exp(desc, _) => {
                self.register_unary_ops(desc, |input, out| Operator::Exp { input, out })
            }
            FloatOpsDescription::Log(desc, _) => {
                self.register_unary_ops(desc, |input, out| Operator::Log { input, out })
            }
            FloatOpsDescription::Log1p(desc, _) => {
                self.register_unary_ops(desc, |input, out| Operator::Log1p { input, out })
            }
            FloatOpsDescription::Cos(desc, _) => {
                self.register_unary_ops(desc, |input, out| Operator::Cos { input, out })
            }
            FloatOpsDescription::Sin(desc, _) => {
                self.register_unary_ops(desc, |input, out| Operator::Sin { input, out })
            }
            FloatOpsDescription::Powf(desc, _) => {
                self.register_scalar_ops(desc, |lhs, rhs, out| Operator::Powf { lhs, rhs, out })
            }
            FloatOpsDescription::Tanh(desc, _) => {
                self.register_unary_ops(desc, |input, out| Operator::Tanh { input, out })
            }
            FloatOpsDescription::Erf(desc, _) => {
                self.register_unary_ops(desc, |input, out| Operator::Erf { input, out })
            }
            _ => false,
        }
    }

    fn register_numeric<B: FusionBackend, E: Element>(
        &mut self,
        ops: &NumericOpsDescription<B, E>,
    ) -> bool {
        match ops {
            NumericOpsDescription::Add(desc, _) => {
                self.register_binary_ops(desc, |lhs, rhs, out| Operator::Add { lhs, rhs, out })
            }
            NumericOpsDescription::AddScalar(desc, _) => {
                self.register_scalar_ops(desc, |lhs, rhs, out| Operator::Add { lhs, rhs, out })
            }
            NumericOpsDescription::Sub(desc, _) => {
                self.register_binary_ops(desc, |lhs, rhs, out| Operator::Sub { lhs, rhs, out })
            }
            NumericOpsDescription::SubScalar(desc, _) => {
                self.register_scalar_ops(desc, |lhs, rhs, out| Operator::Sub { lhs, rhs, out })
            }
            NumericOpsDescription::Mul(desc, _) => {
                self.register_binary_ops(desc, |lhs, rhs, out| Operator::Mul { lhs, rhs, out })
            }
            NumericOpsDescription::MulScalar(desc, _) => {
                self.register_scalar_ops(desc, |lhs, rhs, out| Operator::Mul { lhs, rhs, out })
            }
            NumericOpsDescription::Div(desc, _) => {
                self.register_binary_ops(desc, |lhs, rhs, out| Operator::Div { lhs, rhs, out })
            }
            NumericOpsDescription::DivScalar(desc, _) => {
                self.register_scalar_ops(desc, |lhs, rhs, out| Operator::Div { lhs, rhs, out })
            }
            NumericOpsDescription::Abs(desc, _) => {
                self.register_unary_ops(desc, |input, out| Operator::Abs { input, out })
            }
            _ => false,
        }
    }

    fn register_binary_ops<Func>(&mut self, desc: &BinaryOpsDescription, func: Func) -> bool
    where
        Func: Fn(Variable, Variable, Variable) -> Operator,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let lhs = self.input_to_var(&desc.lhs);
        let rhs = self.input_to_var(&desc.rhs);
        let out = self.output_to_var(&desc.out);

        self.operators.push(func(lhs, rhs, out));

        true
    }

    fn register_unary_ops<Func>(&mut self, desc: &UnaryOpsDescription, func: Func) -> bool
    where
        Func: Fn(Variable, Variable) -> Operator,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let input = self.input_to_var(&desc.input);
        let out = self.output_to_var(&desc.out);

        self.operators.push(func(input, out));

        true
    }

    fn register_scalar_ops<Func, E: Element>(
        &mut self,
        desc: &ScalarOpsDescription<E>,
        func: Func,
    ) -> bool
    where
        Func: Fn(Variable, Variable, Variable) -> Operator,
    {
        if !self.output_is_compatible(&desc.out) {
            return false;
        }

        let lhs = self.input_to_var(&desc.lhs);
        let rhs = Variable::Scalar(self.scalars_f32.len() as u16, Elem::F32);
        self.scalars_f32.push(desc.rhs.elem());
        let out = self.output_to_var(&desc.out);

        self.operators.push(func(lhs, rhs, out));

        true
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_fusion::graph::{BinaryOpsDescription, Ops};
    use burn_fusion::Fusion;
    use burn_tensor::Tensor;

    struct FakeAddOps;

    impl<B: FusionBackend> Ops<B> for FakeAddOps {
        type Args = BinaryOpsDescription;

        fn execute(&self, _: &Self::Args, _: &mut HandleContainer<B>) {
            todo!()
        }
    }

    #[test]
    fn test_fusion_same_behavior() {
        type Backend = Wgpu;
        type FusedBackend = Fusion<Wgpu>;

        let data_1 =
            Tensor::<Backend, 2>::random([1, 32], burn_tensor::Distribution::Default).into_data();
        let data_2 =
            Tensor::<Backend, 2>::random([32, 32], burn_tensor::Distribution::Default).into_data();

        let tensor_1 = Tensor::<Backend, 2>::from_data(data_1.clone());
        let tensor_2 = Tensor::<Backend, 2>::from_data(data_2.clone());
        let tensor_3 = tensor_1.clone() + tensor_2;
        let tensor_4 = tensor_3.clone() - tensor_1;
        let tensor_5 = tensor_4 + 5.0;
        let tensor_6 = tensor_5 + tensor_3;
        let result_ref = tensor_6.into_data();

        let tensor_1 = Tensor::<FusedBackend, 2>::from_data(data_1);
        let tensor_2 = Tensor::<FusedBackend, 2>::from_data(data_2);
        let tensor_3 = tensor_1.clone() + tensor_2;
        let tensor_4 = tensor_3.clone() - tensor_1;
        let tensor_5 = tensor_4 + 5.0;
        let tensor_6 = tensor_5 + tensor_3;
        let result_fused = tensor_6.into_data();

        result_fused.assert_approx_eq(&result_ref, 3);
    }
}
