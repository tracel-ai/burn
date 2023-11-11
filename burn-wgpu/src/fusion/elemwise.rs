use crate::{
    compute::{DynamicKernel, Kernel, WgpuComputeClient, WgpuHandle},
    kernel::{elemwise_workgroup, DynamicKernelSource, SourceTemplate, WORKGROUP_DEFAULT},
    FloatElement, GraphicsApi, IntElement, Wgpu,
};
use burn_fusion::{
    graph::{NumericOpsDescription, TensorOpsDescription},
    FusionBackend, FusionOps, FusionProperties, FusionStatus, HandleContainer, TensorDescription,
    TensorId,
};
use burn_tensor::Element;
use core::fmt::Display;
use core::hash::Hash;
use hashbrown::{HashMap, HashSet};
use std::{collections::hash_map::DefaultHasher, hash::Hasher, sync::Arc};

use super::{Binding, WgslTempate};

pub struct FloatElementWiseFusionOps {
    inputs: Vec<TensorDescription>,
    temps: HashMap<TensorId, u16>,
    tensors: HashMap<TensorId, TensorDescription>,
    ops: Vec<Operator>,
    properties: FusionProperties,
}

impl Default for FloatElementWiseFusionOps {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            temps: HashMap::new(),
            tensors: HashMap::new(),
            ops: Vec::new(),
            properties: FusionProperties::default(),
        }
    }
}

impl<G: GraphicsApi + 'static, F: FloatElement, I: IntElement> FusionOps<Wgpu<G, F, I>>
    for FloatElementWiseFusionOps
{
    fn register(&mut self, ops: Arc<TensorOpsDescription<Wgpu<G, F, I>>>) -> FusionStatus {
        match ops.as_ref() {
            TensorOpsDescription::NumericOpsFloat(ops) => {
                if !self.register_numeric(ops) {
                    return FusionStatus::Closed(self.properties.clone());
                }
            }
            _ => return FusionStatus::Closed(self.properties.clone()),
        };

        self.properties.score += 1;
        self.properties.ready = self.ops.len() > 1;

        self.tensors
            .values()
            .for_each(|tensor| match tensor.status {
                burn_fusion::TensorStatus::ReadWrite => {}
                _ => self.properties.ready = false,
            });

        return FusionStatus::Open(self.properties.clone());
    }

    fn execute(&mut self, handles: &mut HandleContainer<Wgpu<G, F, I>>) {
        let (kernel, handles, client) = self.create_kernel(handles);
        let handles = handles.iter().collect::<Vec<_>>();

        client.execute(kernel, &handles);
    }

    fn reset(&mut self) {
        self.properties = FusionProperties::default();
        self.tensors.clear();
        self.ops.clear();
    }

    fn len(&self) -> usize {
        self.ops.len()
    }
}

impl FloatElementWiseFusionOps {
    fn output_descriptions(&self) -> Vec<TensorDescription> {
        let mut outputs = Vec::new();
        let mut read_tensor = HashSet::new();
        let mut out_tensor = HashSet::new();

        let mark_temp = |ident: &Ident, list: &mut HashSet<TensorId>| {
            match ident {
                Ident::Temp(index) => {
                    if let Some((id, _)) =
                        self.temps.iter().find(|(_id, position)| *position == index)
                    {
                        list.insert(id.clone());
                    }
                }
                _ => {}
            };
        };
        for ops in self.ops.iter() {
            match ops {
                Operator::Add { lhs, rhs, out } => {
                    mark_temp(lhs, &mut read_tensor);
                    mark_temp(rhs, &mut read_tensor);
                    mark_temp(out, &mut out_tensor);
                }
                Operator::Sub { lhs, rhs, out } => {
                    mark_temp(lhs, &mut read_tensor);
                    mark_temp(rhs, &mut read_tensor);
                    mark_temp(out, &mut out_tensor);
                }
                _ => {}
            }
        }

        for out in out_tensor {
            if !read_tensor.contains(&out) {
                outputs.push(self.tensors.get(&out).unwrap().clone());
            }
        }

        for tensor in self.tensors.values() {
            match tensor.status {
                burn_fusion::TensorStatus::ReadOnly => {
                    if self.temps.contains_key(&tensor.id) {
                        // If used after.
                        outputs.push(tensor.clone());
                    }
                }
                _ => (),
            }
        }

        outputs
    }
    fn create_kernel<G, F, I>(
        &mut self,
        handles: &mut HandleContainer<Wgpu<G, F, I>>,
    ) -> (Box<dyn Kernel>, Vec<WgpuHandle>, WgpuComputeClient)
    where
        G: GraphicsApi,
        F: FloatElement,
        I: IntElement,
    {
        // BUILD THE OUTPUTS
        let outputs = self.output_descriptions();
        let mut operations = Vec::new();

        // BUILD THE BINDINGS, INFO and HANDLES.
        let mut input_bindings = Vec::new();
        let mut output_bindings = Vec::new();
        let mut info = Vec::new();
        let mut kernel_handles = Vec::new();
        let mut client = None;

        // REGISTER INPUTS
        for (i, input) in self.inputs.iter().enumerate() {
            if info.is_empty() {
                info.push(input.shape.len()); // Rank
            }

            let mut handle = handles.get_handle(&input);
            info.append(&mut handle.strides);
            info.append(&mut input.shape.clone());
            kernel_handles.push(handle.handle);

            if let None = client {
                client = Some(handle.client.clone());
            }

            input_bindings.push(Binding {
                elem: super::Elem::F32,
                visibility: super::Visibility::Read,
                location: super::Location::Storage,
                size: None,
            });
            operations.push(Operator::ReadGlobal {
                ident: Ident::Input(i as u16),
                position: i,
                position_out: self.inputs.len(), // First output
            });
        }

        let client = client.unwrap();

        let calculate_num_elems = |shape: &[usize]| {
            let mut num_elems = 1;
            for i in shape.iter() {
                num_elems *= i;
            }
            num_elems
        };

        // REGISTER OUTPUTS
        operations.append(&mut self.ops.clone());

        for (i, output) in outputs.iter().enumerate() {
            if let Some(temp) = self.temps.get(&output.id) {
                operations.push(Operator::Assign {
                    input: Ident::Temp(*temp),
                    out: Ident::Output(i as u16),
                });
            }
        }

        let mut num_elems_launch_option = 0;
        for output in outputs {
            let num_elems_output = calculate_num_elems(&output.shape);
            if num_elems_launch_option == 0 {
                num_elems_launch_option = num_elems_output;
            }
            let handle = client.empty(num_elems_output * core::mem::size_of::<f32>());
            // TODO: handles.register_handle(handle);
            let mut strides = dyn_strides(&output.shape);

            info.append(&mut strides);
            info.append(&mut output.shape.clone());
            kernel_handles.push(handle);

            output_bindings.push(Binding {
                elem: super::Elem::F32,
                visibility: super::Visibility::ReadWrite,
                location: super::Location::Storage,
                size: None,
            });
        }

        // INFO
        let info_handle = client.create(bytemuck::cast_slice(&info));
        kernel_handles.push(info_handle);
        let info_binding = Some(Binding {
            elem: super::Elem::U32,
            visibility: super::Visibility::Read,
            location: super::Location::Storage,
            size: Some(info.len()),
        });

        let kernel = WgslTempate {
            inputs: input_bindings,
            outputs: output_bindings,
            info: info_binding,
            workgroup_sizes: super::WorkgroupSize::default(),
            body: Box::new(ElemWiseBody { operations }),
            num_workgroups: true,
            global_invocation_id: true,
        };

        let workgroup = elemwise_workgroup(num_elems_launch_option, WORKGROUP_DEFAULT);

        (
            Box::new(DynamicKernel::new(kernel, workgroup)),
            kernel_handles,
            client,
        )
    }
    fn input_to_ident(&mut self, tensor: &TensorDescription) -> Ident {
        let ident = match self.tensors.contains_key(&tensor.id) {
            false => {
                let ident = Ident::Input(self.inputs.len() as u16);
                self.inputs.push(tensor.clone());
                ident
            }
            true => match self.temps.get(&tensor.id) {
                Some(index) => Ident::Temp(*index),
                None => {
                    let input = self
                        .inputs
                        .iter()
                        .enumerate()
                        .find(|(_, input)| input.id == tensor.id)
                        .unwrap();
                    Ident::Input(input.0 as u16)
                }
            },
        };
        self.tensors.insert(tensor.id.clone(), tensor.clone());

        ident
    }

    fn output_to_ident(&mut self, tensor: &TensorDescription) -> Ident {
        let temp = self.ops.len() as u16;
        self.temps.insert(tensor.id.clone(), temp);
        self.tensors.insert(tensor.id.clone(), tensor.clone());
        Ident::Temp(temp)
    }

    fn register_numeric<B: FusionBackend, E: Element>(
        &mut self,
        ops: &NumericOpsDescription<B, E>,
    ) -> bool {
        match ops {
            NumericOpsDescription::Add(desc, _) => {
                let lhs = self.input_to_ident(&desc.lhs);
                let rhs = self.input_to_ident(&desc.rhs);
                let out = self.output_to_ident(&desc.out);

                self.ops.push(Operator::Add { lhs, rhs, out });

                return true;
            }
            _ => false,
        }
    }
}

#[derive(Hash, Clone)]
enum Ident {
    Input(u16),
    Temp(u16),
    Output(u16),
}

#[derive(Hash, Clone)]
enum Operator {
    Add {
        lhs: Ident,
        rhs: Ident,
        out: Ident,
    },
    Sub {
        lhs: Ident,
        rhs: Ident,
        out: Ident,
    },
    Assign {
        input: Ident,
        out: Ident,
    },
    ReadGlobal {
        ident: Ident,
        position: usize,
        position_out: usize,
    },
}

impl Display for Ident {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ident::Input(number) => f.write_str(format!("input_{number}").as_str()),
            Ident::Temp(number) => f.write_str(format!("temp_{number}").as_str()),
            Ident::Output(number) => f.write_str(format!("output_{number}_global[id]").as_str()),
            _ => todo!(),
        }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Add { lhs, rhs, out } => f.write_str(&format!("let {out} = {lhs} + {rhs};")),
            Operator::Sub { lhs, rhs, out } => f.write_str(&format!("let {out} = {lhs} - {rhs};")),
            Operator::Assign { input, out } => f.write_str(&format!("let {out} = {input};")),
            Operator::ReadGlobal {
                ident,
                position,
                position_out,
            } => {
                let (global, local) = match ident {
                    Ident::Input(number) => {
                        (format!("input_{number}_global"), format!("input_{number}"))
                    }
                    Ident::Temp(_) => panic!("can't ready global a temp ident"),
                    Ident::Output(number) => (
                        format!("output_{number}_global"),
                        format!("output_{number}"),
                    ),
                };

                f.write_str(&format!(
                    "
var index_{local}: u32 = 0u;

for (var i: u32 = 1u; i <= dim; i++) {{
    let position = {position} * (2 * dim);
    let position_out = {position_out} * (2 * dim);

    let stride = info[position + i];
    let stride_out = info[position_out + i];
    let shape = info[position + dim + i];

    index_{local} += id / stride_out % shape * stride;
}}

let {local} = {global}[index_{local}];
"
                ))
            }
        }
    }
}

#[derive(Hash)]
pub struct ElemWiseBody {
    operations: Vec<Operator>,
}

impl Display for ElemWiseBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "let id = global_id.y * (num_workgroups.x * WORKGROUP_SIZE_X) + global_id.x;\n",
        )?;
        f.write_str("let dim: u32 = info[0];\n\n")?;

        for ops in self.operations.iter() {
            f.write_str(ops.to_string().as_str())?;
            f.write_str("\n")?;
        }

        Ok(())
    }
}

impl DynamicKernelSource for ElemWiseBody {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(self.to_string())
    }

    fn id(&self) -> String {
        let mut s = DefaultHasher::new();
        self.hash(&mut s);

        s.finish().to_string()
    }
}
fn dyn_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];

    let mut current = 1;
    shape.iter().enumerate().rev().for_each(|(index, val)| {
        strides[index] = current;
        current *= val;
    });

    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{TestBackend, TestTensor};
    use burn_fusion::graph::{BinaryOpsDescription, Ops};
    use burn_fusion::TensorStatus;

    struct FakeAddOps;

    impl<B: FusionBackend> Ops<B> for FakeAddOps {
        type Args = BinaryOpsDescription;

        fn execute(&self, _: &Self::Args, _: &mut HandleContainer<B>) {
            todo!()
        }
    }

    #[test]
    fn test_source_one_operation() {
        let mut optimization = FloatElementWiseFusionOps::default();
        let mut handles = HandleContainer::<TestBackend>::default();

        let tensor1 = TestTensor::ones([32, 32]);
        let tensor1_desc = TensorDescription {
            id: TensorId::new(0),
            shape: vec![32, 32],
            status: TensorStatus::ReadOnly,
        };
        let tensor2 = TestTensor::ones([32, 32]);
        let tensor2_desc = TensorDescription {
            id: TensorId::new(1),
            shape: vec![32, 32],
            status: TensorStatus::ReadOnly,
        };
        let tensor3_desc = TensorDescription {
            id: TensorId::new(2),
            shape: vec![32, 32],
            status: TensorStatus::NotInit,
        };

        handles.register_float_tensor(&tensor1_desc.id, tensor1.clone().into_primitive());
        handles.register_float_tensor(&tensor2_desc.id, tensor2.clone().into_primitive());

        let ops =
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::<TestBackend, f32>::Add(
                BinaryOpsDescription {
                    lhs: tensor1_desc,
                    rhs: tensor2_desc,
                    out: tensor3_desc,
                },
                Box::new(FakeAddOps),
            ));

        optimization.register(Arc::new(ops));

        let (kernel, handles, client) = optimization.create_kernel(&mut handles);

        let source = kernel.source().complete();
        println!("{source}");
        pretty_assertions::assert_eq!(source, "");
    }

    #[test]
    fn test_source_two_operations() {
        let mut optimization = FloatElementWiseFusionOps::default();
        let mut handles = HandleContainer::<TestBackend>::default();

        let tensor1 = TestTensor::ones([32, 32]);
        let tensor1_desc = TensorDescription {
            id: TensorId::new(0),
            shape: vec![32, 32],
            status: TensorStatus::ReadOnly,
        };
        let tensor2 = TestTensor::ones([32, 32]);
        let tensor2_desc = TensorDescription {
            id: TensorId::new(1),
            shape: vec![32, 32],
            status: TensorStatus::ReadOnly,
        };
        let tensor3_desc = TensorDescription {
            id: TensorId::new(2),
            shape: vec![32, 32],
            status: TensorStatus::NotInit,
        };
        let tensor4_desc = TensorDescription {
            id: TensorId::new(3),
            shape: vec![32, 32],
            status: TensorStatus::NotInit,
        };

        handles.register_float_tensor(&tensor1_desc.id, tensor1.clone().into_primitive());
        handles.register_float_tensor(&tensor2_desc.id, tensor2.clone().into_primitive());

        let ops =
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::<TestBackend, f32>::Add(
                BinaryOpsDescription {
                    lhs: tensor1_desc.clone(),
                    rhs: tensor2_desc,
                    out: tensor3_desc.clone(),
                },
                Box::new(FakeAddOps),
            ));

        optimization.register(Arc::new(ops));
        let ops =
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::<TestBackend, f32>::Add(
                BinaryOpsDescription {
                    lhs: tensor3_desc,
                    rhs: tensor1_desc.clone(),
                    out: tensor4_desc,
                },
                Box::new(FakeAddOps),
            ));
        optimization.register(Arc::new(ops));

        let (kernel, handles, client) = optimization.create_kernel(&mut handles);

        let source = kernel.source().complete();
        println!("{source}");
        pretty_assertions::assert_eq!(source, "");
    }
}
