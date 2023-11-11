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

use super::{Binding, WgpuFusionHandle, WgslTempate};

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

        let mark_temp = |var: &Variable, list: &mut HashSet<TensorId>| {
            match var {
                Variable::Temp(index) => {
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

            let mut handle = handles.get_handle_float(&input);
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
                variable: Variable::Input(i as u16),
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
                operations.push(Operator::AssignGlobal {
                    input: Variable::Temp(*temp),
                    out: Variable::Output(i as u16),
                });
            }
        }

        let mut num_elems_launch_option = 0;
        for output in outputs {
            let num_elems_output = calculate_num_elems(&output.shape);
            if num_elems_launch_option == 0 {
                num_elems_launch_option = num_elems_output;
            }
            let strides = dyn_strides(&output.shape);
            let handle = client.empty(num_elems_output * core::mem::size_of::<f32>());

            handles.register_handle(
                output.id,
                WgpuFusionHandle::new(
                    client.clone(),
                    handle.clone(),
                    crate::WgpuDevice::BestAvailable,
                    strides.clone(),
                ),
            );
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
        let info = info.into_iter().map(|i| i as u32).collect::<Vec<_>>();
        let info_handle = client.create(bytemuck::cast_slice(&info));
        kernel_handles.push(info_handle);
        let info = Some(Binding {
            elem: super::Elem::U32,
            visibility: super::Visibility::Read,
            location: super::Location::Storage,
            size: Some(info.len()),
        });

        println!("Operations {:?}", operations);

        let kernel = WgslTempate {
            inputs: input_bindings,
            outputs: output_bindings,
            info,
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

    fn input_to_var(&mut self, tensor: &TensorDescription) -> Variable {
        let variable = match self.tensors.contains_key(&tensor.id) {
            false => {
                let var = Variable::Input(self.inputs.len() as u16);
                self.inputs.push(tensor.clone());
                var
            }
            true => match self.temps.get(&tensor.id) {
                Some(index) => Variable::Temp(*index),
                None => {
                    let input = self
                        .inputs
                        .iter()
                        .enumerate()
                        .find(|(_, input)| input.id == tensor.id)
                        .unwrap();
                    Variable::Input(input.0 as u16)
                }
            },
        };
        self.tensors.insert(tensor.id.clone(), tensor.clone());

        variable
    }

    fn output_to_var(&mut self, tensor: &TensorDescription) -> Variable {
        let temp = self.ops.len() as u16;
        self.temps.insert(tensor.id.clone(), temp);
        self.tensors.insert(tensor.id.clone(), tensor.clone());
        Variable::Temp(temp)
    }

    fn register_numeric<B: FusionBackend, E: Element>(
        &mut self,
        ops: &NumericOpsDescription<B, E>,
    ) -> bool {
        match ops {
            NumericOpsDescription::Add(desc, _) => {
                let lhs = self.input_to_var(&desc.lhs);
                let rhs = self.input_to_var(&desc.rhs);
                let out = self.output_to_var(&desc.out);

                self.ops.push(Operator::Add { lhs, rhs, out });

                return true;
            }
            NumericOpsDescription::Sub(desc, _) => {
                let lhs = self.input_to_var(&desc.lhs);
                let rhs = self.input_to_var(&desc.rhs);
                let out = self.output_to_var(&desc.out);

                self.ops.push(Operator::Sub { lhs, rhs, out });

                return true;
            }
            _ => false,
        }
    }
}

#[derive(Debug, Hash, Clone)]
enum Variable {
    Input(u16),
    Temp(u16),
    Output(u16),
}

#[derive(Debug, Hash, Clone)]
enum Operator {
    Add {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    Sub {
        lhs: Variable,
        rhs: Variable,
        out: Variable,
    },
    AssignGlobal {
        input: Variable,
        out: Variable,
    },
    ReadGlobal {
        variable: Variable,
        position: usize,
        position_out: usize,
    },
}

impl Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::Input(number) => f.write_fmt(format_args!("input_{number}")),
            Variable::Temp(number) => f.write_fmt(format_args!("temp_{number}")),
            Variable::Output(number) => f.write_fmt(format_args!("output_{number}")),
        }
    }
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Add { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} + {rhs};"))
            }
            Operator::Sub { lhs, rhs, out } => {
                f.write_fmt(format_args!("let {out} = {lhs} - {rhs};"))
            }
            Operator::AssignGlobal { input, out } => {
                f.write_fmt(format_args!("{out}_global[id] = {input};"))
            }
            Operator::ReadGlobal {
                variable,
                position,
                position_out,
            } => {
                let (global, local) = match variable {
                    Variable::Input(number) => {
                        (format!("input_{number}_global"), format!("input_{number}"))
                    }
                    Variable::Temp(_) => panic!("can't ready global a temp variable."),
                    Variable::Output(number) => (
                        format!("output_{number}_global"),
                        format!("output_{number}"),
                    ),
                };

                f.write_fmt(format_args!(
                    "
var index_{local}: u32 = 0u;

for (var i: u32 = 1u; i <= dim; i++) {{
    let position = {position}u * (2u * dim);
    let position_out = {position_out}u * (2u * dim);

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
            f.write_fmt(format_args!("{ops}"))?;
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
    fn test_fusion_two_elems() {
        type Backend = Wgpu;
        type FusedBackend = Fusion<Wgpu>;

        let data_1 =
            Tensor::<Backend, 2>::random([32, 32], burn_tensor::Distribution::Default).into_data();
        let data_2 =
            Tensor::<Backend, 2>::random([1, 32], burn_tensor::Distribution::Default).into_data();

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
        panic!("Haha");
    }
}
