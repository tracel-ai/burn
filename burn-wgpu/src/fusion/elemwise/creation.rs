use crate::compute::{compute_client, DynamicKernel, Kernel, WgpuComputeClient, WgpuHandle};
use crate::fusion::codegen::Function;
use crate::fusion::{calculate_num_elems, dyn_strides};
use crate::fusion::{
    codegen::{
        Binding, Elem, Location, Operator, ShaderCodegen, Variable, Visibility, WorkgroupSize,
    },
    WgpuFusionHandle,
};
use crate::kernel::{elemwise_workgroup, DynamicKernelSource, SourceTemplate, WORKGROUP_DEFAULT};
use crate::{FloatElement, GraphicsApi, IntElement, Wgpu};
use burn_fusion::{HandleContainer, TensorDescription, TensorId};
use burn_tensor::Device;
use hashbrown::{HashMap, HashSet};
use std::collections::hash_map::DefaultHasher;
use std::fmt::Display;
use std::hash::{Hash, Hasher};

pub struct KernelBuilder<G, F, I>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    operations: Vec<Operator>,
    input_bindings: Vec<(Binding, TensorDescription)>,
    output_bindings: Vec<(Binding, TensorDescription)>,
    named_bindings: Vec<(String, Binding, DataBuffer)>,
    functions: Vec<Function>,
    num_elems_output: usize,
    device: Device<Wgpu<G, F, I>>,
    client: WgpuComputeClient,
}

enum DataBuffer {
    F32(Vec<f32>),
    U32(Vec<u32>),
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> KernelBuilder<G, F, I> {
    pub fn new(device: &Device<Wgpu<G, F, I>>) -> Self {
        let client = compute_client::<G>(device);

        Self {
            operations: Vec::new(),
            input_bindings: Vec::new(),
            output_bindings: Vec::new(),
            named_bindings: Vec::new(),
            functions: Vec::new(),
            num_elems_output: 0,
            device: device.clone(),
            client,
        }
    }

    pub fn build(
        mut self,
        handles_fusion: &mut HandleContainer<Wgpu<G, F, I>>,
    ) -> (Box<dyn Kernel>, Vec<WgpuHandle>, WgpuComputeClient) {
        let mut inputs = Vec::with_capacity(self.input_bindings.len());
        let mut outputs = Vec::with_capacity(self.output_bindings.len());
        let mut named = Vec::with_capacity(2);
        let mut info = Vec::new();
        let mut handles =
            Vec::with_capacity(inputs.capacity() + outputs.capacity() + named.capacity());

        let mut register_info_tensor = |tensor: &TensorDescription, handle: &WgpuFusionHandle| {
            if info.is_empty() {
                info.push(handle.strides.len() as u32);
            }

            for s in handle.strides.iter() {
                info.push(*s as u32);
            }
            for s in tensor.shape.iter() {
                info.push(*s as u32);
            }
        };
        for (binding, tensor) in self.input_bindings.into_iter() {
            let handle = handles_fusion.get_handle(&tensor);
            register_info_tensor(&tensor, &handle);

            inputs.push(binding);
            handles.push(handle.handle);
        }

        for (binding, tensor) in self.output_bindings {
            let num_elems = calculate_num_elems(&tensor.shape);
            let handle_fusion = WgpuFusionHandle {
                client: self.client.clone(),
                device: self.device.clone(),
                strides: dyn_strides(&tensor.shape),
                handle: self.client.empty(core::mem::size_of::<F>() * num_elems),
            };
            register_info_tensor(&tensor, &handle_fusion);

            handles.push(handle_fusion.handle.clone());
            handles_fusion.register_handle(tensor.id, handle_fusion);
            outputs.push(binding);
        }

        Self::build_info_handle(&mut self.named_bindings, info);

        for (name, binding, data) in self.named_bindings {
            let handle = self.client.create(match &data {
                DataBuffer::F32(values) => bytemuck::cast_slice(values),
                DataBuffer::U32(values) => bytemuck::cast_slice(values),
            });
            named.push((name, binding));
            handles.push(handle);
        }

        let kernel = ShaderCodegen {
            inputs,
            outputs,
            named,
            workgroup_sizes: WorkgroupSize::default(),
            body: Box::new(ElemWiseBody::new(self.operations)),
            num_workgroups: true,
            global_invocation_id: true,
            functions: self.functions,
        };

        let workgroup = elemwise_workgroup(self.num_elems_output, WORKGROUP_DEFAULT);

        (
            Box::new(DynamicKernel::new(kernel, workgroup)),
            handles,
            self.client,
        )
    }

    pub fn inputs(mut self, inputs: &[&TensorDescription], scalars_f32: &[f32]) -> Self {
        for (i, input) in inputs.iter().enumerate() {
            self.input_bindings.push((
                Binding {
                    elem: Elem::F32,
                    visibility: Visibility::Read,
                    location: Location::Storage,
                    size: None,
                },
                (*input).clone(),
            ));

            self.operations.push(Operator::ReadGlobal {
                variable: Variable::Input(i as u16),
                position: i,
                position_out: inputs.len(), // First output
            });
        }

        if !scalars_f32.is_empty() {
            self.named_bindings.push((
                "scalars_f32".to_string(),
                Binding {
                    elem: Elem::F32,
                    visibility: Visibility::Read,
                    location: Location::Storage,
                    size: Some(scalars_f32.len()),
                },
                DataBuffer::F32(scalars_f32.to_vec()),
            ));
        }

        self
    }

    pub fn body(mut self, operators: &[Operator]) -> Self {
        let mut seen = HashSet::new();
        let mut register_function = |function: Function| {
            if !seen.contains(&function) {
                self.functions.push(function.clone());
                seen.insert(function);
            }
        };

        for ops in operators.iter() {
            match ops {
                Operator::Powf {
                    lhs: _,
                    rhs: _,
                    out: _,
                } => {
                    register_function(Function::Powf(Elem::F32));
                }
                Operator::Erf { input: _, out: _ } => {
                    register_function(Function::Erf(Elem::F32));
                }
                _ => {}
            }
            self.operations.push(ops.clone());
        }

        self
    }

    pub fn outputs(
        mut self,
        outputs: &[&TensorDescription],
        locals: &HashMap<TensorId, u16>,
    ) -> Self {
        let mut num_elems_launch_option = 0;

        for (i, output) in outputs.into_iter().enumerate() {
            let num_elems_output = calculate_num_elems(&output.shape);
            if num_elems_output > num_elems_launch_option {
                num_elems_launch_option = num_elems_output;
            }

            self.output_bindings.push((
                Binding {
                    elem: Elem::F32,
                    visibility: Visibility::ReadWrite,
                    location: Location::Storage,
                    size: None,
                },
                (*output).clone(),
            ));

            let variable = Variable::Output(i as u16);

            self.num_elems_output = num_elems_launch_option;

            if let Some(local_index) = locals.get(&output.id) {
                self.operations.push(Operator::AssignGlobal {
                    input: Variable::Local(*local_index),
                    out: variable,
                });
            }
        }

        self
    }

    fn build_info_handle(named_bindings: &mut Vec<(String, Binding, DataBuffer)>, info: Vec<u32>) {
        named_bindings.push((
            "info".to_string(),
            Binding {
                elem: Elem::U32,
                visibility: Visibility::Read,
                location: Location::Storage,
                size: None, // We avoid putting the lenght here since it will force a new kernel
                            // for each tensor rank.
            },
            DataBuffer::U32(info),
        ));
    }
}

#[derive(Hash, new)]
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

        s.finish().to_string() + "-fused"
    }
}
