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

pub struct ElemWiseKernelCreation<G, F, I>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    operations: Vec<Operator>,
    info: Vec<usize>,
    input_bindings: Vec<(Binding, WgpuHandle)>,
    output_bindings: Vec<(Binding, WgpuHandle)>,
    named_bindings: Vec<(String, Binding, WgpuHandle)>,
    functions: Vec<Function>,
    num_elems_output: usize,
    device: Device<Wgpu<G, F, I>>,
    client: WgpuComputeClient,
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> ElemWiseKernelCreation<G, F, I> {
    pub fn new(device: &Device<Wgpu<G, F, I>>) -> Self {
        let client = compute_client::<G>(device);

        Self {
            operations: Vec::new(),
            info: Vec::new(),
            input_bindings: Vec::new(),
            output_bindings: Vec::new(),
            named_bindings: Vec::new(),
            functions: Vec::new(),
            num_elems_output: 0,
            device: device.clone(),
            client,
        }
    }

    pub fn build(mut self) -> (Box<dyn Kernel>, Vec<WgpuHandle>, WgpuComputeClient) {
        self.build_info_handle();

        let mut inputs = Vec::with_capacity(self.input_bindings.len());
        let mut outputs = Vec::with_capacity(self.output_bindings.len());
        let mut named = Vec::with_capacity(2);
        let mut handles =
            Vec::with_capacity(inputs.capacity() + outputs.capacity() + named.capacity());

        for (binding, handle) in self.input_bindings {
            inputs.push(binding);
            handles.push(handle);
        }
        for (binding, handle) in self.output_bindings {
            outputs.push(binding);
            handles.push(handle);
        }
        for (name, binding, handle) in self.named_bindings {
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

    pub fn inputs(
        mut self,
        handles: &mut HandleContainer<Wgpu<G, F, I>>,
        inputs: &[&TensorDescription],
        scalars_f32: &[f32],
    ) -> Self {
        for (i, input) in inputs.iter().enumerate() {
            if self.info.is_empty() {
                self.info.push(input.shape.len()); // Rank
            }

            let mut handle = handles.get_handle_float(&input);
            self.info.append(&mut handle.strides);
            self.info.append(&mut input.shape.clone());
            self.input_bindings.push((
                Binding {
                    elem: Elem::F32,
                    visibility: Visibility::Read,
                    location: Location::Storage,
                    size: None,
                },
                handle.handle,
            ));

            self.operations.push(Operator::ReadGlobal {
                variable: Variable::Input(i as u16),
                position: i,
                position_out: inputs.len(), // First output
            });
        }

        if !scalars_f32.is_empty() {
            let scalar_handle = self.client.create(bytemuck::cast_slice(&scalars_f32));
            self.named_bindings.push((
                "scalars_f32".to_string(),
                Binding {
                    elem: Elem::F32,
                    visibility: Visibility::Read,
                    location: Location::Storage,
                    size: Some(scalars_f32.len()),
                },
                scalar_handle,
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
        handles: &mut HandleContainer<Wgpu<G, F, I>>,
        outputs: &[&TensorDescription],
        inputs: &[&TensorDescription],
        locals: &HashMap<TensorId, u16>,
    ) -> Self {
        let mut inputs_reused_as_outputs = Vec::new();

        inputs
            .iter()
            .enumerate()
            .for_each(|(i, tensor)| match tensor.status {
                burn_fusion::TensorStatus::ReadWrite => {
                    inputs_reused_as_outputs.push((false, i, tensor));
                }
                _ => {}
            });

        let mut num_elems_launch_option = 0;
        let mut output_number = 0;

        for output in outputs {
            let num_elems_output = calculate_num_elems(&output.shape);
            if num_elems_launch_option == 0 {
                num_elems_launch_option = num_elems_output;
            }
            let strides = dyn_strides(&output.shape);

            let (handle, var) = match inputs_reused_as_outputs
                .iter_mut()
                .find(|(taken, _index, input)| !taken && input.shape == output.shape)
            {
                Some((taken, index, _input)) => {
                    // Inplace
                    *taken = true;
                    let (binding, handle) = self.input_bindings.get_mut(*index).unwrap();
                    binding.visibility = Visibility::ReadWrite;
                    let handle = WgpuFusionHandle::new(
                        self.client.clone(),
                        handle.clone(),
                        self.device.clone(),
                        strides.clone(),
                    );
                    let variable = Variable::Input(*index as u16);
                    (handle, variable)
                }
                None => {
                    let handle = self
                        .client
                        .empty(num_elems_output * core::mem::size_of::<f32>());
                    self.output_bindings.push((
                        Binding {
                            elem: Elem::F32,
                            visibility: Visibility::ReadWrite,
                            location: Location::Storage,
                            size: None,
                        },
                        handle.clone(),
                    ));

                    let handle = WgpuFusionHandle::new(
                        self.client.clone(),
                        handle.clone(),
                        self.device.clone(),
                        strides.clone(),
                    );
                    let variable = Variable::Output(output_number);
                    output_number += 1;
                    (handle, variable)
                }
            };

            handles.register_handle(output.id.clone(), handle);
            let mut strides = dyn_strides(&output.shape);

            self.info.append(&mut strides);
            self.info.append(&mut output.shape.clone());
            self.num_elems_output = num_elems_launch_option;

            if let Some(local_index) = locals.get(&output.id) {
                self.operations.push(Operator::AssignGlobal {
                    input: Variable::Local(*local_index),
                    out: var,
                });
            }
        }

        self
    }

    fn build_info_handle(&mut self) {
        let info = self.info.iter().map(|i| *i as u32).collect::<Vec<_>>();
        let info_handle = self.client.create(bytemuck::cast_slice(&info));

        self.named_bindings.push((
            "info".to_string(),
            Binding {
                elem: Elem::U32,
                visibility: Visibility::Read,
                location: Location::Storage,
                size: Some(info.len()),
            },
            info_handle,
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

        s.finish().to_string()
    }
}
