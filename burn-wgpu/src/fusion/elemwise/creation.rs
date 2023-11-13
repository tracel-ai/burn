use super::FloatElementWiseFusionOps;
use crate::compute::{DynamicKernel, Kernel, WgpuComputeClient, WgpuHandle};
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
use burn_fusion::HandleContainer;
use hashbrown::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

pub struct ElemWiseKernelCreation<G, F, I> {
    operations: Vec<Operator>,
    info: Vec<usize>,
    input_bindings: Vec<(Binding, WgpuHandle)>,
    output_bindings: Vec<(Binding, WgpuHandle)>,
    named_bindings: Vec<(String, Binding, WgpuHandle)>,
    functions: Vec<Function>,
    num_elems_output: usize,
    client: Option<WgpuComputeClient>,
    _g: PhantomData<G>,
    _f: PhantomData<F>,
    _i: PhantomData<I>,
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> ElemWiseKernelCreation<G, F, I> {
    pub fn new(
        ops: &FloatElementWiseFusionOps,
        handles: &mut HandleContainer<Wgpu<G, F, I>>,
    ) -> Self {
        let mut instance = Self {
            operations: Vec::new(),
            info: Vec::new(),
            input_bindings: Vec::new(),
            output_bindings: Vec::new(),
            named_bindings: Vec::new(),
            functions: Vec::new(),
            num_elems_output: 0,
            client: None,
            _g: PhantomData,
            _f: PhantomData,
            _i: PhantomData,
        };
        instance.register_inputs(ops, handles);
        instance.register_body(ops);
        instance.register_outputs(ops, handles);
        instance.kernel_creation_info();

        instance
    }

    pub fn build(self) -> (Box<dyn Kernel>, Vec<WgpuHandle>, WgpuComputeClient) {
        let mut inputs = Vec::with_capacity(self.input_bindings.len());
        let mut outputs = Vec::with_capacity(self.output_bindings.len());
        let mut named = Vec::with_capacity(2);
        let mut handles =
            Vec::with_capacity(inputs.capacity() + outputs.capacity() + named.capacity());
        let client = self.client();

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
            client,
        )
    }

    fn register_inputs(
        &mut self,
        ops: &FloatElementWiseFusionOps,
        handles: &mut HandleContainer<Wgpu<G, F, I>>,
    ) {
        let mut client = None;

        for (i, input) in ops.inputs.iter().enumerate() {
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

            if let None = client {
                client = Some(handle.client.clone());
            }

            self.operations.push(Operator::ReadGlobal {
                variable: Variable::Input(i as u16),
                position: i,
                position_out: ops.inputs.len(), // First output
            });
        }

        let client = client.unwrap();

        if !ops.scalars_f32.is_empty() {
            let scalar_handle = client.create(bytemuck::cast_slice(&ops.scalars_f32));
            self.named_bindings.push((
                "scalars_f32".to_string(),
                Binding {
                    elem: Elem::F32,
                    visibility: Visibility::Read,
                    location: Location::Storage,
                    size: Some(ops.scalars_f32.len()),
                },
                scalar_handle,
            ));
        }

        self.client = Some(client);
    }

    fn client(&self) -> WgpuComputeClient {
        match &self.client {
            Some(value) => value.clone(),
            None => panic!("No client registered yet."),
        }
    }

    fn register_body(&mut self, ops: &FloatElementWiseFusionOps) {
        let mut seen = HashSet::new();
        let mut register_function = |function: Function| {
            if !seen.contains(&function) {
                self.functions.push(function.clone());
                seen.insert(function);
            }
        };

        for ops in ops.operators.iter() {
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
    }

    fn register_outputs(
        &mut self,
        ops: &FloatElementWiseFusionOps,
        handles: &mut HandleContainer<Wgpu<G, F, I>>,
    ) {
        let client = self.client();
        self.num_elems_output = ops.num_elems_output;
        let outputs = ops.output_descriptions();
        let mut inputs_reused_as_outputs = Vec::new();

        ops.inputs.iter().enumerate().for_each(|(i, tensor)| {
            let updated_tensor = ops.tensors.get(&tensor.id).unwrap();

            match updated_tensor.status {
                burn_fusion::TensorStatus::ReadWrite => {
                    inputs_reused_as_outputs.push((false, i, updated_tensor));
                }
                _ => {}
            }
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
                        client.clone(),
                        handle.clone(),
                        crate::WgpuDevice::BestAvailable,
                        strides.clone(),
                    );
                    let variable = Variable::Input(*index as u16);
                    (handle, variable)
                }
                None => {
                    let handle = client.empty(num_elems_output * core::mem::size_of::<f32>());
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
                        client.clone(),
                        handle.clone(),
                        crate::WgpuDevice::BestAvailable,
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

            if let Some(local_index) = ops.locals.get(&output.id) {
                self.operations.push(Operator::AssignGlobal {
                    input: Variable::Local(*local_index),
                    out: var,
                });
            }
        }
    }

    fn kernel_creation_info(&mut self) {
        let client = self.client();
        let info = self.info.iter().map(|i| *i as u32).collect::<Vec<_>>();
        let info_handle = client.create(bytemuck::cast_slice(&info));

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
