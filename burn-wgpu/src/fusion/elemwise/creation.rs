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

#[derive(Default)]
pub struct ElemWiseKernelCreation<G, F, I> {
    operations: Vec<Operator>,
    info: Vec<usize>,
    handles: Vec<WgpuHandle>,
    _g: PhantomData<G>,
    _f: PhantomData<F>,
    _i: PhantomData<I>,
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> ElemWiseKernelCreation<G, F, I> {
    pub fn create_kernel(
        mut self,
        ops: &FloatElementWiseFusionOps,
        handles: &mut HandleContainer<Wgpu<G, F, I>>,
    ) -> (Box<dyn Kernel>, Vec<WgpuHandle>, WgpuComputeClient) {
        let (inputs, scalars, client) = self.register_inputs(ops, handles);
        let (outputs, functions) = self.register_outputs(ops, &client, handles);

        let mut named = Vec::new();
        // Scalar bindings should come after the output.
        if let Some((binding, handle)) = scalars {
            named.push(("scalars_f32".to_string(), binding));
            self.handles.push(handle);
        }

        // Info binding should come after the scalars.
        let (binding, hadnle) = self.kernel_creation_info(&client);
        named.push(("info".to_string(), binding));
        self.handles.push(hadnle);

        let kernel = ShaderCodegen {
            inputs,
            outputs,
            named,
            workgroup_sizes: WorkgroupSize::default(),
            body: Box::new(ElemWiseBody::new(self.operations)),
            num_workgroups: true,
            global_invocation_id: true,
            functions,
        };

        let workgroup = elemwise_workgroup(ops.num_elems_output, WORKGROUP_DEFAULT);

        (
            Box::new(DynamicKernel::new(kernel, workgroup)),
            self.handles,
            client,
        )
    }

    fn register_inputs(
        &mut self,
        ops: &FloatElementWiseFusionOps,
        handles: &mut HandleContainer<Wgpu<G, F, I>>,
    ) -> (
        Vec<Binding>,
        Option<(Binding, WgpuHandle)>,
        WgpuComputeClient,
    ) {
        let mut client = None;
        let mut bindings = Vec::new();

        for (i, input) in ops.inputs.iter().enumerate() {
            if self.info.is_empty() {
                self.info.push(input.shape.len()); // Rank
            }

            let mut handle = handles.get_handle_float(&input);
            self.info.append(&mut handle.strides);
            self.info.append(&mut input.shape.clone());
            self.handles.push(handle.handle);

            if let None = client {
                client = Some(handle.client.clone());
            }

            bindings.push(Binding {
                elem: Elem::F32,
                visibility: Visibility::Read,
                location: Location::Storage,
                size: None,
            });
            self.operations.push(Operator::ReadGlobal {
                variable: Variable::Input(i as u16),
                position: i,
                position_out: ops.inputs.len(), // First output
            });
        }

        let client = client.unwrap();
        let mut binding_scalar = None;

        if !ops.scalars_f32.is_empty() {
            let scalar_handle = client.create(bytemuck::cast_slice(&ops.scalars_f32));
            binding_scalar = Some((
                Binding {
                    elem: Elem::F32,
                    visibility: Visibility::Read,
                    location: Location::Storage,
                    size: Some(ops.scalars_f32.len()),
                },
                scalar_handle,
            ));
        }

        (bindings, binding_scalar, client)
    }

    fn register_outputs(
        &mut self,
        ops: &FloatElementWiseFusionOps,
        client: &WgpuComputeClient,
        handles: &mut HandleContainer<Wgpu<G, F, I>>,
    ) -> (Vec<Binding>, Vec<Function>) {
        let outputs = ops.output_descriptions();
        let mut bindings = Vec::with_capacity(outputs.len());
        let mut functions = Vec::new();
        let mut seen = HashSet::new();

        let mut register_function = |function: Function| {
            if !seen.contains(&function) {
                functions.push(function.clone());
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

        for (i, output) in outputs.iter().enumerate() {
            if let Some(temp) = ops.locals.get(&output.id) {
                self.operations.push(Operator::AssignGlobal {
                    input: Variable::Local(*temp),
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
                output.id.clone(),
                WgpuFusionHandle::new(
                    client.clone(),
                    handle.clone(),
                    crate::WgpuDevice::BestAvailable,
                    strides.clone(),
                ),
            );
            let mut strides = dyn_strides(&output.shape);

            self.info.append(&mut strides);
            self.info.append(&mut output.shape.clone());
            self.handles.push(handle);

            bindings.push(Binding {
                elem: Elem::F32,
                visibility: Visibility::ReadWrite,
                location: Location::Storage,
                size: None,
            });
        }

        (bindings, functions)
    }

    fn kernel_creation_info(&mut self, client: &WgpuComputeClient) -> (Binding, WgpuHandle) {
        let info = self.info.iter().map(|i| *i as u32).collect::<Vec<_>>();
        let info_handle = client.create(bytemuck::cast_slice(&info));

        (
            Binding {
                elem: Elem::U32,
                visibility: Visibility::Read,
                location: Location::Storage,
                size: Some(info.len()),
            },
            info_handle,
        )
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
