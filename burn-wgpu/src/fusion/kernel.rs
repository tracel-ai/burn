use super::codegen::Body;
use crate::compute::{compute_client, DynamicKernel, WgpuComputeClient};
use crate::fusion::codegen::Function;
use crate::fusion::{calculate_num_elems_dyn_rank, strides_dyn_rank};
use crate::fusion::{
    codegen::{
        Binding, ComputeShader, Elem, Location, Operator, Variable, Visibility, WorkgroupSize,
    },
    WgpuFusionHandle,
};
use crate::kernel::{elemwise_workgroup, WORKGROUP_DEFAULT};
use crate::{FloatElement, GraphicsApi, IntElement, Wgpu};
use burn_fusion::{HandleContainer, TensorDescription};
use burn_tensor::Device;
use std::marker::PhantomData;

/// Kernel creation input phase, see [fusion kernel](FusionKernel) for more details.
pub struct InputPhase;
/// Kernel creation body phase, see [fusion kernel](FusionKernel) for more details.
pub struct BodyPhase;
/// Kernel creation output phase, see [fusion kernel](FusionKernel) for more details.
pub struct OutputPhase;
/// Kernel execution phase, see [fusion kernel](FusionKernel) for more details.
pub struct ExecutionPhase;

/// Allows to create custom wgsl kernels based on configured inputs, body and outputs.
///
/// This type has 4 phases that must be executed in order, but no worry the type system won't allow
/// you to make mistakes.
///
///   1. [Input Phase](InputPhase)
///     This phase focuses on registering the input tensor descriptions that are going to be used by
///     the fused kernel.
///   2. [Body Phase](BodyPhase)
///     After the input phase is done, all the operations that happen in the body must be
///     registered.
///   3. [Output Phase](OutputPhase)
///     This step focuses on registering all tensor descriptions that the kernel needs to write to.
///   4. [Execution Phase](ExecutionPhase)
///     Now that all other phases are completed, we can actually run the kernel on the given
///     [handles](HandleContainer). Note that the actual chosen kernel may vary based on the
///     handles provided.
pub struct FusionKernel<G, F, I, Phase = InputPhase>
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
    _phase: PhantomData<Phase>,
}

enum DataBuffer {
    F32(Vec<f32>),
    U32(Vec<u32>),
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> FusionKernel<G, F, I, InputPhase> {
    /// Create a new fusion kernel on the given device.
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
            _phase: PhantomData,
        }
    }

    /// Register the inputs used by the kernel.
    pub fn inputs(
        mut self,
        inputs_tensor: &[&TensorDescription],
        inputs_scalar_f32: &[f32],
    ) -> FusionKernel<G, F, I, BodyPhase> {
        for (i, input) in inputs_tensor.iter().enumerate() {
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
                position_out: inputs_tensor.len(), // First output
            });
        }

        if !inputs_scalar_f32.is_empty() {
            self.named_bindings.push((
                "scalars_f32".to_string(),
                Binding {
                    elem: Elem::F32,
                    visibility: Visibility::Read,
                    location: Location::Storage,
                    size: Some(inputs_scalar_f32.len()),
                },
                DataBuffer::F32(inputs_scalar_f32.to_vec()),
            ));
        }

        FusionKernel {
            operations: self.operations,
            input_bindings: self.input_bindings,
            output_bindings: self.output_bindings,
            named_bindings: self.named_bindings,
            functions: self.functions,
            num_elems_output: self.num_elems_output,
            device: self.device,
            client: self.client,
            _phase: PhantomData,
        }
    }
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> FusionKernel<G, F, I, BodyPhase> {
    /// Register the [operators](Operator) that the kernel must execute in the order provided.
    pub fn body(mut self, operators: &[Operator]) -> FusionKernel<G, F, I, OutputPhase> {
        let mut register_function = |function: Function| {
            if !self.functions.contains(&function) {
                self.functions.push(function);
            }
        };

        // Since not all operators are native to WGSL, we need to add the custom ones.
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

        FusionKernel {
            operations: self.operations,
            input_bindings: self.input_bindings,
            output_bindings: self.output_bindings,
            named_bindings: self.named_bindings,
            functions: self.functions,
            num_elems_output: self.num_elems_output,
            device: self.device,
            client: self.client,
            _phase: PhantomData,
        }
    }
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> FusionKernel<G, F, I, OutputPhase> {
    /// Register the outputs with their local variable index.
    ///
    /// Note that the index corresponds to the registered [operator](Operator) number at the
    /// [body phase](BodyPhase).
    /// So the 4th operator registered creates the local variable 3 (N-1, since the 1th index is 0).
    pub fn outputs(
        mut self,
        outputs: &[&TensorDescription],
        locals: &[u16],
    ) -> FusionKernel<G, F, I, ExecutionPhase> {
        let mut num_elems_launch_option = 0;

        for (i, (output, local)) in outputs.iter().zip(locals).enumerate() {
            let num_elems_output = calculate_num_elems_dyn_rank(&output.shape);
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

            self.operations.push(Operator::AssignGlobal {
                input: Variable::Local(*local),
                out: Variable::Output(i as u16),
            });
        }

        self.num_elems_output = num_elems_launch_option;

        FusionKernel {
            operations: self.operations,
            input_bindings: self.input_bindings,
            output_bindings: self.output_bindings,
            named_bindings: self.named_bindings,
            functions: self.functions,
            num_elems_output: self.num_elems_output,
            device: self.device,
            client: self.client,
            _phase: PhantomData,
        }
    }
}

impl<G: GraphicsApi, F: FloatElement, I: IntElement> FusionKernel<G, F, I, ExecutionPhase> {
    /// Execute the kernel on the provided [handles](HandleContainer).
    pub fn execute(mut self, handle_container: &mut HandleContainer<Wgpu<G, F, I>>) {
        let mut inputs = Vec::with_capacity(self.input_bindings.len());
        let mut outputs = Vec::with_capacity(self.output_bindings.len());
        let mut named = Vec::with_capacity(2);
        let mut info = Vec::new();
        let mut handles =
            Vec::with_capacity(inputs.capacity() + outputs.capacity() + named.capacity());

        // Inner function to fill the info buffer.
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

        // We start by registering the inputs.
        for (binding, tensor) in self.input_bindings.into_iter() {
            let handle = handle_container.get_handle(&tensor);
            register_info_tensor(&tensor, &handle);

            inputs.push(binding);
            handles.push(handle.handle);
        }

        // Then we follow with the outputs.
        for (binding, tensor) in self.output_bindings {
            let num_elems = calculate_num_elems_dyn_rank(&tensor.shape);
            let handle_fusion = WgpuFusionHandle {
                client: self.client.clone(),
                device: self.device.clone(),
                strides: strides_dyn_rank(&tensor.shape),
                handle: self.client.empty(core::mem::size_of::<F>() * num_elems),
            };
            register_info_tensor(&tensor, &handle_fusion);

            handles.push(handle_fusion.handle.clone());
            handle_container.register_handle(tensor.id, handle_fusion);
            outputs.push(binding);
        }

        // Now we can create the info handle.
        Self::build_info_handle(&mut self.named_bindings, info);

        // Finally we finish with the named bindings.
        for (name, binding, data) in self.named_bindings {
            let handle = self.client.create(match &data {
                DataBuffer::F32(values) => bytemuck::cast_slice(values),
                DataBuffer::U32(values) => bytemuck::cast_slice(values),
            });
            named.push((name, binding));
            handles.push(handle);
        }

        // We create the shader codegen type and launch the kernel.
        let kernel = ComputeShader {
            inputs,
            outputs,
            named,
            workgroup_size: WorkgroupSize::default(),
            body: Body::new(self.operations),
            num_workgroups: true,
            global_invocation_id: true,
            functions: self.functions,
        };

        let workgroup = elemwise_workgroup(self.num_elems_output, WORKGROUP_DEFAULT);
        let kernel = Box::new(DynamicKernel::new(kernel, workgroup));

        self.client
            .execute(kernel, &handles.iter().collect::<Vec<_>>());
    }

    fn build_info_handle(named_bindings: &mut Vec<(String, Binding, DataBuffer)>, info: Vec<u32>) {
        named_bindings.push((
            "info".to_string(),
            Binding {
                elem: Elem::U32,
                visibility: Visibility::Read,
                location: Location::Storage,
                size: None, // We avoid putting the length here since it will force a new kernel
                            // for each tensor rank.
            },
            DataBuffer::U32(info),
        ));
    }
}
