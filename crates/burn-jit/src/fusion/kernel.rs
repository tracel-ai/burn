use crate::codegen::Compilation;
use crate::codegen::CompilationInfo;
use crate::codegen::CompilationSettings;
use crate::compute::FullCompilationPhase;
use crate::compute::JitKernel;
use crate::compute::Kernel;
use crate::compute::WorkGroup;
use crate::fusion::strides_dyn_rank;
use crate::fusion::JitFusionHandle;
use crate::gpu::ComputeShader;
use crate::kernel::GpuComputeShaderPhase;
use crate::FloatElement;
use crate::IntElement;
use crate::JitBackend;
use crate::Runtime;
use burn_compute::client::ComputeClient;
use burn_compute::server::Binding;
use burn_compute::tune::AutotuneOperation;
use burn_fusion::stream::Context;
use burn_tensor::repr::TensorDescription;
use burn_tensor::repr::TensorStatus;
use burn_tensor::Device;
use std::marker::PhantomData;
use std::sync::Arc;

use super::tracing::ExecutionInfo;

#[derive(new)]
pub struct FusionKernel<R: Runtime> {
    id: String, // Same ID for all different settings.
    info: Arc<CompilationInfo>,
    settings: CompilationSettings,
    runtime_info: Vec<OutputRuntimeInfo>,
    workgroup: WorkGroup,
    _runtime: PhantomData<R>,
}

pub trait FusionKernelFactory<R: Runtime> {
    /// Create a new kernel.
    fn create(
        &self,
        handles_inputs: &[JitFusionHandle<R>],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
        stateful: bool, // Should be set to false when running autotune.
    ) -> FusionKernel<R>;
}

/// An instantiation of a [kernel](Kernel) that can be executed.
#[derive(new)]
pub struct ExecutableKernel<R: Runtime> {
    kernel: Box<dyn JitKernel>,
    bindings: Vec<Binding<R::Server>>,
    client: ComputeClient<R::Server, R::Channel>,
}

/// An instantiation of a [kernel](Kernel) that can be autotuned.
///
/// The main difference with an [executable kernel](ExecutableKernel) is that this kernel can be
/// cloned and executed multiple times to properly collect benchmarks.
///
/// The clone function used is defined in the trait [AutotuneOperation] instead of [Clone].
#[derive(new)]
pub struct AutotunableKernel<R: Runtime> {
    kernel: Arc<dyn JitKernel>,
    bindings: Vec<Binding<R::Server>>,
    client: ComputeClient<R::Server, R::Channel>,
}

// Information related to the output of this kernel.
#[derive(Debug)]
pub enum OutputRuntimeInfo {
    Inplace { input_index: usize },
    Array { size: usize },
}

impl<R: Runtime> ExecutableKernel<R> {
    /// Execute the kernel.
    pub fn execute(self) {
        self.client
            .execute(Kernel::JitGpu(self.kernel), self.bindings)
    }
}

impl<R: Runtime> AutotuneOperation for AutotunableKernel<R> {
    fn execute(self: Box<Self>) {
        self.client
            .execute(Kernel::JitGpu(Box::new(self.kernel)), self.bindings)
    }

    fn clone(&self) -> Box<dyn AutotuneOperation> {
        Box::new(Self {
            kernel: self.kernel.clone(),
            bindings: self.bindings.clone(),
            client: self.client.clone(),
        })
    }
}

impl<R: Runtime> From<ExecutableKernel<R>> for AutotunableKernel<R> {
    fn from(value: ExecutableKernel<R>) -> Self {
        Self {
            kernel: Arc::new(value.kernel),
            bindings: value.bindings,
            client: value.client,
        }
    }
}

impl<R: Runtime> FusionKernel<R> {
    pub fn create<K, F, I>(
        factory: &K,
        running_info: &ExecutionInfo<'_>,
        context: &mut Context<'_, JitBackend<R, F, I>>,
        device: Device<JitBackend<R, F, I>>,
        client: ComputeClient<R::Server, R::Channel>,
        stateful: bool,
    ) -> ExecutableKernel<R>
    where
        K: FusionKernelFactory<R>,
        F: FloatElement,
        I: IntElement,
    {
        let (handles_input, inputs_description_updated, outputs_description_updated) =
            process_inputs_outputs(
                &running_info.inputs,
                &running_info.outputs,
                context,
                stateful,
            );

        let fusion_kernel = factory.create(
            &handles_input,
            &inputs_description_updated,
            &outputs_description_updated,
            stateful,
        );

        let rank_input = running_info
            .inputs
            .first()
            .map(|desc| desc.shape.len())
            .unwrap_or(1);
        let rank_output = running_info
            .outputs
            .first()
            .map(|desc| desc.shape.len())
            .unwrap_or(1);
        let rank = usize::max(rank_input, rank_output);

        let num_tensors = running_info.inputs.len() + running_info.outputs.len();
        // The buffer starts with the rank, then each tensor shape and stride.
        let info_size = (num_tensors * rank * 2) + 1;

        let mut num_handles = num_tensors + 1;
        if running_info.scalars.num_float > 0 {
            num_handles += 1;
        }
        if running_info.scalars.num_int > 0 {
            num_handles += 1;
        }

        let mut info = Vec::with_capacity(info_size);
        let mut bindings = Vec::with_capacity(num_handles);
        let mut output_register = Vec::with_capacity(outputs_description_updated.len());

        // We register the info and handles for the inputs.
        for (handle, tensor) in handles_input.iter().zip(inputs_description_updated) {
            register_info_tensor(&mut info, tensor, handle);
            bindings.push(handle.handle.clone().binding());
        }

        // We register the info and handles for the outputs.
        for (tensor, output_info) in outputs_description_updated
            .into_iter()
            .zip(fusion_kernel.runtime_info.iter())
        {
            match output_info {
                // Use the input inplace for this output.
                OutputRuntimeInfo::Inplace { input_index } => {
                    let input = handles_input.get(*input_index).unwrap();

                    let handle_fusion = JitFusionHandle {
                        client: client.clone(),
                        device: device.clone(),
                        strides: strides_dyn_rank(&tensor.shape),
                        handle: input.handle.clone(),
                    };
                    output_register.push((tensor.id, handle_fusion));
                }
                // Create a new buffer for this output.
                OutputRuntimeInfo::Array { size } => {
                    let handle_fusion = JitFusionHandle {
                        client: client.clone(),
                        device: device.clone(),
                        strides: strides_dyn_rank(&tensor.shape),
                        handle: client.empty(*size),
                    };

                    register_info_tensor(&mut info, tensor, &handle_fusion);
                    bindings.push(handle_fusion.handle.clone().binding());
                    output_register.push((tensor.id, handle_fusion));
                }
            };
        }

        // Create the info buffer.
        bindings.push(client.create(bytemuck::cast_slice(&info)).binding());

        // Finally we finish with the named bindings.
        if running_info.scalars.num_float > 0 {
            bindings.push(
                client
                    .create(bytemuck::cast_slice(
                        &context.scalar_floats[0..running_info.scalars.num_float],
                    ))
                    .binding(),
            );
        }

        if running_info.scalars.num_int > 0 {
            bindings.push(
                client
                    .create(bytemuck::cast_slice(
                        &context.scalar_ints[0..running_info.scalars.num_int],
                    ))
                    .binding(),
            );
        }

        // We have to register the output handles to the context.
        for (id, handle) in output_register {
            context.handles.register_handle(id, handle);
        }

        let workgroup = fusion_kernel.workgroup.clone();
        ExecutableKernel::new(
            Box::new(FullCompilationPhase::<R::Compiler, FusionKernel<R>>::new(
                fusion_kernel,
                workgroup,
            )),
            bindings,
            client,
        )
    }
}

impl<R: Runtime> GpuComputeShaderPhase for FusionKernel<R> {
    fn compile(&self) -> ComputeShader {
        log::info!("Compiling ... {:?}", self.id());
        Compilation::new(self.info.as_ref().clone()).compile(self.settings.clone())
    }

    fn id(&self) -> String {
        format!("{}", self.settings) + self.id.as_str()
    }
}

fn register_info_tensor<R: Runtime>(
    info: &mut Vec<u32>,
    tensor: &TensorDescription,
    handle: &JitFusionHandle<R>,
) {
    if info.is_empty() {
        info.push(handle.strides.len() as u32);
    }

    for s in handle.strides.iter() {
        info.push(*s as u32);
    }
    for s in tensor.shape.iter() {
        info.push(*s as u32);
    }
}

fn process_inputs_outputs<'a, R, F, I>(
    inputs: &[&TensorDescription],
    outputs: &[&TensorDescription],
    context: &'a mut Context<'_, JitBackend<R, F, I>>,
    stateful: bool,
) -> (
    Vec<JitFusionHandle<R>>,
    Vec<&'a TensorDescription>,
    Vec<&'a TensorDescription>,
)
where
    R: Runtime,
    F: FloatElement,
    I: IntElement,
{
    let mut inputs_description_updated = Vec::with_capacity(inputs.len());
    let mut outputs_description_updated = Vec::with_capacity(outputs.len());
    let mut handles_input = Vec::new();

    for tensor in inputs.iter() {
        let status = if stateful {
            &tensor.status // Important to take the status of the relative graph and not
                           // the global graph, since the status of the global graph
                           // might be of a later operation on the same tensor id.
        } else {
            &TensorStatus::ReadOnly
        };

        let tensor = context.tensors.get(&tensor.id).unwrap();
        let handle = context.handles.get_handle(&tensor.id, status);

        handles_input.push(handle);
        inputs_description_updated.push(tensor);
    }

    for tensor in outputs.iter() {
        let tensor = context.tensors.get(&tensor.id).unwrap();
        outputs_description_updated.push(tensor);
    }

    (
        handles_input,
        inputs_description_updated,
        outputs_description_updated,
    )
}
