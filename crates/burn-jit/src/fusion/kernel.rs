use cubecl::calculate_num_elems_dyn_rank;
use cubecl::prelude::*;

use crate::fusion::strides_dyn_rank;
use crate::fusion::JitFusionHandle;
use crate::kernel::Kernel;
use crate::JitRuntime;
use burn_fusion::stream::Context;
use burn_tensor::repr::TensorDescription;
use burn_tensor::repr::TensorStatus;
use cubecl::client::ComputeClient;
use cubecl::server::Binding;
use cubecl::tune::AutotuneOperation;
use std::marker::PhantomData;
use std::sync::Arc;

use super::tracing::ExecutionInfo;

#[derive(new)]
pub struct FusionKernel<R: JitRuntime> {
    id: u64, // Same ID for all different settings.
    info: Arc<KernelExpansion>,
    settings: KernelSettings,
    runtime_info: Vec<OutputRuntimeInfo>,
    cube_count: CubeCount<R::Server>,
    _runtime: PhantomData<R>,
}

pub trait FusionKernelFactory<R: JitRuntime> {
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
pub struct ExecutableKernel<R: JitRuntime> {
    kernel: Box<dyn CubeTask<R::Compiler>>,
    cube_count: CubeCount<R::Server>,
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
pub struct AutotunableKernel<R: JitRuntime> {
    kernel: Arc<dyn CubeTask<R::Compiler>>,
    count: CubeCount<R::Server>,
    bindings: Vec<Binding<R::Server>>,
    client: ComputeClient<R::Server, R::Channel>,
}

// Information related to the output of this kernel.
#[derive(Debug)]
pub enum OutputRuntimeInfo {
    Inplace { input_index: usize },
    Array { size: usize },
}

impl<R: JitRuntime> ExecutableKernel<R> {
    /// Execute the kernel.
    pub fn execute(self) {
        unsafe {
            self.client
                .execute_unchecked(self.kernel, self.cube_count, self.bindings)
        }
    }
}

impl<R: JitRuntime> AutotuneOperation for AutotunableKernel<R> {
    fn execute(self: Box<Self>) {
        self.client
            .execute(Box::new(self.kernel), self.count, self.bindings)
    }

    fn clone(&self) -> Box<dyn AutotuneOperation> {
        Box::new(Self {
            kernel: self.kernel.clone(),
            count: self.count.clone(),
            bindings: self.bindings.clone(),
            client: self.client.clone(),
        })
    }
}

impl<R: JitRuntime> From<ExecutableKernel<R>> for AutotunableKernel<R> {
    fn from(value: ExecutableKernel<R>) -> Self {
        Self {
            kernel: Arc::new(value.kernel),
            count: value.cube_count.clone(),
            bindings: value.bindings,
            client: value.client,
        }
    }
}

impl<R: JitRuntime> FusionKernel<R> {
    pub fn create<K>(
        factory: &K,
        running_info: &ExecutionInfo<'_>,
        context: &mut Context<'_, JitFusionHandle<R>>,
        device: R::Device,
        client: ComputeClient<R::Server, R::Channel>,
        stateful: bool,
    ) -> ExecutableKernel<R>
    where
        K: FusionKernelFactory<R>,
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
        if running_info.scalars.num_f32 > 0 {
            num_handles += 1;
        }
        if running_info.scalars.num_f16 > 0 {
            num_handles += 1;
        }
        if running_info.scalars.num_bf16 > 0 {
            num_handles += 1;
        }
        if running_info.scalars.num_int > 0 {
            num_handles += 1;
        }

        let mut info = Vec::with_capacity(info_size);
        let mut bindings = Vec::with_capacity(num_handles);
        let mut output_register = Vec::with_capacity(outputs_description_updated.len());

        // We register the info and handles for the inputs.
        for (handle, tensor) in handles_input.iter().zip(inputs_description_updated.iter()) {
            register_info_tensor(&mut info, tensor, handle);
            bindings.push(handle.handle.clone().binding());
        }

        // We register the info and handles for the outputs.
        for (tensor, output_info) in outputs_description_updated
            .iter()
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

        // [2, I0stride0, I0stride1, I0shape0, I0shape1i, I1... O0...,  I0len, I1len1, O0len]
        if R::require_array_lengths() {
            for input in inputs_description_updated.iter() {
                let len = calculate_num_elems_dyn_rank(&input.shape);
                info.push(len as u32);
            }

            for output in outputs_description_updated.iter() {
                let len = calculate_num_elems_dyn_rank(&output.shape);
                info.push(len as u32);
            }
        }

        // Create the info buffer.
        bindings.push(client.create(bytemuck::cast_slice(&info)).binding());

        // Finally we finish with the named bindings.
        if running_info.scalars.num_f32 > 0 {
            let bytes = bytemuck::cast_slice(&context.scalar_f32[0..running_info.scalars.num_f32]);
            bindings.push(client.create(bytes).binding());
        }

        if running_info.scalars.num_f16 > 0 {
            let bytes = bytemuck::cast_slice(&context.scalar_f16[0..running_info.scalars.num_f16]);
            bindings.push(client.create(bytes).binding());
        }

        if running_info.scalars.num_bf16 > 0 {
            let bytes =
                bytemuck::cast_slice(&context.scalar_bf16[0..running_info.scalars.num_bf16]);
            bindings.push(client.create(bytes).binding());
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

        let cube_count = fusion_kernel.cube_count.clone();
        ExecutableKernel::new(
            Box::new(KernelTask::<R::Compiler, _>::new(fusion_kernel)),
            cube_count,
            bindings,
            client,
        )
    }
}

impl<R: JitRuntime> Kernel for FusionKernel<R> {
    fn define(&self) -> KernelDefinition {
        log::info!("Compiling ... {:?}", self.id());
        KernelIntegrator::new(self.info.as_ref().clone()).integrate(self.settings.clone())
    }

    fn id(&self) -> cubecl::KernelId {
        cubecl::KernelId::new::<Self>().info((self.settings.clone(), self.id))
    }
}

fn register_info_tensor<R: JitRuntime>(
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

fn process_inputs_outputs<'a, R>(
    inputs: &[&TensorDescription],
    outputs: &[&TensorDescription],
    context: &'a mut Context<'_, JitFusionHandle<R>>,
    stateful: bool,
) -> (
    Vec<JitFusionHandle<R>>,
    Vec<&'a TensorDescription>,
    Vec<&'a TensorDescription>,
)
where
    R: JitRuntime,
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
