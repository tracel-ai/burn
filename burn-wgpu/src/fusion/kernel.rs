use crate::compute::{Kernel, WgpuComputeClient, WgpuHandle};
use crate::fusion::strides_dyn_rank;
use crate::fusion::WgpuFusionHandle;
use crate::{FloatElement, GraphicsApi, IntElement, Wgpu};
use burn_compute::tune::AutotuneOperation;
use burn_fusion::stream::Context;
use burn_fusion::{TensorDescription, TensorStatus};
use burn_tensor::Device;
use std::sync::Arc;

/// Many kernels can be used for the same set of tensor operations fused into one.
///
/// This type makes it easy to group those potential kernels and execute the best one depending on the context.
#[derive(new)]
pub struct FusionKernelSet {
    kernels: Vec<Box<dyn FusionKernel>>,
}

/// An instantiation of a [kernel](Kernel) that can be executed.
#[derive(new)]
pub struct ExecutableKernel {
    kernel: Box<dyn Kernel>,
    handles: Vec<WgpuHandle>,
    client: WgpuComputeClient,
}

/// An instantiation of a [kernel](Kernel) that can be autotuned.
///
/// The main difference with an [executable kernel](ExecutableKernel) is that this kernel can be
/// cloned and executed multiple times to properly collect benchmarks.
///
/// The clone function used is defined in the trait [AutotuneOperation] instead of [Clone].
#[derive(new)]
pub struct AutotunableKernel {
    kernel: Arc<dyn Kernel>,
    handles: Vec<WgpuHandle>,
    client: WgpuComputeClient,
}

/// A selected kernel encapsulates a kernel that should be executed with the provided
/// [output info](OutputInfo).
///
/// It isn't ready for execution yet but should provide all information necessary to
/// a [kernel set](FusionKernelSet) to create an [executable kernel](ExecutableKernel).
#[derive(new)]
pub struct SelectedKernel {
    kernel: Box<dyn Kernel>,
    info: Vec<OutputInfo>,
}

/// The priority of a kernel.
pub enum Priority {
    /// When a kernel can be executed in the specified context with its priority, higher is better.
    Available(u8),
    /// When a kernel can't be executed in the specified context.
    Unavailable,
}

// Information related to the output of this kernel.
pub enum OutputInfo {
    Inplace { input_index: usize },
    Array { size: usize },
}

impl ExecutableKernel {
    /// Execute the kernel.
    pub fn execute(self) {
        self.client
            .execute(self.kernel, &self.handles.iter().collect::<Vec<_>>())
    }
}

impl AutotuneOperation for AutotunableKernel {
    fn execute(self: Box<Self>) {
        self.client.execute(
            Box::new(self.kernel),
            &self.handles.iter().collect::<Vec<_>>(),
        )
    }

    fn clone(&self) -> Box<dyn AutotuneOperation> {
        Box::new(Self {
            kernel: self.kernel.clone(),
            handles: self.handles.iter().map(Clone::clone).collect(),
            client: self.client.clone(),
        })
    }
}

impl From<ExecutableKernel> for AutotunableKernel {
    fn from(value: ExecutableKernel) -> Self {
        Self {
            kernel: Arc::new(value.kernel),
            handles: value.handles,
            client: value.client,
        }
    }
}

pub trait FusionKernel: Send + Sync {
    /// Returns the priority of this kernel based on the input and output information.
    fn priority(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> Priority;
    /// Returns a [selected kernel](SelectedKernel) that can be executed by the compute server.
    fn kernel(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> SelectedKernel;
}

impl FusionKernelSet {
    /// Select the best kernel based on the given information.
    #[allow(clippy::too_many_arguments)]
    pub fn select<G: GraphicsApi, F: FloatElement, I: IntElement>(
        &self,
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
        scalars_f32: usize,
        scalars_i32: usize,
        context: &mut Context<'_, Wgpu<G, F, I>>,
        device: Device<Wgpu<G, F, I>>,
        client: WgpuComputeClient,
        stateful: bool,
    ) -> ExecutableKernel {
        let (handles_input, inputs_description_updated, outputs_description_updated) =
            process_inputs_outputs(inputs, outputs, context, stateful);

        let selected = self.select_kernel(
            &handles_input,
            &inputs_description_updated,
            &outputs_description_updated,
        );

        let rank_input = inputs.first().map(|desc| desc.shape.len()).unwrap_or(1);
        let rank_output = outputs.first().map(|desc| desc.shape.len()).unwrap_or(1);
        let rank = usize::max(rank_input, rank_output);

        let num_tensors = inputs.len() + outputs.len();
        // The buffer starts with the rank, then each tensor shape and stride.
        let info_size = (num_tensors * rank * 2) + 1;

        let mut num_handles = num_tensors + 1;
        if scalars_f32 > 0 {
            num_handles += 1;
        }
        if scalars_i32 > 0 {
            num_handles += 1;
        }

        let mut info = Vec::with_capacity(info_size);
        let mut handles = Vec::with_capacity(num_handles);
        let mut output_register = Vec::with_capacity(outputs_description_updated.len());

        // We register the info and handles for the inputs.
        for (handle, tensor) in handles_input.into_iter().zip(inputs_description_updated) {
            register_info_tensor(&mut info, tensor, &handle);
            handles.push(handle.handle);
        }

        // We register the info and handles for the outputs.
        for (tensor, output_info) in outputs_description_updated
            .into_iter()
            .zip(selected.info.iter())
        {
            match output_info {
                // Use the input inplace for this output.
                OutputInfo::Inplace { input_index } => {
                    let handle = handles.get(*input_index).unwrap().clone();
                    let handle_fusion = WgpuFusionHandle {
                        client: client.clone(),
                        device: device.clone(),
                        strides: strides_dyn_rank(&tensor.shape),
                        handle,
                    };
                    output_register.push((tensor.id, handle_fusion));
                }
                // Create a new buffer for this output.
                OutputInfo::Array { size } => {
                    let handle_fusion = WgpuFusionHandle {
                        client: client.clone(),
                        device: device.clone(),
                        strides: strides_dyn_rank(&tensor.shape),
                        handle: client.empty(*size),
                    };

                    register_info_tensor(&mut info, tensor, &handle_fusion);
                    handles.push(handle_fusion.handle.clone());
                    output_register.push((tensor.id, handle_fusion));
                }
            };
        }

        // Create the info buffer.
        handles.push(client.create(bytemuck::cast_slice(&info)));

        // Finally we finish with the named bindings.
        if scalars_f32 > 0 {
            handles
                .push(client.create(bytemuck::cast_slice(&context.scalar_floats[0..scalars_f32])));
        }

        if scalars_i32 > 0 {
            handles.push(client.create(bytemuck::cast_slice(&context.scalar_ints[0..scalars_i32])));
        }

        // We have to register the output handles to the context.
        for (id, handle) in output_register {
            context.handles.register_handle(id, handle);
        }

        ExecutableKernel::new(selected.kernel, handles, client)
    }

    fn select_kernel(
        &self,
        handles_input: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> SelectedKernel {
        // For now we simply select the kernel with the highest priority.
        let mut selected = self
            .kernels
            .iter()
            .filter_map(
                |source| match source.priority(handles_input, inputs, outputs) {
                    Priority::Available(priority) => Some((source, priority)),
                    Priority::Unavailable => None,
                },
            )
            .collect::<Vec<_>>();

        selected.sort_by(|(_, priority_a), (_, priority_b)| priority_a.cmp(priority_b));

        let selected = selected.pop().unwrap().0;

        selected.kernel(handles_input, inputs, outputs)
    }
}

fn register_info_tensor(
    info: &mut Vec<u32>,
    tensor: &TensorDescription,
    handle: &WgpuFusionHandle,
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

fn process_inputs_outputs<'a, G: GraphicsApi, F: FloatElement, I: IntElement>(
    inputs: &[&TensorDescription],
    outputs: &[&TensorDescription],
    context: &'a mut Context<'_, Wgpu<G, F, I>>,
    stateful: bool,
) -> (
    Vec<WgpuFusionHandle>,
    Vec<&'a TensorDescription>,
    Vec<&'a TensorDescription>,
) {
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
