use super::source::DynKernelSource;
use crate::codegen::{calculate_num_elems_dyn_rank, InplaceMapping};
use crate::compute::{compute_client, Kernel};
use crate::fusion::strides_dyn_rank;
use crate::fusion::WgpuFusionHandle;
use crate::{FloatElement, GraphicsApi, IntElement, Wgpu};
use burn_fusion::graph::Context;
use burn_fusion::TensorDescription;
use burn_tensor::Device;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Many kernels can be used for the same set of tensor operations fused into one.
///
/// This type makes it easy to group those potential kernels and execute the best one depending on
/// the context.
#[derive(new)]
pub struct FusionOperationSet {
    operations: Vec<Box<dyn FusionOperation>>,
}

/// The priority of a kernel.
pub enum Priority {
    /// When a kernel can be executed in the specified context with its priority, higher is better.
    Available(u8),
    /// When a kernel can't be executed in the specified context.
    Unavailable,
}

#[derive(new, Serialize, Deserialize)]
pub struct FusionKernelSource {
    pub normal: DynKernelSource,
    pub inplace: Option<DynKernelSource>,
}

#[derive(new)]
pub enum KernelVariant {
    Normal(Box<dyn Kernel>),
    Inplace(Box<dyn Kernel>, Vec<InplaceMapping>),
}

pub trait FusionOperation: Send + Sync {
    /// Returns the priority of this kernel based on the input and output information.
    ///
    /// # Notes
    ///
    /// The indices indicate the start of each entry in the info buffer.
    /// Each entry starts with the strides then the shape.
    fn priority(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> Priority;
    /// Returns a [kernel](Kernel) that can be executed by the compute server.
    ///
    /// # Notes
    ///
    /// The indices indicate the start of each entry in the info buffer.
    /// Each entry starts with the strides then the shape.
    fn kernel(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> KernelVariant;
    /// Returns the source for this kernel, to be used for serialization.
    fn source(&self) -> FusionKernelSource;
}

impl FusionOperationSet {
    pub fn state(&self) -> Vec<FusionKernelSource> {
        self.operations.iter().map(|kernel| kernel.source()).collect()
    }

    /// Execute the best kernel based on the given information.
    pub fn execute<G: GraphicsApi, F: FloatElement, I: IntElement>(
        &self,
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
        scalars_f32: usize,
        scalars_i32: usize,
        context: &mut Context<'_, Wgpu<G, F, I>>,
        device: Device<Wgpu<G, F, I>>,
    ) {
        let client = compute_client::<G>(&device);
        let mut info = Vec::new();
        let mut inputs_description_updated = Vec::with_capacity(inputs.len());
        let mut outputs_description_updated = Vec::with_capacity(outputs.len());
        let mut handles_input = Vec::new();
        let mut handles = Vec::with_capacity(inputs.len() + outputs.len() + 2);

        // Inner function to fill the info buffer.
        let register_info_tensor =
            |info: &mut Vec<u32>, tensor: &TensorDescription, handle: &WgpuFusionHandle| {
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

        for tensor in inputs.iter() {
            let status = &tensor.status; // Important to take the status of the relative graph and not
                                         // the global graph, since the status of the global graph
                                         // might be of a later operation on the same tensor id.
            let tensor = context.tensors.get(&tensor.id).unwrap();
            let handle = context.handles.get_handle(&tensor.id, status);

            handles_input.push(handle);
            inputs_description_updated.push(tensor);
        }

        for tensor in outputs.iter() {
            let tensor = context.tensors.get(&tensor.id).unwrap();
            outputs_description_updated.push(tensor);
        }

        // For now we simply select the kernel with the highest priority.
        let mut selected = self
            .operations
            .iter()
            .filter_map(|source| {
                match source.priority(
                    &handles_input,
                    &inputs_description_updated,
                    &outputs_description_updated,
                ) {
                    Priority::Available(priority) => Some((source, priority)),
                    Priority::Unavailable => None,
                }
            })
            .collect::<Vec<_>>();

        selected.sort_by(|(_, priority_a), (_, priority_b)| priority_a.cmp(priority_b));

        let selected = selected.pop().unwrap().0;
        let kernel = selected.kernel(
            &handles_input,
            &inputs_description_updated,
            &outputs_description_updated,
        );
        let (kernel, output2input) = match kernel {
            KernelVariant::Normal(kernel) => (kernel, HashMap::default()),
            KernelVariant::Inplace(kernel, mapping) => {
                let output2input: HashMap<usize, usize> = HashMap::from_iter(
                    mapping
                        .iter()
                        .map(|mapping| (mapping.position_output, mapping.position_input)),
                );
                (kernel, output2input)
            }
        };

        for (handle, tensor) in handles_input.into_iter().zip(inputs_description_updated) {
            register_info_tensor(&mut info, tensor, &handle);
            handles.push(handle.handle);
        }

        // Then we follow with the outputs.
        for (i, tensor) in outputs_description_updated.into_iter().enumerate() {
            match output2input.get(&i) {
                Some(position_input) => {
                    let handle = handles.get(*position_input).unwrap().clone();
                    let handle_fusion = WgpuFusionHandle {
                        client: client.clone(),
                        device: device.clone(),
                        strides: strides_dyn_rank(&tensor.shape),
                        handle,
                    };

                    context
                        .handles
                        .register_handle(tensor.id.clone(), handle_fusion);
                }
                None => {
                    let num_elems = calculate_num_elems_dyn_rank(&tensor.shape);
                    let handle_fusion = WgpuFusionHandle {
                        client: client.clone(),
                        device: device.clone(),
                        strides: strides_dyn_rank(&tensor.shape),
                        // TODO: Change size_of for the real type, can create bug.
                        handle: client.empty(core::mem::size_of::<F>() * num_elems),
                    };

                    register_info_tensor(&mut info, tensor, &handle_fusion);

                    handles.push(handle_fusion.handle.clone());
                    context
                        .handles
                        .register_handle(tensor.id.clone(), handle_fusion);
                }
            };
        }

        handles.push(client.create(bytemuck::cast_slice(&info)));

        // Finally we finish with the named bindings.
        if scalars_f32 > 0 {
            handles
                .push(client.create(bytemuck::cast_slice(&context.scalar_floats[0..scalars_f32])));
        }

        if scalars_i32 > 0 {
            handles.push(client.create(bytemuck::cast_slice(&context.scalar_ints[0..scalars_i32])));
        }

        // Execute the kernel.
        client.execute(kernel, &handles.iter().collect::<Vec<_>>());
    }
}
