use super::source::FusedKernelSource;
use crate::codegen::calculate_num_elems_dyn_rank;
use crate::compute::{compute_client, Kernel};
use crate::fusion::strides_dyn_rank;
use crate::fusion::WgpuFusionHandle;
use crate::{FloatElement, GraphicsApi, IntElement, Wgpu};
use burn_fusion::graph::Context;
use burn_fusion::TensorDescription;
use burn_tensor::Device;

/// Many kernels can be used for the same set of tensor operations fused into one.
///
/// This type makes it easy to group those potential kernels and execute the best one depending on
/// the context.
#[derive(new)]
pub struct FusionKernelSet {
    kernels: Vec<Box<dyn FusionKernel>>,
}

/// The priority of a kernel.
pub enum Priority {
    /// When a kernel can be executed in the specified context with its priority, higher is better.
    Available(u8),
    /// When a kernel can't be executed in the specified context.
    Unavailable,
}

pub trait FusionKernel: Send + Sync {
    /// Returns the priority of this kernel based on the input and output information.
    ///
    /// # Notes
    ///
    /// The indices indicate the start of each entry in the info buffer.
    /// Each entry starts with the strides then the shape.
    fn priority(&self, indices_input: &[usize], indices_output: &[usize], info: &[u32])
        -> Priority;
    /// Returns a [kernel](Kernel) that can be executed by the compute server.
    ///
    /// # Notes
    ///
    /// The indices indicate the start of each entry in the info buffer.
    /// Each entry starts with the strides then the shape.
    fn kernel(
        &self,
        indices_input: &[usize],
        indices_output: &[usize],
        info: &[u32],
    ) -> Box<dyn Kernel>;
    /// Returns the source for this kernel, to be used for serialization.
    fn source(&self) -> FusedKernelSource;
}

impl FusionKernelSet {
    pub fn state(&self) -> Vec<FusedKernelSource> {
        self.kernels.iter().map(|kernel| kernel.source()).collect()
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
        enum InfoType {
            Input,
            Output,
        }

        let client = compute_client::<G>(&device);
        let mut info = Vec::new();
        let mut input_indices = Vec::new();
        let mut output_indices = Vec::new();
        let mut handles = Vec::with_capacity(inputs.len() + outputs.len() + 2);

        // Inner function to fill the info buffer.
        let mut register_info_tensor =
            |tensor: &TensorDescription, handle: &WgpuFusionHandle, ty: InfoType| {
                if info.is_empty() {
                    info.push(handle.strides.len() as u32);
                }

                match ty {
                    InfoType::Input => input_indices.push(info.len()),
                    InfoType::Output => output_indices.push(info.len()),
                };

                for s in handle.strides.iter() {
                    info.push(*s as u32);
                }
                for s in tensor.shape.iter() {
                    info.push(*s as u32);
                }
            };

        // We start by registering the inputs.
        for tensor in inputs.iter() {
            let status = &tensor.status; // Important to take the status of the relative graph and not
                                         // the global graph, since the status of the global graph
                                         // might be of a later operation on the same tensor id.
            let tensor = context.tensors.get(&tensor.id).unwrap();
            let handle = context.handles.get_handle(&tensor.id, status);

            register_info_tensor(tensor, &handle, InfoType::Input);
            handles.push(handle.handle);
        }

        let mut num_elems_output = 0;

        // Then we follow with the outputs.
        for tensor in outputs.iter() {
            let tensor = context.tensors.get(&tensor.id).unwrap();

            let num_elems = calculate_num_elems_dyn_rank(&tensor.shape);
            if num_elems > num_elems_output {
                num_elems_output = num_elems;
            }
            let handle_fusion = WgpuFusionHandle {
                client: client.clone(),
                device: device.clone(),
                strides: strides_dyn_rank(&tensor.shape),
                handle: client.empty(core::mem::size_of::<F>() * num_elems),
            };

            register_info_tensor(tensor, &handle_fusion, InfoType::Output);

            handles.push(handle_fusion.handle.clone());
            context
                .handles
                .register_handle(tensor.id.clone(), handle_fusion);
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

        // For now we simply select the kernel with the highest priority.
        let mut selected = self
            .kernels
            .iter()
            .filter_map(
                |source| match source.priority(&input_indices, &output_indices, &info) {
                    Priority::Available(priority) => Some((source, priority)),
                    Priority::Unavailable => None,
                },
            )
            .collect::<Vec<_>>();

        selected.sort_by(|(_, priority_a), (_, priority_b)| priority_a.cmp(priority_b));

        let kernel = selected
            .pop()
            .unwrap()
            .0
            .kernel(&input_indices, &output_indices, &info);

        // Execute the kernel.
        client.execute(kernel, &handles.iter().collect::<Vec<_>>());
    }
}
