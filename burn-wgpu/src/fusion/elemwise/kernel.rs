use crate::{
    codegen::{calculate_num_elems_dyn_rank, InplaceMapping},
    compute::DynamicKernel,
    fusion::{
        kernel::{FusionKernel, OutputInfo, Priority, SelectedKernel},
        source::GpuKernelSource,
        WgpuFusionHandle,
    },
    kernel::elemwise_workgroup,
};
use burn_fusion::TensorDescription;
use std::sync::Arc;

pub struct ScalarElementWise {
    source: ElementWiseSource,
}

pub struct VecElementWise {
    source: ElementWiseSource,
}

impl FusionKernel for ScalarElementWise {
    fn kernel(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> SelectedKernel {
        self.source.kernel(handles_inputs, inputs, outputs)
    }

    fn priority(
        &self,
        _handles_inputs: &[WgpuFusionHandle],
        _inputs: &[&TensorDescription],
        _outputs: &[&TensorDescription],
    ) -> Priority {
        Priority::Available(0)
    }
}

impl FusionKernel for VecElementWise {
    fn kernel(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> SelectedKernel {
        self.source.kernel(handles_inputs, inputs, outputs)
    }

    fn priority(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        _outputs: &[&TensorDescription],
    ) -> Priority {
        let is_unavailable_input = |handle: &WgpuFusionHandle, desc: &TensorDescription| {
            let rank = handle.strides.len();

            // Last dimension strides should be 1, otherwise vecX won't be contiguous.
            if handle.strides[rank - 1] != 1 {
                return true;
            }

            // The last dimension should be a multiple of the vector size.
            desc.shape[rank - 1] % self.source.factor != 0
        };
        let is_unavailable_output = |desc: &TensorDescription| {
            let rank = desc.shape.len();

            // The last dimension should be a multiple of the vector size.
            desc.shape[rank - 1] % self.source.factor != 0
        };

        for (handle, tensor) in handles_inputs.iter().zip(inputs.iter()) {
            if is_unavailable_input(handle, tensor) {
                return Priority::Unavailable;
            }
        }

        // Only need to check when there is no input.
        if handles_inputs.is_empty() {
            for tensor in _outputs.iter() {
                if is_unavailable_output(tensor) {
                    return Priority::Unavailable;
                }
            }
        }

        Priority::Available(self.source.factor as u8)
    }
}

impl ElementWiseSource {
    fn kernel(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> SelectedKernel {
        let workgroup_size_x = self.source_normal.shader.workgroup_size.x;
        let workgroup_size_y = self.source_normal.shader.workgroup_size.y;
        assert_eq!(
            workgroup_size_x, workgroup_size_y,
            "The grid must be a square"
        );
        let workgroup_size = workgroup_size_x as usize;

        match inplace_available(&self.mappings, handles_inputs) {
            true => {
                let reference_tensor = inputs[self.mappings[0].position_input];
                let num_elems = calculate_num_elems_dyn_rank(&reference_tensor.shape);
                let workgroup = elemwise_workgroup(num_elems / self.factor, workgroup_size);
                let kernel = Box::new(DynamicKernel::new(self.source_inplace.clone(), workgroup));
                let output_infos =
                    self.inplace_output2input
                        .iter()
                        .enumerate()
                        .map(|(output_pos, input_pos)| match input_pos {
                            Some(input_index) => OutputInfo::Inplace {
                                input_index: *input_index,
                            },
                            None => {
                                // Always use the source normal, since the inplace will not have
                                // binding alignment.
                                let elem =
                                    self.source_normal.shader.outputs[output_pos].item.elem();
                                let size = calculate_num_elems_dyn_rank(&outputs[output_pos].shape)
                                    * elem.size();
                                OutputInfo::Array { size }
                            }
                        });

                SelectedKernel::new(kernel, output_infos.collect())
            }
            false => {
                let reference_tensor = outputs[0];
                let num_elems = calculate_num_elems_dyn_rank(&reference_tensor.shape);
                let workgroup = elemwise_workgroup(num_elems / self.factor, workgroup_size);
                let kernel = Box::new(DynamicKernel::new(self.source_normal.clone(), workgroup));
                let output_infos = outputs.iter().enumerate().map(|(pos, tensor)| {
                    let elem = self.source_normal.shader.outputs[pos].item.elem();
                    let size = calculate_num_elems_dyn_rank(&tensor.shape) * elem.size();
                    OutputInfo::Array { size }
                });

                SelectedKernel::new(kernel, output_infos.collect())
            }
        }
    }
}

struct ElementWiseSource {
    source_normal: Arc<GpuKernelSource>,
    source_inplace: Arc<GpuKernelSource>,
    mappings: Vec<InplaceMapping>,
    inplace_output2input: Vec<Option<usize>>,
    factor: usize,
}

impl ElementWiseSource {
    pub fn new(
        normal: GpuKernelSource,
        inplace: GpuKernelSource,
        mappings: Vec<InplaceMapping>,
        num_output: usize,
        factor: usize,
    ) -> Self {
        let mut inplace_output2input = vec![None; num_output];

        for mapping in mappings.iter() {
            inplace_output2input[mapping.position_output] = Some(mapping.position_input);
        }

        Self {
            source_normal: Arc::new(normal),
            source_inplace: Arc::new(inplace),
            mappings,
            inplace_output2input,
            factor,
        }
    }
}

impl ScalarElementWise {
    pub fn new(
        normal: GpuKernelSource,
        inplace: GpuKernelSource,
        mappings: Vec<InplaceMapping>,
        num_output: usize,
    ) -> Self {
        Self {
            source: ElementWiseSource::new(normal, inplace, mappings, num_output, 1),
        }
    }
}

impl VecElementWise {
    pub fn new(
        normal: GpuKernelSource,
        inplace: GpuKernelSource,
        mappings: Vec<InplaceMapping>,
        num_output: usize,
        factor: usize,
    ) -> Self {
        Self {
            source: ElementWiseSource::new(normal, inplace, mappings, num_output, factor),
        }
    }
}

fn inplace_available(mappings: &[InplaceMapping], handles_inputs: &[WgpuFusionHandle]) -> bool {
    if mappings.is_empty() {
        return false;
    }

    for mapping in mappings.iter() {
        let handle = &handles_inputs[mapping.position_input];

        if !handle.handle.can_mut() {
            return false;
        }

        let mut current = 0;
        for stride in handle.strides.iter().rev() {
            if current > *stride {
                return false;
            }
            current = *stride;
        }
    }

    true
}
