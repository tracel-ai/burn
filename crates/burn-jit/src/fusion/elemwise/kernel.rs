use burn_tensor::repr::TensorDescription;

use crate::{
    codegen::{
        calculate_num_elems_dyn_rank,
        dialect::gpu::{self, WorkgroupSize},
        CompilationInfo, CompilationSettings,
    },
    fusion::{
        kernel::{FusionKernel, FusionKernelFactory, OutputRuntimeInfo},
        JitFusionHandle,
    },
    kernel::elemwise_workgroup,
    Runtime,
};
use std::{marker::PhantomData, sync::Arc};

#[derive(new)]
pub struct ElementWiseKernelFactory<R: Runtime> {
    id: String,
    info: Arc<CompilationInfo>,
    grid: WorkgroupSize,
    _runtime: PhantomData<R>,
}

impl<R: Runtime> FusionKernelFactory<R> for ElementWiseKernelFactory<R> {
    fn create(
        &self,
        handles_inputs: &[JitFusionHandle<R>],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
        stateful: bool,
    ) -> FusionKernel<R> {
        let workgroup_size_x = self.grid.x;
        let workgroup_size_y = self.grid.y;

        assert_eq!(
            workgroup_size_x, workgroup_size_y,
            "The grid must be a square"
        );
        let workgroup_size = workgroup_size_x as usize;

        let vectorize_4 = can_vectorize(handles_inputs, inputs, outputs, 4);
        let vectorize_2 = can_vectorize(handles_inputs, inputs, outputs, 2);

        let mut settings = CompilationSettings::default();
        let mut factor = 1;

        settings = settings.dynamic_settings(&self.info, inputs, outputs, handles_inputs, stateful);

        if vectorize_4 {
            settings = settings.vectorize(gpu::Vectorization::Vec4);
            factor = 4;
        }

        if !vectorize_4 && vectorize_2 {
            settings = settings.vectorize(gpu::Vectorization::Vec2);
            factor = 2;
        }

        match !settings.mappings.is_empty() {
            true => {
                let mut inplace_output2input = vec![None; self.info.outputs.len()];

                for mapping in settings.mappings.iter() {
                    inplace_output2input[mapping.pos_output] = Some(mapping.pos_input);
                }

                let reference_tensor = inputs[settings.mappings[0].pos_input];
                let num_elems = calculate_num_elems_dyn_rank(&reference_tensor.shape);
                let workgroup = elemwise_workgroup(num_elems / factor, workgroup_size);
                let output_infos =
                    inplace_output2input
                        .iter()
                        .enumerate()
                        .map(|(output_pos, input_pos)| match input_pos {
                            Some(input_index) => OutputRuntimeInfo::Inplace {
                                input_index: *input_index,
                            },
                            None => {
                                let size = calculate_num_elems_dyn_rank(&outputs[output_pos].shape)
                                    * self.info.outputs[output_pos].elem_size::<R>();
                                OutputRuntimeInfo::Array { size }
                            }
                        });

                FusionKernel::new(
                    self.id.clone(),
                    self.info.clone(),
                    settings,
                    output_infos.collect(),
                    workgroup,
                )
            }
            false => {
                let reference_tensor = outputs[0];
                let num_elems = calculate_num_elems_dyn_rank(&reference_tensor.shape);
                let workgroup = elemwise_workgroup(num_elems / factor, workgroup_size);
                let output_infos = outputs.iter().enumerate().map(|(pos, tensor)| {
                    let size = calculate_num_elems_dyn_rank(&tensor.shape)
                        * self.info.outputs[pos].elem_size::<R>();
                    OutputRuntimeInfo::Array { size }
                });

                FusionKernel::new(
                    self.id.clone(),
                    self.info.clone(),
                    settings,
                    output_infos.collect(),
                    workgroup,
                )
            }
        }
    }
}

fn can_vectorize<R: Runtime>(
    handles_inputs: &[JitFusionHandle<R>],
    inputs: &[&TensorDescription],
    outputs: &[&TensorDescription],
    factor: usize,
) -> bool {
    let is_unavailable_input = |handle: &JitFusionHandle<R>, desc: &TensorDescription| {
        let rank = handle.strides.len();

        // Last dimension strides should be 1, otherwise vecX won't be contiguous.
        if handle.strides[rank - 1] != 1 {
            return true;
        }

        // The last dimension should be a multiple of the vector size.
        desc.shape[rank - 1] % factor != 0
    };
    let is_unavailable_output = |desc: &TensorDescription| {
        let rank = desc.shape.len();

        // The last dimension should be a multiple of the vector size.
        desc.shape[rank - 1] % factor != 0
    };

    for (handle, tensor) in handles_inputs.iter().zip(inputs.iter()) {
        if is_unavailable_input(handle, tensor) {
            return false;
        }
    }

    // Only need to check when there is no input.
    if handles_inputs.is_empty() {
        for tensor in outputs.iter() {
            if is_unavailable_output(tensor) {
                return false;
            }
        }
    }

    true
}
