use burn_cube::{
    calculate_cube_count_elemwise, calculate_num_elems_dyn_rank, ir::CubeDim, KernelExpansion,
    KernelSettings,
};
use burn_tensor::repr::TensorDescription;

use crate::{
    fusion::{
        dynamic_settings,
        kernel::{FusionKernel, FusionKernelFactory, OutputRuntimeInfo},
        JitFusionHandle,
    },
    JitRuntime,
};
use std::{marker::PhantomData, sync::Arc};

#[derive(new)]
pub struct ElementWiseKernelFactory<R: JitRuntime> {
    id: String,
    info: Arc<KernelExpansion>,
    cube_dim: CubeDim,
    _runtime: PhantomData<R>,
}

impl<R: JitRuntime> FusionKernelFactory<R> for ElementWiseKernelFactory<R> {
    fn create(
        &self,
        handles_inputs: &[JitFusionHandle<R>],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
        stateful: bool,
    ) -> FusionKernel<R> {
        let cube_dim_x = self.cube_dim.x;
        let cube_dim_y = self.cube_dim.y;

        assert_eq!(cube_dim_x, cube_dim_y, "The grid must be a square");
        let cube_dim = cube_dim_x as usize;

        let vectorize_4 = can_vectorize(handles_inputs, inputs, outputs, 4);
        let vectorize_2 = can_vectorize(handles_inputs, inputs, outputs, 2);

        let mut settings = KernelSettings::default();
        let mut factor = 1;

        settings = dynamic_settings(
            settings,
            &self.info,
            inputs,
            outputs,
            handles_inputs,
            stateful,
        );

        if vectorize_4 {
            settings = settings.vectorize_global(4);
            factor = 4;
        } else if vectorize_2 {
            settings = settings.vectorize_global(2);
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
                let cube_count = calculate_cube_count_elemwise(num_elems / factor, cube_dim);
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
                    cube_count,
                )
            }
            false => {
                let reference_tensor = outputs[0];
                let num_elems = calculate_num_elems_dyn_rank(&reference_tensor.shape);
                let cube_count = calculate_cube_count_elemwise(num_elems / factor, cube_dim);
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
                    cube_count,
                )
            }
        }
    }
}

fn can_vectorize<R: JitRuntime>(
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
