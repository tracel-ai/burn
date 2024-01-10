use crate::{
    codegen::{calculate_num_elems_dyn_rank, InplaceMapping},
    compute::DynamicKernel,
    fusion::{
        kernel::{FusionKernel, KernelVariant, Priority},
        source::DynKernelSource,
        WgpuFusionHandle,
    },
    kernel::{elemwise_workgroup, WORKGROUP_DEFAULT},
};
use burn_fusion::TensorDescription;
use std::sync::Arc;

#[derive(new)]
pub struct ScalarElementWise {
    pub(crate) source_normal: Arc<DynKernelSource>,
    pub(crate) source_inplace: Arc<DynKernelSource>,
    pub(crate) mappings: Vec<InplaceMapping>,
}

#[derive(new)]
pub struct VecElementWise<const D: u8> {
    pub(crate) source_normal: Arc<DynKernelSource>,
    pub(crate) source_inplace: Arc<DynKernelSource>,
    pub(crate) mappings: Vec<InplaceMapping>,
}

impl FusionKernel for ScalarElementWise {
    fn kernel(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> KernelVariant {
        match inplace_available(&self.mappings, handles_inputs) {
            true => {
                let reference_tensor = inputs[self.mappings[0].position_input];
                let num_elems = calculate_num_elems_dyn_rank(&reference_tensor.shape);
                let workgroup = elemwise_workgroup(num_elems, WORKGROUP_DEFAULT);
                let kernel = Box::new(DynamicKernel::new(self.source_inplace.clone(), workgroup));

                KernelVariant::Inplace(kernel, self.mappings.clone())
            }
            false => {
                let reference_tensor = outputs[0];
                let num_elems = calculate_num_elems_dyn_rank(&reference_tensor.shape);
                let workgroup = elemwise_workgroup(num_elems, WORKGROUP_DEFAULT);
                let kernel = Box::new(DynamicKernel::new(self.source_normal.clone(), workgroup));

                KernelVariant::Normal(kernel)
            }
        }
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

impl<const D: u8> FusionKernel for VecElementWise<D> {
    fn kernel(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        outputs: &[&TensorDescription],
    ) -> KernelVariant {
        match inplace_available(&self.mappings, handles_inputs) {
            true => {
                let reference_tensor = inputs[self.mappings[0].position_input];
                let num_elems = calculate_num_elems_dyn_rank(&reference_tensor.shape);
                let workgroup = elemwise_workgroup(num_elems / D as usize, WORKGROUP_DEFAULT);
                let kernel = Box::new(DynamicKernel::new(self.source_inplace.clone(), workgroup));

                KernelVariant::Inplace(kernel, self.mappings.clone())
            }
            false => {
                let reference_tensor = outputs[0];
                let num_elems = calculate_num_elems_dyn_rank(&reference_tensor.shape);
                let workgroup = elemwise_workgroup(num_elems, WORKGROUP_DEFAULT);
                let kernel = Box::new(DynamicKernel::new(self.source_normal.clone(), workgroup));

                KernelVariant::Normal(kernel)
            }
        }
    }
    fn priority(
        &self,
        handles_inputs: &[WgpuFusionHandle],
        inputs: &[&TensorDescription],
        _outputs: &[&TensorDescription],
    ) -> Priority {
        let is_unavailable = |handle: &WgpuFusionHandle, desc: &TensorDescription| {
            let rank = handle.strides.len();

            // Last dimension strides should be 1, otherwise vecX won't be contiguous.
            if handle.strides[rank - 1] != 1 {
                return true;
            }

            // The last dimension should be a multiple of the vector size.
            if desc.shape[rank - 1] % D as usize != 0 {
                return true;
            }

            false
        };

        for (handle, tensor) in handles_inputs.iter().zip(inputs.iter()) {
            if is_unavailable(handle, tensor) {
                return Priority::Unavailable;
            }
        }

        Priority::Available(D)
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
