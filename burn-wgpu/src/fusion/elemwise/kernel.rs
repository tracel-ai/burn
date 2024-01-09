use burn_fusion::TensorDescription;

use crate::{
    codegen::InplaceMapping,
    compute::{DynamicKernel, Kernel},
    fusion::{
        kernel::{FusionKernel, Priority},
        source::FusedKernelSource,
        WgpuFusionHandle,
    },
    kernel::{elemwise_workgroup, WORKGROUP_DEFAULT},
};
use std::sync::Arc;

#[derive(new)]
pub struct ScalarElementWise {
    pub(crate) source: Arc<FusedKernelSource>,
}

#[derive(new)]
pub struct VecElementWise<const D: u8> {
    pub(crate) source: Arc<FusedKernelSource>,
}

#[derive(new)]
pub struct InplaceElementWise {
    pub(crate) kernel: Box<dyn FusionKernel>,
    pub(crate) mapping: Vec<InplaceMapping>,
}

impl FusionKernel for ScalarElementWise {
    fn kernel(&self, position: usize, info: &[u32]) -> Box<dyn Kernel> {
        let rank = info[0] as usize;
        let mut num_elems: usize = 1;
        let index = position * rank + 1;
        let start = index + rank; // shape after strides.
        let end = start + rank;

        for i in info[start..end].iter() {
            num_elems *= *i as usize;
        }

        let workgroup = elemwise_workgroup(num_elems, WORKGROUP_DEFAULT);

        Box::new(DynamicKernel::new(self.source.clone(), workgroup))
    }

    fn priority(&self, _handles_input: &[(WgpuFusionHandle, &TensorDescription)]) -> Priority {
        Priority::Available(0)
    }

    fn source(&self) -> FusedKernelSource {
        self.source.as_ref().clone()
    }
}

impl<const D: u8> FusionKernel for VecElementWise<D> {
    fn kernel(&self, position: usize, info: &[u32]) -> Box<dyn Kernel> {
        let rank = info[0] as usize;
        let mut num_elems: usize = 1;
        let index = position * rank + 1;
        let start = index + rank; // shape after strides.
        let end = start + rank;

        for i in info[start..end].iter() {
            num_elems *= *i as usize;
        }

        let workgroup = elemwise_workgroup(num_elems / D as usize, WORKGROUP_DEFAULT);

        Box::new(DynamicKernel::new(self.source.clone(), workgroup))
    }

    fn priority(&self, handles_input: &[(WgpuFusionHandle, &TensorDescription)]) -> Priority {
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

        for handle in handles_input {
            if is_unavailable(&handle.0, handle.1) {
                return Priority::Unavailable;
            }
        }

        Priority::Available(D)
    }

    fn source(&self) -> FusedKernelSource {
        self.source.as_ref().clone()
    }
}

impl FusionKernel for InplaceElementWise {
    fn kernel(&self, position: usize, info: &[u32]) -> Box<dyn Kernel> {
        self.kernel.kernel(position, info)
    }

    fn priority(&self, handles_input: &[(WgpuFusionHandle, &TensorDescription)]) -> Priority {
        return Priority::Unavailable;
        let priotity = self.kernel.priority(handles_input);

        match priotity {
            Priority::Available(score) => {
                for mapping in self.mapping.iter() {
                    let handle = &handles_input[mapping.position_input];

                    if !handle.0.handle.can_mut() {
                        return Priority::Unavailable;
                    }

                    let mut current = 0;
                    for stride in handle.0.strides.iter().rev() {
                        if current > *stride {
                            return Priority::Unavailable;
                        }
                        current = *stride;
                    }
                }

                Priority::Available(score + 1)
            }
            Priority::Unavailable => Priority::Unavailable,
        }
    }

    fn source(&self) -> FusedKernelSource {
        self.kernel.source()
    }

    fn inplace_mappings(&self) -> &[InplaceMapping] {
        &self.mapping
    }
}
