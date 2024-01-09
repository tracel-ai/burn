use crate::{
    codegen::InplaceMapping,
    compute::{DynamicKernel, Kernel},
    fusion::{
        kernel::{FusionKernel, Priority},
        source::FusedKernelSource,
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

    fn priority(&self, _num_inputs: usize, _info: &[u32]) -> Priority {
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

    fn priority(&self, num_inputs: usize, info: &[u32]) -> Priority {
        let rank = info[0] as usize;

        let is_unavailable = |index: usize| {
            let last_stride_index = index + rank - 1;
            let last_shape_index = index + (2 * rank) - 1;

            // Last dimension strides should be 1, otherwise vecX won't be contiguous.
            if info[last_stride_index] != 1 {
                return true;
            }

            // The last dimension should be a multiple of the vector size.
            if info[last_shape_index] % D as u32 != 0 {
                return true;
            }

            false
        };

        for pos in 0..num_inputs {
            let index = pos * rank + 1;
            if is_unavailable(index) {
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

    fn priority(&self, num_inputs: usize, info: &[u32]) -> Priority {
        let priotity = self.kernel.priority(num_inputs, info);

        match priotity {
            Priority::Available(score) => {
                let rank = info[0] as usize;

                for mapping in self.mapping.iter() {
                    let index_input = mapping.position_input * (2 * rank) + 1;

                    let strides_input = &info[index_input..index_input + rank];

                    let mut current = 0;
                    for stride in strides_input.iter().rev() {
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
