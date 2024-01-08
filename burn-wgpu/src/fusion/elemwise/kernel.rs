use crate::{
    compute::DynamicKernel,
    fusion::{
        kernel::{FusionKernel, Priority},
        source::FusedKernelSource,
    },
    kernel::{elemwise_workgroup, WORKGROUP_DEFAULT},
};

#[derive(new)]
pub struct ScalarElemenWise {
    pub(crate) source: FusedKernelSource,
}

#[derive(new)]
pub struct VecElemenWise<const D: u8> {
    pub(crate) source: FusedKernelSource,
}

impl FusionKernel for ScalarElemenWise {
    fn kernel(
        &self,
        _input_indices: &[usize],
        output_indices: &[usize],
        info: &[u32],
    ) -> Box<dyn crate::compute::Kernel> {
        let rank = info[0] as usize;
        let mut num_elems: usize = 1;
        let index = output_indices[0];
        let start = index + rank; // shape after strides.
        let end = start + rank;

        for i in info[start..end].iter() {
            num_elems *= *i as usize;
        }

        let workgroup = elemwise_workgroup(num_elems, WORKGROUP_DEFAULT);

        Box::new(DynamicKernel::new(self.source.clone(), workgroup))
    }

    fn priority(
        &self,
        _input_indices: &[usize],
        _output_indices: &[usize],
        _info: &[u32],
    ) -> Priority {
        Priority::Available(0)
    }
}

impl<const D: u8> FusionKernel for VecElemenWise<D> {
    fn kernel(
        &self,
        _input_indices: &[usize],
        output_indices: &[usize],
        info: &[u32],
    ) -> Box<dyn crate::compute::Kernel> {
        let rank = info[0] as usize;
        let mut num_elems: usize = 1;
        let index = output_indices[0];
        let start = index + rank; // shape after strides.
        let end = start + rank;

        for i in info[start..end].iter() {
            num_elems *= *i as usize;
        }

        let workgroup = elemwise_workgroup(num_elems / D as usize, WORKGROUP_DEFAULT);

        Box::new(DynamicKernel::new(self.source.clone(), workgroup))
    }

    fn priority(
        &self,
        input_indices: &[usize],
        output_indices: &[usize],
        info: &[u32],
    ) -> Priority {
        let rank = info[0] as usize;

        let is_unavailable = |index: &usize| {
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

        for index in input_indices {
            if is_unavailable(index) {
                return Priority::Unavailable;
            }
        }

        for index in output_indices {
            if is_unavailable(index) {
                return Priority::Unavailable;
            }
        }

        Priority::Available(D)
    }
}
