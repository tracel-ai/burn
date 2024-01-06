use crate::{
    compute::DynamicKernel,
    fusion::{
        kernel::{FusionKernelSelection, Priority},
        source::FusedKernelSource,
    },
    kernel::{elemwise_workgroup, WORKGROUP_DEFAULT},
};

pub struct ScalarElemenWiseKernelSelection {
    pub(crate) source: FusedKernelSource,
}

pub struct Vec4ElemenWiseKernelSelection {
    pub(crate) source: FusedKernelSource,
}

impl FusionKernelSelection for ScalarElemenWiseKernelSelection {
    fn select(
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

impl FusionKernelSelection for Vec4ElemenWiseKernelSelection {
    fn select(
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

        let workgroup = elemwise_workgroup(num_elems / 4, WORKGROUP_DEFAULT);

        Box::new(DynamicKernel::new(self.source.clone(), workgroup))
    }

    fn priority(
        &self,
        input_indices: &[usize],
        output_indices: &[usize],
        info: &[u32],
    ) -> Priority {
        Priority::Available(0)
    }
}
