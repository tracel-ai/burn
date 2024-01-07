use crate::{
    compute::DynamicKernel,
    fusion::{
        kernel::{FusionKernelSelection, Priority},
        source::FusedKernelSource,
    },
    kernel::{elemwise_workgroup, WORKGROUP_DEFAULT},
};

#[derive(new)]
pub struct ScalarElemenWiseKernelSelection {
    pub(crate) source: FusedKernelSource,
}

#[derive(new)]
pub struct VecElemenWiseKernelSelection<const D: u8> {
    pub(crate) source: FusedKernelSource,
}

impl FusionKernelSelection for ScalarElemenWiseKernelSelection {
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

    fn shader(&self) -> crate::codegen::ComputeShader {
        self.source.shader.clone()
    }
}

impl<const D: u8> FusionKernelSelection for VecElemenWiseKernelSelection<D> {
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
        println!("Workgroup {workgroup:?}");

        Box::new(DynamicKernel::new(self.source.clone(), workgroup))
    }

    fn priority(
        &self,
        input_indices: &[usize],
        output_indices: &[usize],
        info: &[u32],
    ) -> Priority {
        let rank = info[0] as usize;

        // TODO: More checks to do with regard to strides.
        let is_unavailable = |index: &usize| {
            let start_shape = index + rank;
            let end_shape = start_shape + rank;

            if info[index + rank - 1] != 1 {
                return true;
            }

            for shape in info[end_shape - 1..end_shape].iter() {
                if shape % D as u32 != 0 {
                    return true;
                }
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

    fn shader(&self) -> crate::codegen::ComputeShader {
        self.source.shader.clone()
    }
}
