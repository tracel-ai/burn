use crate::BytesResource;

pub struct DummyElementwiseAddition;

pub trait DummyKernel {
    fn compute<'a>(&self, resources: &mut [BytesResource]);
}

impl DummyKernel for DummyElementwiseAddition {
    fn compute<'a>(&self, inputs: &mut [BytesResource]) {
        let lhs = &inputs[0].read();
        let rhs = &inputs[1].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();

        for i in 0..size {
            out[i] = lhs[i] + rhs[i];
        }
    }
}
