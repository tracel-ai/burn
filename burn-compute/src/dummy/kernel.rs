use crate::BytesResource;
use alloc::boxed::Box;
use derive_new::new;

#[derive(new)]
pub struct DummyElementwiseAddition {}

pub trait DummyKernel {
    fn compute<'a>(&self, resources: &mut [BytesResource]);
}

#[derive(new)]
pub struct DummyKernelDescription {
    kernel: Box<dyn DummyKernel>,
}

impl DummyKernelDescription {
    pub fn compute(&self, inputs: &mut [BytesResource]) {
        self.kernel.compute(inputs);
    }
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
