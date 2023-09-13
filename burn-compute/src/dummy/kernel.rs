use alloc::{boxed::Box, vec::Vec};
use derive_new::new;

use crate::Memory;

#[derive(new)]
pub struct DummyElementwiseAddition {}

pub trait DummyKernel {
    fn compute(&self, resources: Vec<Memory>);
}

#[derive(new)]
pub struct DummyKernelDescription {
    kernel: Box<dyn DummyKernel>,
}

impl DummyKernelDescription {
    pub fn compute(&self, memories: Vec<Memory>) {
        self.kernel.compute(memories)
    }
}

impl DummyKernel for DummyElementwiseAddition {
    fn compute(&self, mut memories: Vec<Memory>) {
        let lhs = memories[0].to_bytes();
        let rhs = memories[1].to_bytes();
        let length = lhs.len();
        let mut tmp: Vec<u8> = Vec::with_capacity(length);
        for i in 0..length {
            tmp.push(lhs[i] + rhs[i]);
        }
        memories[2].write(tmp)
    }
}
