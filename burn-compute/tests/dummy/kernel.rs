use std::{thread::sleep, time::Duration};

use burn_compute::storage::BytesResource;

/// The DummyKernel trait should be implemented for every supported operation
pub trait DummyKernel: Send {
    fn compute(&self, resources: &mut [BytesResource]);
}

/// Contains the algorithm for element-wise addition
pub struct DummyElementwiseAddition;
pub struct DummyElementwiseAdditionSlowWrong;
pub struct DummyElementwiseMultiplication;
pub struct DummyElementwiseMultiplicationSlowWrong;
pub struct CacheTestFastOn3;
pub struct CacheTestSlowOn3;

impl DummyKernel for DummyElementwiseAddition {
    fn compute(&self, inputs: &mut [BytesResource]) {
        // Notice how the kernel is responsible for determining which inputs
        // are read-only and which are writable.
        let lhs = &inputs[0].read();
        let rhs = &inputs[1].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();

        for i in 0..size {
            out[i] = lhs[i] + rhs[i];
        }
    }
}

impl DummyKernel for DummyElementwiseAdditionSlowWrong {
    fn compute(&self, inputs: &mut [BytesResource]) {
        // Slow and wrong on purpose, for tests
        let lhs = &inputs[0].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();

        for i in 0..size {
            sleep(Duration::from_millis(10));
            out[i] = lhs[i]
        }
    }
}
impl DummyKernel for DummyElementwiseMultiplication {
    fn compute(&self, inputs: &mut [BytesResource]) {
        let lhs = &inputs[0].read();
        let rhs = &inputs[1].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();

        for i in 0..size {
            out[i] = lhs[i] * rhs[i];
        }
    }
}
impl DummyKernel for DummyElementwiseMultiplicationSlowWrong {
    fn compute(&self, inputs: &mut [BytesResource]) {
        // Slow and wrong on purpose, for tests
        let lhs = &inputs[0].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();

        for i in 0..size {
            sleep(Duration::from_millis(10));
            out[i] = lhs[i];
        }
    }
}
impl DummyKernel for CacheTestFastOn3 {
    fn compute(&self, inputs: &mut [BytesResource]) {
        // This is an artificial kernel designed for testing cache only
        let lhs = &inputs[0].read();
        let rhs = &inputs[1].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();
        if size == 3 {
            for i in 0..size {
                out[i] = lhs[i];
            }
        } else {
            for i in 0..size {
                sleep(Duration::from_millis(10));
                out[i] = lhs[i];
            }
        }
    }
}

impl DummyKernel for CacheTestSlowOn3 {
    fn compute(&self, inputs: &mut [BytesResource]) {
        // This is an artificial kernel designed for testing cache only
        let lhs = &inputs[0].read();
        let rhs = &inputs[1].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();
        if size == 3 {
            for i in 0..size {
                sleep(Duration::from_millis(10));
                out[i] = rhs[i];
            }
        } else {
            for i in 0..size {
                out[i] = rhs[i];
            }
        }
    }
}
