use std::{thread::sleep, time::Duration};

use burn_compute::storage::BytesResource;

use crate::dummy::DummyKernel;

const SLEEP_MS: u64 = 1;

pub struct DummyElementwiseAdditionSlowWrong;
pub struct DummyElementwiseMultiplication;
pub struct DummyElementwiseMultiplicationSlowWrong;
pub struct CacheTestFastOn3;
pub struct CacheTestSlowOn3;
pub struct ParameteredKernel;

impl DummyKernel for DummyElementwiseAdditionSlowWrong {
    fn compute(&self, inputs: &mut [BytesResource]) {
        // Slow and wrong on purpose, for tests
        let lhs = &inputs[0].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();

        for i in 0..size {
            sleep(Duration::from_millis(SLEEP_MS));
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
            sleep(Duration::from_millis(SLEEP_MS));
            out[i] = lhs[i];
        }
    }
}
impl DummyKernel for CacheTestFastOn3 {
    fn compute(&self, inputs: &mut [BytesResource]) {
        // This is an artificial kernel designed for testing cache only
        let lhs = &inputs[0].read();
        let out = &mut inputs[2].write();

        let size = lhs.len();
        if size == 3 {
            out[..size].copy_from_slice(&lhs[..size]);
        } else {
            for i in 0..size {
                sleep(Duration::from_millis(SLEEP_MS));
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
                sleep(Duration::from_millis(SLEEP_MS));
                out[i] = rhs[i];
            }
        } else {
            out[..size].copy_from_slice(&rhs[..size]);
        }
    }
}

impl DummyKernel for ParameteredKernel {
    fn compute(&self, inputs: &mut [BytesResource]) {
        // This is an artificial kernel designed for info buffer
        let lhs = &inputs[0].read();
        let rhs = &inputs[1].read();
        let out = &mut inputs[2].write();
        let info = &inputs[3].read();

        for i in 0..lhs.len() {
            out[i] = lhs[i] + rhs[i] + info[0];
        }
    }
}
