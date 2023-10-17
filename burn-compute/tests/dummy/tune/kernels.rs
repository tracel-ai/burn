use std::{thread::sleep, time::Duration};

use burn_compute::{server::ComputeServer, storage::BytesResource};

use crate::dummy::{DummyClient, DummyElementwiseAddition, DummyKernel, DummyServer};

use super::{AdditionOp, CacheTestOp, DummyBenchmark, MultiplicationOp};

const SLEEP_MS: u64 = 1;

pub struct DummyElementwiseAdditionSlowWrong;
pub struct DummyElementwiseMultiplication;
pub struct DummyElementwiseMultiplicationSlowWrong;
pub struct CacheTestFastOn3;
pub struct CacheTestSlowOn3;

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

macro_rules! make_kernel {
    ($name:ident, $kernel:ident, $operation:ident) => {
        pub struct $name {}

        impl $name {
            pub fn make_benchmark(client: DummyClient) -> DummyBenchmark<$operation> {
                let kernel_constructor: Box<dyn Fn() -> <DummyServer as ComputeServer>::Kernel> =
                    Box::new(|| Box::new($kernel {}));
                DummyBenchmark::<$operation>::new(client, kernel_constructor)
            }
        }
    };
}

make_kernel!(
    DummyElementwiseAdditionType,
    DummyElementwiseAddition,
    AdditionOp
);
make_kernel!(
    DummyElementwiseAdditionSlowWrongType,
    DummyElementwiseAdditionSlowWrong,
    AdditionOp
);
make_kernel!(
    DummyElementwiseMultiplicationType,
    DummyElementwiseMultiplication,
    MultiplicationOp
);
make_kernel!(
    DummyElementwiseMultiplicationSlowWrongType,
    DummyElementwiseMultiplicationSlowWrong,
    MultiplicationOp
);
make_kernel!(CacheTestFastOn3Type, CacheTestFastOn3, CacheTestOp);
make_kernel!(CacheTestSlowOn3Type, CacheTestSlowOn3, CacheTestOp);
