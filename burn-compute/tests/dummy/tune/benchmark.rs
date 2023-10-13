use std::marker::PhantomData;

use burn_compute::{
    server::Handle,
    tune::{Operation, TuneBenchmark},
};
use derive_new::new;

use crate::dummy::{
    tune::{
        CacheTestFastOn3Type, CacheTestSlowOn3Type, DummyElementwiseAdditionSlowWrongType,
        DummyElementwiseAdditionType, DummyElementwiseMultiplicationSlowWrongType,
        DummyElementwiseMultiplicationType,
    },
    DummyClient, DummyKernel, DummyServer,
};

use super::{AdditionOp, CacheTestOp, MultiplicationOp};

#[derive(new)]
pub struct DummyBenchmark<'a, O> {
    client: &'a DummyClient,
    kernel_constructor: Box<dyn Fn() -> Box<dyn DummyKernel>>,
    _operation: PhantomData<O>,
}

impl<'a, O: Operation> TuneBenchmark<O, DummyServer> for DummyBenchmark<'a, O> {
    fn make_kernel(&self) -> Box<dyn DummyKernel> {
        (self.kernel_constructor)()
    }

    fn execute(&self, kernel: Box<dyn DummyKernel>, handles: &[&Handle<DummyServer>]) {
        self.client.execute(kernel, handles);
    }

    fn sync(&self) {
        self.client.sync()
    }
}

pub fn get_addition_benchmarks(client: &DummyClient) -> Vec<DummyBenchmark<'_, AdditionOp>> {
    vec![
        DummyElementwiseAdditionType::make_benchmark(client),
        DummyElementwiseAdditionSlowWrongType::make_benchmark(client),
    ]
}

pub fn get_multiplication_benchmarks(
    client: &DummyClient,
) -> Vec<DummyBenchmark<'_, MultiplicationOp>> {
    vec![
        DummyElementwiseMultiplicationSlowWrongType::make_benchmark(client),
        DummyElementwiseMultiplicationType::make_benchmark(client),
    ]
}

pub fn get_cache_test_benchmarks(client: &DummyClient) -> Vec<DummyBenchmark<'_, CacheTestOp>> {
    vec![
        CacheTestFastOn3Type::make_benchmark(client),
        CacheTestSlowOn3Type::make_benchmark(client),
    ]
}
