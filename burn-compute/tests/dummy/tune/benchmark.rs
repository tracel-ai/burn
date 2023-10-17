use std::marker::PhantomData;

use burn_compute::{
    server::Handle,
    tune::{AutotuneKernel, TuneBenchmark},
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
pub struct DummyBenchmark<O> {
    client: DummyClient,
    kernel_constructor: Box<dyn Fn() -> Box<dyn DummyKernel>>,
    _operation: PhantomData<O>,
}

impl<'a, O: AutotuneKernel<DummyServer>> TuneBenchmark<O, DummyServer> for DummyBenchmark<O> {
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

pub fn get_addition_kernels(client: DummyClient) -> Vec<DummyBenchmark<AdditionOp>> {
    vec![
        DummyElementwiseAdditionType::make_benchmark(client.clone()),
        DummyElementwiseAdditionSlowWrongType::make_benchmark(client),
    ]
}

pub fn get_multiplication_benchmarks(client: DummyClient) -> Vec<DummyBenchmark<MultiplicationOp>> {
    vec![
        DummyElementwiseMultiplicationSlowWrongType::make_benchmark(client.clone()),
        DummyElementwiseMultiplicationType::make_benchmark(client),
    ]
}

pub fn get_cache_test_benchmarks(client: DummyClient) -> Vec<DummyBenchmark<CacheTestOp>> {
    vec![
        CacheTestFastOn3Type::make_benchmark(client.clone()),
        CacheTestSlowOn3Type::make_benchmark(client),
    ]
}
