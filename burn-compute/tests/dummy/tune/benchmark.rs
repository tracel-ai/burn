use std::marker::PhantomData;

use burn_compute::{
    server::Handle,
    tune::{BenchmarkPool, Operation, TuneBenchmark},
};
use derive_new::new;
use hashbrown::HashMap;
use spin::Mutex;

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
    type Args = Box<dyn DummyKernel>;

    fn prepare(&self) -> Self::Args {
        let kernel = (self.kernel_constructor)();
        kernel
    }

    fn execute_with_handles(&self, args: Self::Args, handles: &[&Handle<DummyServer>]) {
        self.client.execute(args, handles);
    }

    fn sync(&self) {
        self.client.sync()
    }

    fn take_kernel(&self) -> Box<dyn DummyKernel> {
        (self.kernel_constructor)()
    }
}

pub fn make_benchmark_pool<'a, O, S>(
    benchmarks: Vec<DummyBenchmark<'a, O>>,
) -> Mutex<BenchmarkPool<DummyBenchmark<'a, O>, O, S>>
where
    O: Operation,
{
    let cache = HashMap::new();
    let kernel_pool = BenchmarkPool::new(cache, benchmarks);
    Mutex::new(kernel_pool)
}

pub fn get_addition_benchmarks<'a>(client: &'a DummyClient) -> Vec<DummyBenchmark<'a, AdditionOp>> {
    vec![
        DummyElementwiseAdditionType::make_benchmark(&client),
        DummyElementwiseAdditionSlowWrongType::make_benchmark(&client),
    ]
}

pub fn get_multiplication_benchmarks<'a>(
    client: &'a DummyClient,
) -> Vec<DummyBenchmark<'a, MultiplicationOp>> {
    vec![
        DummyElementwiseMultiplicationSlowWrongType::make_benchmark(&client),
        DummyElementwiseMultiplicationType::make_benchmark(&client),
    ]
}

pub fn get_cache_test_benchmarks<'a>(
    client: &'a DummyClient,
) -> Vec<DummyBenchmark<'a, CacheTestOp>> {
    vec![
        CacheTestFastOn3Type::make_benchmark(&client),
        CacheTestSlowOn3Type::make_benchmark(&client),
    ]
}
