use std::marker::PhantomData;

use burn_common::benchmark::Benchmark;
use burn_compute::{
    server::{ComputeServer, Handle},
    tune::{InputHashable, KernelPool, Operation, TuneBenchmark},
};
use derive_new::new;
use hashbrown::HashMap;
use spin::Mutex;

use super::{
    CacheTestFastOn3, CacheTestSlowOn3, DummyClient, DummyElementwiseAddition,
    DummyElementwiseAdditionSlowWrong, DummyElementwiseMultiplication,
    DummyElementwiseMultiplicationSlowWrong, DummyKernel, DummyServer,
};

#[derive(new, PartialEq, Eq, Hash)]
pub struct ArrayHashable {
    pub sizes: [usize; 3],
}

impl InputHashable for ArrayHashable {
    fn custom_hash(&self) -> String {
        let mut hash = String::new();
        for size in self.sizes {
            let exp = f32::ceil(f32::log2(size as f32)) as u32;
            hash.push_str(2_u32.pow(exp).to_string().as_str());
            hash.push_str(",");
        }
        hash
    }
}

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

macro_rules! make_kernel {
    ($name:ident, $kernel:ident, $operation:ident) => {
        pub struct $name {}

        impl $name {
            pub fn make_benchmark<'a>(client: &'a DummyClient) -> DummyBenchmark<'a, $operation> {
                let kernel_constructor: Box<dyn Fn() -> <DummyServer as ComputeServer>::Kernel> =
                    Box::new(|| Box::new($kernel {}));
                DummyBenchmark::<'a, $operation>::new(client, kernel_constructor)
            }
        }
    };
}

#[derive(PartialEq, Eq, Hash)]
pub struct AdditionOp {}
impl Operation for AdditionOp {
    type Input = ArrayHashable;
}

#[derive(PartialEq, Eq, Hash)]
pub struct MultiplicationOp {}
impl Operation for MultiplicationOp {
    type Input = ArrayHashable;
}

#[derive(PartialEq, Eq, Hash)]
pub struct CacheTestOp {}
impl Operation for CacheTestOp {
    type Input = ArrayHashable;
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

pub fn make_kernel_pool<'a, O, S>(
    benchmarks: Vec<DummyBenchmark<'a, O>>,
) -> Mutex<KernelPool<DummyBenchmark<'a, O>, O, S>>
where
    O: Operation,
{
    let cache = HashMap::new();
    let kernel_pool = KernelPool::new(cache, benchmarks);
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
