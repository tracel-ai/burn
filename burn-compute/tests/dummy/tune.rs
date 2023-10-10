use std::sync::Arc;

use burn_common::benchmark::Benchmark;
use burn_compute::{
    channel::ComputeChannel,
    client::ComputeClient,
    server::{ComputeServer, Handle},
    tune::{InputHashable, KernelPool, Operation, TuneBenchmark},
};
use derive_new::new;
use hashbrown::HashMap;
use spin::Mutex;

use super::{
    DummyChannel, DummyClient, DummyElementwiseAddition, DummyElementwiseAdditionAlt,
    DummyElementwiseMultiplication, DummyElementwiseMultiplicationAlt, DummyKernel, DummyServer,
};

#[derive(new, PartialEq, Eq, Hash)]
pub struct ArrayHashable {
    pub sizes: [usize; 3],
}
impl ArrayHashable {
    fn to_bytes(&self) -> &[u8] {
        todo!()
    }
}

impl InputHashable for ArrayHashable {
    fn custom_hash(&self) -> String {
        let mut hash = String::new();
        for size in self.sizes {
            hash.push_str(size.to_string().as_str());
            hash.push_str(",");
        }
        hash
    }
}

#[derive(new)]
pub struct DummyBenchmark<'a> {
    client: &'a DummyClient,
    kernel_constructor: Box<dyn Fn() -> Box<dyn DummyKernel>>,
    // kernel: Option<Box<dyn DummyKernel>>,
    handles: &'a [&'a Handle<DummyServer>],
}

impl<'a> Benchmark for DummyBenchmark<'a> {
    type Args = Box<dyn DummyKernel>;

    fn prepare(&self) -> Self::Args {
        (self.kernel_constructor)()
    }

    fn execute(&self, args: Self::Args) {
        self.client.execute(args, self.handles);
    }

    fn name(&self) -> String {
        "i'm dummy".to_owned()
    }

    fn sync(&self) {
        self.client.sync()
    }
}

impl<'a, O> TuneBenchmark<O, DummyServer> for DummyBenchmark<'a> {
    fn take_kernel(&self) -> Box<dyn DummyKernel> {
        (self.kernel_constructor)()
    }
}

macro_rules! make_kernel {
    ($name:ident, $kernel:ident, $operation:ident) => {
        pub struct $name {}

        impl $name {
            pub fn make_benchmark<'a>(
                client: &'a DummyClient,
                handles: &'a [&'a Handle<DummyServer>],
            ) -> DummyBenchmark<'a> {
                let kernel_constructor: Box<dyn Fn() -> <DummyServer as ComputeServer>::Kernel> =
                    Box::new(|| Box::new($kernel {}));
                DummyBenchmark::new(client, kernel_constructor, handles)
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

make_kernel!(
    DummyElementwiseAdditionType,
    DummyElementwiseAddition,
    AdditionOp
);
make_kernel!(
    DummyElementwiseAdditionAltType,
    DummyElementwiseAdditionAlt,
    AdditionOp
);
make_kernel!(
    DummyElementwiseMultiplicationType,
    DummyElementwiseMultiplication,
    MultiplicationOp
);
make_kernel!(
    DummyElementwiseMultiplicationAltType,
    DummyElementwiseMultiplicationAlt,
    MultiplicationOp
);

pub fn make_kernel_pool<'a, O, S>(
    client: &'a DummyClient,
    handles: &'a [&'a Handle<DummyServer>],
) -> Mutex<KernelPool<DummyBenchmark<'a>, O, S>>
where
    O: Operation,
{
    let cache = HashMap::new();
    let benchmarks: Vec<DummyBenchmark> = vec![
        DummyElementwiseAdditionType::make_benchmark(&client, handles),
        DummyElementwiseAdditionAltType::make_benchmark(&client, handles),
        // DummyElementwiseMultiplicationType::kernel_type(),
        // DummyElementwiseMultiplicationAltType::kernel_type(),
    ];

    let kernel_pool = KernelPool::new(cache, benchmarks);
    Mutex::new(kernel_pool)
}
