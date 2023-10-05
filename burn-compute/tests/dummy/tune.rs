use std::sync::Arc;

use burn_common::benchmark::Benchmark;
use burn_compute::{
    channel::ComputeChannel,
    client::ComputeClient,
    server::{ComputeServer, Handle},
    tune::{InputHashable, KernelPool, KernelType, Operation, TuneBenchmark},
};
use derive_new::new;
use hashbrown::HashMap;
use spin::Mutex;

use crate::dummy;

use super::{
    DummyChannel, DummyClient, DummyElementwiseAddition, DummyElementwiseAdditionAlt,
    DummyElementwiseMultiplication, DummyElementwiseMultiplicationAlt, DummyServer,
};

#[derive(new, PartialEq, Eq, Hash)]
pub struct ArrayHashable {
    sizes: [usize; 3],
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

#[derive(PartialEq, Eq, Hash)]
pub struct DummyOperationAdd {}
impl Operation<DummyServer> for DummyOperationAdd {
    type Input = ArrayHashable;
}

#[derive(PartialEq, Eq, Hash)]
pub struct DummyOperationMul {}
impl Operation<DummyServer> for DummyOperationMul {
    type Input = ArrayHashable;
}

#[derive(new)]
pub struct DummyBenchmark<S, C> {
    client: ComputeClient<S, C>,
    kernel: Arc<<DummyServer as ComputeServer>::Kernel>,
    handles: &[&Handle<DummyServer>]
}

impl<S, C> Benchmark for DummyBenchmark<S, C>
where
    S: ComputeServer,
    C: ComputeChannel<S>,
{
    type Args = ArrayHashable;

    fn prepare(&self) -> Self::Args {
        self.inputs
    }

    fn execute(&self, args: Self::Args) {
        self.client.execute(kernel, handles);
        todo!()
    }

    fn name(&self) -> String {
        "i'm dummy".to_owned()
    }

    fn sync(&self) {
        self.client.sync()
    }
}

impl<S, C> TuneBenchmark for DummyBenchmark<S, C>
where
    S: ComputeServer,
    C: ComputeChannel<S>,
{
    type Args = ArrayHashable;
}

macro_rules! make_kernel {
    ($name:ident, $kernel:ident) => {
        pub struct $name {}

        impl $name {
            pub fn kernel_type<O>(
                client: &DummyClient,
                handles: &[&Handle<DummyServer>],
            ) -> KernelType<DummyBenchmark<DummyServer, DummyChannel>, O, DummyServer, DummyChannel>
            {
                let kernel: Arc<<DummyServer as ComputeServer>::Kernel> =
                    Arc::new(Box::new($kernel {}));
                let tune_benchmark = Arc::new(DummyBenchmark::new(client, kernel, handles));
                let kernel_type: KernelType<
                    DummyBenchmark<DummyServer, DummyChannel>,
                    O,
                    DummyServer,
                    DummyChannel,
                > = KernelType::new(kernel, tune_benchmark);
                kernel_type
            }
        }
    };
}

make_kernel!(DummyElementwiseAdditionType, DummyElementwiseAddition);
make_kernel!(DummyElementwiseAdditionAltType, DummyElementwiseAdditionAlt);
make_kernel!(
    DummyElementwiseMultiplicationType,
    DummyElementwiseMultiplication
);
make_kernel!(
    DummyElementwiseMultiplicationAltType,
    DummyElementwiseMultiplicationAlt
);

pub fn make_kernel_pool<O>(
    client: DummyClient,
    handles: &[&Handle<DummyServer>],
) -> Mutex<KernelPool<DummyBenchmark<DummyServer, DummyChannel>, O, DummyServer, DummyChannel>>
where
    O: Operation<DummyServer>,
{
    let cache = HashMap::new();
    let kernel_types: Vec<
        KernelType<DummyBenchmark<DummyServer, DummyChannel>, O, DummyServer, DummyChannel>,
    > = vec![
        DummyElementwiseAdditionType::kernel_type(&client, handles),
        DummyElementwiseAdditionAltType::kernel_type(&client, handles),
        // DummyElementwiseMultiplicationType::kernel_type(),
        // DummyElementwiseMultiplicationAltType::kernel_type(),
    ];

    let kernel_pool = KernelPool::new(cache, kernel_types);
    Mutex::new(kernel_pool)
}
