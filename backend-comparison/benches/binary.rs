use std::marker::PhantomData;

use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_tensor::benchmark::{run_benchmark, Benchmark};

pub struct BinaryBenchmark<B: Backend, const D: usize> {
    shape: Shape<D>,
    num_repeats: usize,
    backend: PhantomData<B>,
}

impl<B: Backend, const D: usize> Benchmark<B> for BinaryBenchmark<B, D> {
    type Args = (Tensor<B, D>, Tensor<B, D>);

    fn name(&self) -> String {
        "Binary Ops".into()
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        for _ in 0..self.num_repeats {
            // Choice of add is arbitrary
            B::add(lhs.clone().into_primitive(), rhs.clone().into_primitive());
        }
    }

    fn prepare(&self, device: &B::Device) -> Self::Args {
        let lhs = Tensor::random(self.shape.clone(), Distribution::Default).to_device(device);
        let rhs = Tensor::random(self.shape.clone(), Distribution::Default).to_device(device);

        (lhs, rhs)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) {
    const D: usize = 3;
    let shape: Shape<D> = [32, 512, 1024].into();
    let num_repeats = 10;

    let benchmark = BinaryBenchmark::<B, D> {
        shape,
        num_repeats,
        backend: PhantomData,
    };

    run_benchmark(benchmark, device)
}

fn main() {
    backend_comparison::bench_on_backend!();
}
