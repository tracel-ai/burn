use std::marker::PhantomData;

use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_tensor::benchmark::{run_benchmark, Benchmark};
use derive_new::new;

#[derive(new)]
struct UnaryBenchmark<B: Backend, const D: usize> {
    shape: Shape<D>,
    num_repeats: usize,
    backend: PhantomData<B>,
}

impl<B: Backend, const D: usize> Benchmark<B> for UnaryBenchmark<B, D> {
    type Args = Tensor<B, D>;

    fn name(&self) -> String {
        "Unary Ops".into()
    }

    fn execute(&self, args: Self::Args) {
        for _ in 0..self.num_repeats {
            // Choice of tanh is arbitrary
            B::tanh(args.clone().into_primitive());
        }
    }

    fn prepare(&self, device: &B::Device) -> Self::Args {
        Tensor::random_device(self.shape.clone(), Distribution::Default, device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) {
    const D: usize = 3;
    let shape: Shape<D> = [32, 512, 1024].into();
    let num_repeats = 10;

    let benchmark = UnaryBenchmark::<B, D>::new(shape, num_repeats);

    run_benchmark(benchmark, device)
}

fn main() {
    backend_comparison::bench_on_backend!();
}
