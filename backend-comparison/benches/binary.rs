use backend_comparison::persistence::Persistence;
use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_common::benchmark::{run_benchmark, Benchmark};

pub struct BinaryBenchmark<B: Backend, const D: usize> {
    shape: Shape<D>,
    num_repeats: usize,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for BinaryBenchmark<B, D> {
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

    fn prepare(&self) -> Self::Args {
        let lhs = Tensor::random(self.shape.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::random(self.shape.clone(), Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) {
    let benchmark = BinaryBenchmark::<B, 3> {
        shape: [32, 512, 1024].into(),
        num_repeats: 10,
        device: device.clone(),
    };

    Persistence::persist::<B>(vec![run_benchmark(benchmark)], device)
}

fn main() {
    backend_comparison::bench_on_backend!();
}
