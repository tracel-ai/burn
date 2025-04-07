use backend_comparison::persistence::save;
use burn::tensor::{Distribution, Shape, Tensor, backend::Backend};
use burn_common::benchmark::{Benchmark, run_benchmark};
use derive_new::new;

#[derive(new)]
struct UnaryBenchmark<B: Backend, const D: usize> {
    shape: Shape,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for UnaryBenchmark<B, D> {
    type Args = Tensor<B, D>;

    fn name(&self) -> String {
        "unary".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, args: Self::Args) {
        // Choice of tanh is arbitrary
        B::float_tanh(args.clone().into_primitive().tensor());
    }

    fn prepare(&self) -> Self::Args {
        Tensor::random(self.shape.clone(), Distribution::Default, &self.device)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(
    device: &B::Device,
    feature_name: &str,
    url: Option<&str>,
    token: Option<&str>,
) {
    const D: usize = 3;
    let shape: Shape = [32, 512, 1024].into();

    let benchmark = UnaryBenchmark::<B, D>::new(shape, device.clone());

    save::<B>(
        vec![run_benchmark(benchmark)],
        device,
        feature_name,
        url,
        token,
    )
    .unwrap();
}

fn main() {
    backend_comparison::bench_on_backend!();
}
