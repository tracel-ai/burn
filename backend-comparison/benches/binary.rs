use std::marker::PhantomData;

use backend_comparison::persistence::save;
use burn::tensor::{backend::Backend, Bool, Distribution, Element, Shape, Tensor};
use burn_common::benchmark::{run_benchmark, Benchmark};
use rand::rng;

pub struct BinaryBenchmark<B: Backend, const D: usize> {
    shape: Shape,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for BinaryBenchmark<B, D> {
    type Args = (Tensor<B, D, Bool>, Tensor<B, D, Bool>);

    fn name(&self) -> String {
        "binary".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        let _ = lhs.bool_or(rhs);
    }

    fn prepare(&self) -> Self::Args {
        let lhs = Tensor::<B, D>::random(self.shape.clone(), Distribution::Default, &self.device)
            .greater_elem(0.5);
        let rhs = Tensor::<B, D>::random(self.shape.clone(), Distribution::Default, &self.device)
            .greater_elem(0.5);

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device);
    }
}

pub struct BinaryScalarBenchmark<B: Backend, const D: usize, E: Element> {
    shape: Shape,
    device: B::Device,
    _ty: PhantomData<E>,
}

impl<B: Backend, const D: usize, E: Element> Benchmark for BinaryScalarBenchmark<B, D, E> {
    type Args = (Tensor<B, D>, E, E);

    fn name(&self) -> String {
        "binary_scalar".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, (lhs, min, max): Self::Args) {
        let _ = lhs.clamp(min, max);
    }

    fn prepare(&self) -> Self::Args {
        let lhs = Tensor::random(self.shape.clone(), Distribution::Default, &self.device);
        let min = E::random(Distribution::Default, &mut rng());
        let max = E::random(Distribution::Default, &mut rng());

        (lhs, min, max)
    }

    fn sync(&self) {
        B::sync(&self.device);
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(
    device: &B::Device,
    feature_name: &str,
    url: Option<&str>,
    token: Option<&str>,
) {
    let benchmark = BinaryBenchmark::<B, 3> {
        shape: [512, 512, 1024].into(),
        device: device.clone(),
    };
    let benchmark_scalar = BinaryScalarBenchmark::<B, 3, B::FloatElem> {
        shape: [512, 512, 1024].into(),
        device: device.clone(),
        _ty: PhantomData,
    };

    save::<B>(
        vec![run_benchmark(benchmark), run_benchmark(benchmark_scalar)],
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
