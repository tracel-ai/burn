use std::marker::PhantomData;

use backend_comparison::persistence::save;
use burn::tensor::{Distribution, Element, Shape, Tensor, backend::Backend};
use burn_common::benchmark::{Benchmark, run_benchmark};
use rand::rng;

pub struct BinaryBenchmark<B: Backend, const D: usize> {
    shape: Shape,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for BinaryBenchmark<B, D> {
    type Args = (Tensor<B, D>, Tensor<B, D>);

    fn name(&self) -> String {
        "binary".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        let _ = lhs.greater(rhs);
    }

    fn prepare(&self) -> Self::Args {
        let lhs = Tensor::<B, D>::random(self.shape.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::<B, D>::random(self.shape.clone(), Distribution::Default, &self.device);

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
    type Args = (Tensor<B, D>, E);

    fn name(&self) -> String {
        "binary_scalar".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        let _ = lhs.equal_elem(rhs);
    }

    fn prepare(&self) -> Self::Args {
        let lhs = Tensor::random(self.shape.clone(), Distribution::Default, &self.device);
        let rhs = E::random(Distribution::Default, &mut rng());

        (lhs, rhs)
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
