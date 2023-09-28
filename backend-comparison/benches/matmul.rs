use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_tensor::benchmark::{run_benchmark, Benchmark};
use derive_new::new;
use std::marker::PhantomData;

#[derive(new)]
struct MatmulBenchmark<B, const D: usize> {
    shape_lhs: Shape<D>,
    shape_rhs: Shape<D>,
    num_repeats: usize,
    backend: PhantomData<B>,
}

impl<B: Backend, const D: usize> Benchmark<B> for MatmulBenchmark<B, D> {
    type Args = (Tensor<B, D>, Tensor<B, D>);

    fn name(&self) -> String {
        format!(
            "Matmul {:?} x {:?}",
            self.shape_lhs.dims, self.shape_rhs.dims
        )
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        for _ in 0..self.num_repeats {
            lhs.clone().matmul(rhs.clone());
        }
    }

    fn prepare(&self, device: &B::Device) -> Self::Args {
        let lhs = Tensor::random_device(self.shape_lhs.clone(), Distribution::Default, device);
        let rhs = Tensor::random_device(self.shape_rhs.clone(), Distribution::Default, device);

        (lhs, rhs)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) {
    const D: usize = 3;
    let num_repeats = 3;
    let batch_size = 3;
    let m = 1024;
    let k = 2048;
    let n = 1024;
    let shape_lhs = [batch_size, m, k].into();
    let shape_rhs = [batch_size, k, n].into();

    let benchmark = MatmulBenchmark::<B, D>::new(shape_lhs, shape_rhs, num_repeats);
    run_benchmark(benchmark, device);
}

fn main() {
    backend_comparison::bench_on_backend!();
}
