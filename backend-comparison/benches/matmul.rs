use backend_comparison::persistence::save;
use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_common::benchmark::{run_benchmark, Benchmark};
use derive_new::new;

#[derive(new)]
struct MatmulBenchmark<B: Backend, const D: usize> {
    shape_lhs: Shape<D>,
    shape_rhs: Shape<D>,
    num_repeats: usize,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for MatmulBenchmark<B, D> {
    type Args = (Tensor<B, D>, Tensor<B, D>);

    fn name(&self) -> String {
        "matmul".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec!(self.shape_lhs.dims.into(), self.shape_rhs.dims.into())
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn num_repeats(&self) -> usize {
        self.num_repeats
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        for _ in 0..self.num_repeats() {
            lhs.clone().matmul(rhs.clone());
        }
    }

    fn prepare(&self) -> Self::Args {
        let lhs = Tensor::random(self.shape_lhs.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::random(self.shape_rhs.clone(), Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device)
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

    let benchmark = MatmulBenchmark::<B, D>::new(shape_lhs, shape_rhs, num_repeats, device.clone());

    save::<B>(vec![run_benchmark(benchmark)], device).unwrap();
}

fn main() {
    backend_comparison::bench_on_backend!();
}
