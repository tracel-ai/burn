use backend_comparison::persistence::save;
use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_common::{
    benchmark::{run_benchmark, Benchmark},
    sync_type::SyncType,
};
use derive_new::new;

#[derive(new)]
struct MatmulBenchmark<B: Backend, const D: usize> {
    shape_lhs: Shape<D>,
    shape_rhs: Shape<D>,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for MatmulBenchmark<B, D> {
    type Args = (Tensor<B, D>, Tensor<B, D>);

    fn name(&self) -> String {
        "matmul".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape_lhs.dims.into(), self.shape_rhs.dims.into()]
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        lhs.clone().matmul(rhs.clone());
    }

    fn prepare(&self) -> Self::Args {
        let lhs = Tensor::random(self.shape_lhs.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::random(self.shape_rhs.clone(), Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device, SyncType::Wait)
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
    let batch_size = 32;
    let m = 1024;
    let k = 1024;
    let n = 1024;
    let shape_lhs = [batch_size, m, k].into();
    let shape_rhs = [batch_size, k, n].into();

    let benchmark = MatmulBenchmark::<B, D>::new(shape_lhs, shape_rhs, device.clone());

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
