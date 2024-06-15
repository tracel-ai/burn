use backend_comparison::persistence::save;
use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_common::{
    benchmark::{run_benchmark, Benchmark},
    sync_type::SyncType,
};

pub struct BinaryBenchmark<B: Backend, const D: usize> {
    shape: Shape<D>,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for BinaryBenchmark<B, D> {
    type Args = (Tensor<B, D>, Tensor<B, D>);

    fn name(&self) -> String {
        "binary".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.into()]
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        // Choice of add is arbitrary
        B::float_add(lhs.clone().into_primitive(), rhs.clone().into_primitive());
    }

    fn prepare(&self) -> Self::Args {
        let lhs = Tensor::random(self.shape.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::random(self.shape.clone(), Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device, SyncType::Wait);
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
        shape: [32, 512, 1024].into(),
        device: device.clone(),
    };

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
