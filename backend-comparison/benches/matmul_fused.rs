use backend_comparison::persistence::save;
use burn::tensor::{
    activation::{gelu, relu},
    backend::Backend,
    Distribution, Shape, Tensor,
};
use burn_common::benchmark::{run_benchmark, Benchmark};
use derive_new::new;

#[derive(new)]
struct MatmulBenchmark<B: Backend, const D: usize> {
    shape_lhs: Shape,
    shape_rhs: Shape,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for MatmulBenchmark<B, D> {
    type Args = (Tensor<B, D>, Tensor<B, D>, Tensor<B, 1>);

    fn name(&self) -> String {
        "matmul_relu_bias_gelu".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape_lhs.dims.clone(), self.shape_rhs.dims.clone()]
    }

    fn execute(&self, (lhs, rhs, bias): Self::Args) {
        let _output = gelu(relu(lhs.matmul(rhs)) + bias.unsqueeze());
    }

    fn prepare(&self) -> Self::Args {
        let lhs = Tensor::random(self.shape_lhs.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::random(self.shape_rhs.clone(), Distribution::Default, &self.device);
        let bias = Tensor::random(
            [self.shape_rhs.dims[2]],
            Distribution::Default,
            &self.device,
        );

        (lhs, rhs, bias)
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
    let benchmarks = [
        (2, 4096, 4096, 4096),
        (32, 2048, 2048, 2048),
        (256, 1024, 1024, 1024),
        (1024, 256, 256, 256),
    ]
    .into_iter()
    .map(|(b, m, n, k)| {
        let shape_lhs = [b, m, k].into();
        let shape_rhs = [b, k, n].into();

        MatmulBenchmark::<B, 3>::new(shape_lhs, shape_rhs, device.clone())
    })
    .map(run_benchmark)
    .collect();

    save::<B>(benchmarks, device, feature_name, url, token).unwrap();
}

fn main() {
    backend_comparison::bench_on_backend!();
}
