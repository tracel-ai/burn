use backend_comparison::persistence::save;
use burn::tensor::{backend::Backend, module::max_pool2d, Distribution, Shape, Tensor};
use burn_common::benchmark::{run_benchmark, Benchmark};

pub struct MaxPool2dBenchmark<B: Backend> {
    shape: Shape,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    name: &'static str,
    device: B::Device,
}

impl<B: Backend> Benchmark for MaxPool2dBenchmark<B> {
    type Args = Tensor<B, 4>;

    fn name(&self) -> String {
        format!("max_pool2d_{}", self.name)
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, x: Self::Args) {
        max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );
    }

    fn prepare(&self) -> Self::Args {
        let [batches, ch, h, w] = self.shape.dims();
        Tensor::random([batches, h, w, ch], Distribution::Default, &self.device)
            .permute([0, 3, 1, 2])
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
    let benchmark = MaxPool2dBenchmark::<B> {
        name: "default",
        shape: [32, 128, 512, 512].into(),
        kernel_size: [5, 5],
        stride: [2, 2],
        padding: [2, 2],
        dilation: [2, 2],
        device: device.clone(),
    };
    let benchmark2 = MaxPool2dBenchmark::<B> {
        name: "unit_stride",
        shape: [32, 32, 512, 512].into(),
        kernel_size: [5, 5],
        stride: [1, 1],
        padding: [2, 2],
        dilation: [1, 1],
        device: device.clone(),
    };

    save::<B>(
        vec![run_benchmark(benchmark), run_benchmark(benchmark2)],
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
