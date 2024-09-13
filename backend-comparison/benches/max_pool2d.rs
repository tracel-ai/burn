use backend_comparison::persistence::save;
use burn::tensor::{backend::Backend, module::max_pool2d, Distribution, Shape, Tensor};
use burn_common::{
    benchmark::{run_benchmark, Benchmark},
    sync_type::SyncType,
};

pub struct MaxPool2dBenchmark<B: Backend> {
    shape: Shape<4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    device: B::Device,
}

impl<B: Backend> Benchmark for MaxPool2dBenchmark<B> {
    type Args = Tensor<B, 4>;

    fn name(&self) -> String {
        "max_pool2d".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.into()]
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
        Tensor::random(self.shape.clone(), Distribution::Default, &self.device)
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
    let benchmark = MaxPool2dBenchmark::<B> {
        shape: [32, 32, 512, 512].into(),
        kernel_size: [5, 5],
        stride: [2, 2],
        padding: [2, 2],
        dilation: [2, 2],
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
