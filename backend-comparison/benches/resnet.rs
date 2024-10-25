use backend_comparison::persistence::save;
use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_common::benchmark::{run_benchmark, Benchmark};

// Files retrieved during build to avoid reimplementing ResNet for benchmarks
mod block {
    extern crate alloc;
    include!(concat!(env!("OUT_DIR"), "/block.rs"));
}

mod model {
    include!(concat!(env!("OUT_DIR"), "/resnet.rs"));
}

pub struct ResNetBenchmark<B: Backend> {
    shape: Shape,
    device: B::Device,
}

impl<B: Backend> Benchmark for ResNetBenchmark<B> {
    type Args = (model::ResNet<B>, Tensor<B, 4>);

    fn name(&self) -> String {
        "resnet50".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }

    fn execute(&self, (model, input): Self::Args) {
        let _out = model.forward(input);
    }

    fn prepare(&self) -> Self::Args {
        // 1k classes like ImageNet
        let model = model::ResNet::resnet50(1000, &self.device);
        let input = Tensor::random(self.shape.clone(), Distribution::Default, &self.device);

        (model, input)
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
    let benchmark = ResNetBenchmark::<B> {
        shape: [1, 3, 224, 224].into(),
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
