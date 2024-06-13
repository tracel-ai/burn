use backend_comparison::persistence::save;
use burn::tensor::{
    backend::Backend, module::conv2d, ops::ConvOptions, Distribution, Shape, Tensor,
};
use burn_common::{
    benchmark::{run_benchmark, Benchmark},
    sync_type::SyncType,
};

pub struct Conv2dBenchmark<B: Backend> {
    input_shape: Shape<4>,
    weight_shape: Shape<4>,
    bias_shape: Shape<1>,
    options: ConvOptions<2>,
    device: B::Device,
}

impl<B: Backend> Benchmark for Conv2dBenchmark<B> {
    type Args = (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 1>);

    fn name(&self) -> String {
        "conv2d".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![
            self.input_shape.dims.into(),
            self.weight_shape.dims.into(),
            self.bias_shape.dims.into(),
        ]
    }

    fn execute(&self, (x, w, b): Self::Args) {
        conv2d(x, w, Some(b), self.options.clone());
    }

    fn prepare(&self) -> Self::Args {
        (
            Tensor::random(
                self.input_shape.clone(),
                Distribution::Default,
                &self.device,
            ),
            Tensor::random(
                self.weight_shape.clone(),
                Distribution::Default,
                &self.device,
            ),
            Tensor::random(self.bias_shape.clone(), Distribution::Default, &self.device),
        )
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
    // Shapes
    let batch_size = 16;
    let channels_in = 16;
    let channels_out = 16;
    let height_in = 512;
    let width_in = 512;
    let kernel_size_0 = 3;
    let kernel_size_1 = 3;

    // Options
    let strides = [1, 1];
    let padding = [0, 0];
    let dilations = [1, 1];
    let groups = 1;
    let options = ConvOptions::new(strides, padding, dilations, groups);
    let benchmark = Conv2dBenchmark::<B> {
        input_shape: [batch_size, channels_in, height_in, width_in].into(),
        weight_shape: [
            channels_in,
            channels_out / groups,
            kernel_size_0,
            kernel_size_1,
        ]
        .into(),
        bias_shape: [channels_out].into(),
        options,
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
