use backend_comparison::persistence::save;
use burn::tensor::{
    backend::Backend, module::conv_transpose2d, ops::ConvTransposeOptions, Distribution, Shape,
    Tensor,
};
use burn_common::benchmark::{run_benchmark, Benchmark};

pub struct ConvTranspose2dBenchmark<B: Backend> {
    input_shape: Shape,
    weight_shape: Shape,
    bias_shape: Shape,
    options: ConvTransposeOptions<2>,
    device: B::Device,
}

impl<B: Backend> Benchmark for ConvTranspose2dBenchmark<B> {
    type Args = (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 1>);

    fn name(&self) -> String {
        "conv_transpose2d".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![
            self.input_shape.dims.clone(),
            self.weight_shape.dims.clone(),
            self.bias_shape.dims.clone(),
        ]
    }

    fn execute(&self, (x, w, b): Self::Args) {
        conv_transpose2d(x, w, Some(b), self.options.clone());
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
    // Shapes
    let batch_size = 16;
    let channels_in = 16;
    let channels_out = 16;
    let height_in = 64;
    let width_in = 64;
    let kernel_size_0 = 8;
    let kernel_size_1 = 8;

    // Options
    let strides = [1, 1];
    let padding = [0, 0];
    let padding_out = [0, 0];
    let dilations = [1, 1];
    let groups = 1;
    let options = ConvTransposeOptions::new(strides, padding, padding_out, dilations, groups);
    let benchmark = ConvTranspose2dBenchmark::<B> {
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
