use backend_comparison::persistence::save;
use burn::tensor::{
    Distribution, Shape, Tensor, backend::Backend, module::conv_transpose3d,
    ops::ConvTransposeOptions,
};
use burn_common::benchmark::{Benchmark, run_benchmark};

pub struct ConvTranspose3dBenchmark<B: Backend> {
    input_shape: Shape,
    weight_shape: Shape,
    bias_shape: Shape,
    options: ConvTransposeOptions<3>,
    device: B::Device,
}

impl<B: Backend> Benchmark for ConvTranspose3dBenchmark<B> {
    type Args = (Tensor<B, 5>, Tensor<B, 5>, Tensor<B, 1>);

    fn name(&self) -> String {
        "conv_transpose3d".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![
            self.input_shape.dims.clone(),
            self.weight_shape.dims.clone(),
            self.bias_shape.dims.clone(),
        ]
    }

    fn execute(&self, (x, w, b): Self::Args) {
        conv_transpose3d(x, w, Some(b), self.options.clone());
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
    let depth_in = 4;
    let height_in = 16;
    let width_in = 16;
    let kernel_size_0 = 8;
    let kernel_size_1 = 8;
    let kernel_size_2 = 8;

    // Options
    let strides = [1, 1, 1];
    let padding = [0, 0, 0];
    let padding_out = [0, 0, 0];
    let dilations = [1, 1, 1];
    let groups = 1;
    let options = ConvTransposeOptions::new(strides, padding, padding_out, dilations, groups);
    let benchmark = ConvTranspose3dBenchmark::<B> {
        input_shape: [batch_size, channels_in, depth_in, height_in, width_in].into(),
        weight_shape: [
            channels_in,
            channels_out / groups,
            kernel_size_0,
            kernel_size_1,
            kernel_size_2,
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
