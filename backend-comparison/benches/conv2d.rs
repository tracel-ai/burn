use std::hint::black_box;

use backend_comparison::persistence::save;
use burn::tensor::{
    backend::Backend, module::conv2d, ops::ConvOptions, Distribution, Shape, Tensor,
};
use burn_common::benchmark::{run_benchmark, Benchmark};

pub struct Conv2dBenchmark<B: Backend> {
    suffix: &'static str,
    input_shape: Shape,
    weight_shape: Shape,
    bias_shape: Shape,
    options: ConvOptions<2>,
    device: B::Device,
}

impl<B: Backend> Benchmark for Conv2dBenchmark<B> {
    type Args = (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 1>);

    fn name(&self) -> String {
        format!("conv2d-{}", self.suffix)
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![
            self.input_shape.dims.clone(),
            self.weight_shape.dims.clone(),
            self.bias_shape.dims.clone(),
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
        B::sync(&self.device)
    }

    fn num_samples(&self) -> usize {
        40
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
        suffix: "input_16x512x512_weight_16x3x3_stride_1",
        input_shape: [batch_size, channels_in, height_in, width_in].into(),
        weight_shape: [
            channels_out,
            channels_in / groups,
            kernel_size_0,
            kernel_size_1,
        ]
        .into(),
        bias_shape: [channels_out].into(),
        options,
        device: device.clone(),
    };

    let conv1 = Conv2dBenchmark::<B> {
        suffix: "input_3x227x227_weight_96x11x11_stride_4",
        input_shape: [batch_size, 3, 227, 227].into(),
        weight_shape: [96, 3, 11, 11].into(),
        bias_shape: [96].into(),
        options: ConvOptions::new([4, 4], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let conv2 = Conv2dBenchmark::<B> {
        suffix: "input_3x231x231_weight_96x11x11_stride_4",
        input_shape: [batch_size, 3, 231, 231].into(),
        weight_shape: [96, 3, 11, 11].into(),
        bias_shape: [96].into(),
        options: ConvOptions::new([4, 4], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let conv3 = Conv2dBenchmark::<B> {
        suffix: "input_3x227x227_weight_64x7x7_stride_2",
        input_shape: [batch_size, 3, 227, 227].into(),
        weight_shape: [64, 3, 7, 7].into(),
        bias_shape: [64].into(),
        options: ConvOptions::new([2, 2], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let conv4 = Conv2dBenchmark::<B> {
        suffix: "input_64x224x224_weight_64x7x7_stride_2",
        input_shape: [batch_size, 64, 224, 224].into(),
        weight_shape: [64, 64, 7, 7].into(),
        bias_shape: [64].into(),
        options: ConvOptions::new([2, 2], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let conv5 = Conv2dBenchmark::<B> {
        suffix: "input_96x24x24_weight_256x5x5_stride_1",
        input_shape: [batch_size, 96, 24, 24].into(),
        weight_shape: [256, 96, 5, 5].into(),
        bias_shape: [256].into(),
        options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let conv6 = Conv2dBenchmark::<B> {
        suffix: "input_256x12x12_weight_512x3x3_stride_1",
        input_shape: [batch_size, 256, 12, 12].into(),
        weight_shape: [512, 256, 3, 3].into(),
        bias_shape: [512].into(),
        options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let conv7 = Conv2dBenchmark::<B> {
        suffix: "input_3x224x224_weight_64x3x3_stride_1",
        input_shape: [batch_size, 3, 224, 224].into(),
        weight_shape: [64, 3, 3, 3].into(),
        bias_shape: [64].into(),
        options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let conv8 = Conv2dBenchmark::<B> {
        suffix: "input_64x112x112_weight_128x3x3_stride_1",
        input_shape: [batch_size, 64, 112, 112].into(),
        weight_shape: [128, 64, 3, 3].into(),
        bias_shape: [128].into(),
        options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let conv9 = Conv2dBenchmark::<B> {
        suffix: "input_64x56x56_weight_64x3x3_stride_1",
        input_shape: [batch_size, 64, 56, 56].into(),
        weight_shape: [64, 64, 3, 3].into(),
        bias_shape: [64].into(),
        options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let conv10 = Conv2dBenchmark::<B> {
        suffix: "input_128x28x28_weight_128x3x3_stride_1",
        input_shape: [batch_size, 128, 28, 28].into(),
        weight_shape: [128, 128, 3, 3].into(),
        bias_shape: [128].into(),
        options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let conv11 = Conv2dBenchmark::<B> {
        suffix: "input_256x14x14_weight_256x3x3_stride_1",
        input_shape: [batch_size, 256, 14, 14].into(),
        weight_shape: [256, 256, 3, 3].into(),
        bias_shape: [256].into(),
        options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let conv12 = Conv2dBenchmark::<B> {
        suffix: "input_512x7x7_weight_512x3x3_stride_1",
        input_shape: [batch_size, 512, 7, 7].into(),
        weight_shape: [512, 512, 3, 3].into(),
        bias_shape: [512].into(),
        options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let conv13 = Conv2dBenchmark::<B> {
        suffix: "input_96x224x224_weight_64x1x1_stride_1",
        input_shape: [batch_size, 96, 224, 224].into(),
        weight_shape: [64, 96, 1, 1].into(),
        bias_shape: [64].into(),
        options: ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        device: device.clone(),
    };

    let benches = vec![
        benchmark, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11,
        conv12, conv13,
    ];
    let mut results = Vec::new();

    for bench in benches {
        println!("Running {}", bench.name());
        let result = black_box(run_benchmark(bench));
        results.push(result);
    }

    save::<B>(results, device, feature_name, url, token).unwrap();
}

fn main() {
    backend_comparison::bench_on_backend!();
}
