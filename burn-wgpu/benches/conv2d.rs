use burn_tensor::{backend::Backend, ops::ConvOptions, Distribution, Shape, Tensor};
use burn_wgpu::{
    benchmark::Benchmark,
    kernel::conv::{tune, conv2d},
    run_benchmark, GraphicsApi, WgpuBackend, WgpuDevice,
};
use std::marker::PhantomData;

trait Conv2dFunction<B: Backend> {
    fn run(
        input: Tensor<B, 4>,
        weight: Tensor<B, 4>,
        bias: Option<Tensor<B, 1>>,
        options: ConvOptions<2>,
    ) -> Tensor<B, 4>;
}

struct Conv2dBenchmark<E> {
    shape_input: Shape<4>,
    shape_weight: Shape<4>,
    // bias: Option<Tensor<E, 1>>, // no bias for now
    options: ConvOptions<2>,
    num_repeats: usize,
    conv2d: PhantomData<E>,
}

impl<F, G> Benchmark<G> for Conv2dBenchmark<F>
where
    F: Conv2dFunction<WgpuBackend<G, f32, i32>>,
    G: GraphicsApi,
{
    type Args = (
        Tensor<WgpuBackend<G, f32, i32>, 4>,
        Tensor<WgpuBackend<G, f32, i32>, 4>,
        Option<Tensor<WgpuBackend<G, f32, i32>, 1>>,
        ConvOptions<2>,
    );

    fn name(&self) -> String {
        format!("{:?} Convolution", self.shape_input.dims) // update later
    }

    fn num_samples(&self) -> usize {
        self.num_repeats
    }

    fn execute(&self, (input, weight, bias, conv_options): Self::Args) {
        for _ in 0..self.num_repeats {
            F::run(input.clone(), weight.clone(), None, conv_options.clone());
        }
    }

    fn prepare(&self, device: &WgpuDevice) -> Self::Args {
        let intensor =
            Tensor::random(self.shape_input.clone(), Distribution::Default).to_device(device);
        let wtensor =
            Tensor::random(self.shape_weight.clone(), Distribution::Default).to_device(device);

        (intensor, wtensor, None, self.options.clone())
    }
}

macro_rules! benchmark {
    ($name:ident, $func:expr) => {
        struct $name;

        impl<G: GraphicsApi> Conv2dFunction<WgpuBackend<G, f32, i32>> for $name {
            fn run(
                input: Tensor<WgpuBackend<G, f32, i32>, 4>,
                weight: Tensor<WgpuBackend<G, f32, i32>, 4>,
                bias: Option<Tensor<WgpuBackend<G, f32, i32>, 1>>,
                options: ConvOptions<2>,
            ) -> Tensor<WgpuBackend<G, f32, i32>, 4> {
                Tensor::from_primitive($func(
                    input.into_primitive(),
                    weight.into_primitive(),
                    None,
                    options,
                    None,
                    None
                ))
            }
        }
    };
}

benchmark!(DefaultConv, conv2d);

struct Conv2dAututone;

impl<G: GraphicsApi> Conv2dFunction<WgpuBackend<G, f32, i32>> for Conv2dAututone {
    fn run(
        input: Tensor<WgpuBackend<G, f32, i32>, 4>,
        weight: Tensor<WgpuBackend<G, f32, i32>, 4>,
        bias: Option<Tensor<WgpuBackend<G, f32, i32>, 1>>,
        options: ConvOptions<2>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, 4> {
        Tensor::from_primitive(tune::<G, f32>(
            input.into_primitive(),
            weight.into_primitive(),
            None,
            options,
        ))
    }
}

fn main() {
    let num_repeats = 5;
    let batch_size = 32; // Decreasing from 32 may provide insight
    let input_channels = 3; //3 is RGB, but 32 or 64 may work
    let h = 224;
    let w = 224;
    let output_channels = 64; //64, 128, 256, 512
    let filter_height = 3;
    let filter_width = 3;
    let c_o = ConvOptions {stride: [1,1], padding: [1,1], dilation: [1,1], groups: 1 };

    run_benchmark!(Conv2dBenchmark::<Conv2dAututone>{
        shape_input: [batch_size, input_channels, h, w].into(),
        shape_weight: [output_channels, input_channels, filter_height, filter_width].into(),
        options: c_o.clone(),
        num_repeats,
        conv2d: PhantomData
    });

    run_benchmark!(Conv2dBenchmark::<DefaultConv>{
        shape_input: [batch_size, input_channels, h, w].into(),
        shape_weight: [output_channels, input_channels, filter_height, filter_width].into(),
        options: c_o.clone(),
        num_repeats,
        conv2d: PhantomData
    });
}
