use burn_tensor::{ops::ConvOptions, Distribution, Shape, Tensor};

use crate::{
    benchmark::Benchmark,
    element::{FloatElement, WgpuElement},
    kernel,
    tensor::{WgpuTensor, WgpuTensorDyn},
    tune::{AutoTuneFunction, AutoTuneKey, Execution, KernelFunction, Tunable},
    GraphicsApi, WgpuBackend, WgpuDevice,
};
use std::{marker::PhantomData, sync::Arc};

const WORKGROUP_SIZES: [usize; 3] = [8, 16, 32];

#[derive(new)]
struct Conv2dBenchmark<E: WgpuElement, const D: usize> {
    input_shape: Shape<D>,
    weight_shape: Shape<D>,
    bias: Option<Shape<D>>,
    options: ConvOptions<2>,
    num_repeats: usize,
    conv2d: PhantomData<E>,
    func: AutoTuneFunction<(WgpuTensorDyn<E>, WgpuTensorDyn<E>, Option<WgpuTensorDyn<E>>, ConvOptions<2>), WgpuTensorDyn<E>>, // input and weight, does bias really affect it?
}

impl<E, const D: usize, G> Benchmark<G> for Conv2dBenchmark<E, D>
where
    E: WgpuElement + FloatElement,
    G: GraphicsApi,
{
    type Args = (WgpuTensorDyn<E>, WgpuTensorDyn<E>, Option<WgpuTensorDyn<E>>, ConvOptions<2>);

    fn name(&self) -> String {
        format!("{:?} Convolution", self.input_shape.dims) // update later
    }

    fn num_samples(&self) -> usize {
        5 // where does this come from?
    }

    fn execute(&self, (input, weight, bias, conv_options): Self::Args) {
        for _ in 0..self.num_repeats {
            self.func.call((input.clone(), weight.clone(), bias.clone(), conv_options.clone()));
        }
    }

    fn prepare(&self, device: &WgpuDevice) -> Self::Args {
        let input_tensor = Tensor::<WgpuBackend<G, E, i32>, D>::random_device(
            self.input_shape.clone(),
            Distribution::Default,
            device,
        );

        let weight_tensor = Tensor::<WgpuBackend<G, E, i32>, D>::random_device(
            self.weight_shape.clone(),
            Distribution::Default,
            device,
        );

        let bias_tensor = self.bias.as_ref().map(|shape| {
            Tensor::<WgpuBackend<G, E, i32>, D>::random_device(
                shape.clone(),
                Distribution::Default,
                device
            ).into_primitive().into()
        });

        (
            input_tensor.into_primitive().into(),
            weight_tensor.into_primitive().into(),
            bias_tensor,
            self.options.clone()
        )
    }
}
