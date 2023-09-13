use burn_tensor::{backend::Backend, Distribution, Shape, Tensor, ops::ConvOptions};
use burn_wgpu::{
    benchmark::Benchmark,
    kernel::conv::{Conv2d, tune},
    run_benchmark, GraphicsApi, WgpuBackend, WgpuDevice,
};
use std::marker::PhantomData;

trait Conv2dFunction<B: Backend, const D: usize> {
    fn run(input: Tensor<B, D>, weight: Tensor<B, D>, bias: Option<Tensor<B, D>>, options: ConvOptions<2>) -> Tensor<B, D>;
}

struct Conv2dBenchmark<E: Backend> {
    input: Tensor<E, 4>,
    weight: Tensor<E, 4>,
    bias: Option<Tensor<E, 1>>,
    options: ConvOptions<2>,
    num_repeats: usize,
    conv2d: PhantomData<E>
}