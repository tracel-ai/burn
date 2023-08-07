use burn_tensor::{Distribution, Shape, Tensor};

use crate::{
    benchmark::Benchmark,
    element::{FloatElement, WgpuElement},
    kernel,
    tensor::WgpuTensor,
    tune::{AutoTuneFunction, KernelFunction},
    GraphicsApi, WgpuBackend, WgpuDevice,
};
use std::marker::PhantomData;

#[derive(Default)]
pub struct MatmulTiling2dTunable<
    E: WgpuElement,
    const D: usize,
    const B_M: usize,
    const B_N: usize,
    const B_K: usize,
    const T_M: usize,
    const T_N: usize,
    const WORKGROUP_SIZE_X: usize,
    const WORKGROUP_SIZE_Y: usize,
> {
    _elem: PhantomData<E>,
}

impl<
        E: WgpuElement,
        const D: usize,
        const B_M: usize,
        const B_N: usize,
        const B_K: usize,
        const T_M: usize,
        const T_N: usize,
        const WORKGROUP_SIZE_X: usize,
        const WORKGROUP_SIZE_Y: usize,
    > KernelFunction
    for MatmulTiling2dTunable<E, D, B_M, B_N, B_K, T_M, T_N, WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y>
{
    type Input = (WgpuTensor<E, D>, WgpuTensor<E, D>);
    type Output = WgpuTensor<E, D>;

    fn call(&self, (lhs, rhs): Self::Input) -> Self::Output {
        kernel::matmul::matmul_tiling_2d::<
            E,
            D,
            B_M,
            B_N,
            B_K,
            T_M,
            T_N,
            WORKGROUP_SIZE_X,
            WORKGROUP_SIZE_Y,
        >(lhs, rhs)
    }
}

#[derive(new)]
pub struct MatmulBenchmark<F: WgpuElement, const D: usize> {
    shape_lhs: Shape<D>,
    shape_rhs: Shape<D>,
    num_repeats: usize,
    matmul: PhantomData<F>,
    func: AutoTuneFunction<(WgpuTensor<F, D>, WgpuTensor<F, D>), WgpuTensor<F, D>>,
}

impl<E, const D: usize, G> Benchmark<G> for MatmulBenchmark<E, D>
where
    E: WgpuElement + FloatElement,
    G: GraphicsApi,
{
    type Args = (WgpuTensor<E, D>, WgpuTensor<E, D>);

    fn name(&self) -> String {
        format!("{:?} x {:?}", self.shape_lhs.dims, self.shape_rhs.dims)
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        for _ in 0..self.num_repeats {
            self.func.call((lhs.clone(), rhs.clone()));
        }
    }

    fn prepare(&self, device: &WgpuDevice) -> Self::Args {
        let lhs = Tensor::<WgpuBackend<G, E, i32>, D>::random(
            self.shape_lhs.clone(),
            Distribution::Default,
        )
        .to_device(device);
        let rhs = Tensor::<WgpuBackend<G, E, i32>, D>::random(
            self.shape_rhs.clone(),
            Distribution::Default,
        )
        .to_device(device);

        (lhs.into_primitive(), rhs.into_primitive())
    }
}
