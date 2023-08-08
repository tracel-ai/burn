use burn_tensor::{Distribution, Shape, Tensor};

use crate::{
    benchmark::Benchmark,
    element::{FloatElement, WgpuElement},
    kernel,
    tensor::WgpuTensor,
    tune::{AutoTuneFunction, AutoTuneKey, Execution, KernelFunction, Tunable},
    GraphicsApi, WgpuBackend, WgpuDevice,
};
use std::{marker::PhantomData, sync::Arc};

macro_rules! tiling2d_tunable {
    ($name:ident, $func:expr) => {
        #[derive(new, Default)]
        struct $name<E: WgpuElement, const D: usize> {
            b_m: usize,
            b_n: usize,
            b_k: usize,
            t_m: usize,
            t_n: usize,
            workgroup_size_x: usize,
            workgroup_size_y: usize,
            _elem: PhantomData<E>,
        }

        impl<E: WgpuElement, const D: usize> KernelFunction for $name<E, D> {
            type Input = (WgpuTensor<E, D>, WgpuTensor<E, D>);
            type Output = WgpuTensor<E, D>;

            fn call(&self, (lhs, rhs): Self::Input) -> Self::Output {
                $func(
                    lhs,
                    rhs,
                    self.b_m,
                    self.b_n,
                    self.b_k,
                    self.t_m,
                    self.t_n,
                    self.workgroup_size_x,
                    self.workgroup_size_y,
                )
            }
        }
    };
}

tiling2d_tunable!(
    Tiling2DContiguousLoad,
    kernel::matmul::contiguous::matmul_tiling_2d
);

tiling2d_tunable!(Tiling2DTileLoad, kernel::matmul::tile::matmul_tiling_2d);
tiling2d_tunable!(
    Tiling2DContiguousLoadVectorized,
    kernel::matmul::contiguous_vectorized::matmul_tiling_2d
);

tiling2d_tunable!(
    Tiling2DTileLoadVectorized,
    kernel::matmul::tile_vectorized::matmul_tiling_2d
);

#[derive(new)]
struct MatmulBenchmark<F: WgpuElement, const D: usize> {
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

/// Choose the best matmul kernel by using autotuning.
pub fn tune<G: GraphicsApi, E, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<E, D>
where
    E: WgpuElement + FloatElement,
{
    // TODO interpolation
    let id = AutoTuneKey::new(
        vec![Vec::from(lhs.shape.dims), Vec::from(rhs.shape.dims)],
        "matmul".to_string(),
    );
    let context = lhs.context.clone();

    match context.tuner.execute(&id, (lhs, rhs)) {
        Execution::Executed(output) => return output,
        Execution::NoCacheFound((lhs, rhs)) => {
            let mut lhs_shape = [1; D];
            let mut rhs_shape = [1; D];

            lhs_shape[D - 2] = 512;
            lhs_shape[D - 1] = 512;
            rhs_shape[D - 2] = 512;
            rhs_shape[D - 1] = 512;

            let lhs_shape = Shape::new(lhs_shape);
            let rhs_shape = Shape::new(rhs_shape);

            let matmul_benchmark =
                |func: AutoTuneFunction<(WgpuTensor<E, D>, WgpuTensor<E, D>), WgpuTensor<E, D>>| {
                    Tunable::<G, _, _>::new(
                        func.clone(),
                        Arc::new(MatmulBenchmark::new(
                            lhs_shape.clone(),
                            rhs_shape.clone(),
                            5, // number of samples
                            func.clone(),
                        )),
                    )
                };

            let mut tunables = Vec::new();

            for block_size in [64, 128] {
                for block_size_k in [16, 32] {
                    for tile_size in [4, 8] {
                        tunables.push(matmul_benchmark(Arc::new(
                            Tiling2DContiguousLoad::<E, D>::new(
                                block_size,
                                block_size,
                                block_size_k,
                                tile_size,
                                tile_size,
                                block_size / tile_size,
                                block_size / tile_size,
                            ),
                        )));
                        tunables.push(matmul_benchmark(Arc::new(
                            Tiling2DContiguousLoadVectorized::<E, D>::new(
                                block_size,
                                block_size,
                                block_size_k,
                                tile_size,
                                tile_size,
                                block_size / tile_size,
                                block_size / tile_size,
                            ),
                        )));
                        tunables.push(matmul_benchmark(Arc::new(Tiling2DTileLoad::<E, D>::new(
                            block_size,
                            block_size,
                            block_size_k,
                            tile_size,
                            tile_size,
                            block_size / tile_size,
                            block_size / tile_size,
                        ))));
                        tunables.push(matmul_benchmark(Arc::new(
                            Tiling2DTileLoadVectorized::<E, D>::new(
                                block_size,
                                block_size,
                                block_size_k,
                                tile_size,
                                tile_size,
                                block_size / tile_size,
                                block_size / tile_size,
                            ),
                        )));
                    }
                }
            }

            context
                .tuner
                .tune(id, (lhs, rhs), tunables, &context.device)
        }
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::{backend::Backend, Distribution, Shape, Tensor};

    use crate::{tests::TestBackend, WgpuDevice};

    #[test]
    fn assert_matmul_works() {
        TestBackend::seed(0);
        let shape: Shape<2> = [40, 40].into();
        let device = WgpuDevice::default();

        let tensor_1 =
            Tensor::<TestBackend, 2>::random_device(shape.clone(), Distribution::Default, &device);
        let tensor_2 =
            Tensor::<TestBackend, 2>::random_device(shape.clone(), Distribution::Default, &device);
        tensor_1.matmul(tensor_2);
    }
}
