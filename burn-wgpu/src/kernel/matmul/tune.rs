use burn_tensor::{Distribution, Shape, Tensor};
use mem_coalescing::matmul_mem_coalescing;

use crate::{
    benchmark::Benchmark,
    element::{FloatElement, WgpuElement},
    kernel,
    tensor::WgpuTensor,
    tune::{AutoTuneFunction, AutoTuneKey, Execution, KernelFunction, Tunable},
    GraphicsApi, WgpuBackend, WgpuDevice,
};
use std::{marker::PhantomData, sync::Arc};

use super::mem_coalescing;

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
struct MemoryCoalescing<E: WgpuElement, const D: usize> {
    workgroup_size_x: usize,
    workgroup_size_y: usize,
    _elem: PhantomData<E>,
}

impl<E: WgpuElement, const D: usize> KernelFunction for MemoryCoalescing<E, D> {
    type Input = (WgpuTensor<E, D>, WgpuTensor<E, D>);
    type Output = WgpuTensor<E, D>;

    fn call(&self, (lhs, rhs): Self::Input) -> Self::Output {
        matmul_mem_coalescing::<E, D>(lhs, rhs, self.workgroup_size_x, self.workgroup_size_y)
    }
}

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
        5
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
    let (shape_lhs, shape_rhs) = calculate_benchmark_shapes(lhs.shape.clone(), rhs.shape.clone());
    let id = AutoTuneKey::new(
        vec![Vec::from(shape_lhs.dims), Vec::from(shape_rhs.dims)],
        "matmul".to_string(),
    );
    let context = lhs.context.clone();

    match context.tuner.execute(&id, (lhs, rhs)) {
        Execution::Executed(output) => output,
        Execution::NoCacheFound((lhs, rhs)) => {
            let tunables = matmul_candidates::<G, E, D>(shape_lhs, shape_rhs);
            context
                .tuner
                .tune(id, (lhs, rhs), tunables, &context.device)
        }
    }
}

fn calculate_benchmark_shapes<const D: usize>(
    lhs: Shape<D>,
    rhs: Shape<D>,
) -> (Shape<D>, Shape<D>) {
    let anchor = |a| f32::powf(2., f32::min(f32::round(f32::log(a as f32, 2.)), 12.)) as usize;
    let m = anchor(lhs.dims[D - 2]);
    let k = anchor(lhs.dims[D - 1]);
    let n = anchor(rhs.dims[D - 1]);

    let mut lhs_shape = [1; D];
    lhs_shape[D - 2] = m;
    lhs_shape[D - 1] = k;
    let lhs_shape = Shape::new(lhs_shape);

    let mut rhs_shape = [1; D];
    rhs_shape[D - 2] = k;
    rhs_shape[D - 1] = n;
    let rhs_shape = Shape::new(rhs_shape);

    (lhs_shape, rhs_shape)
}

type MatmulTunable<G, E, const D: usize> =
    Tunable<G, (WgpuTensor<E, D>, WgpuTensor<E, D>), WgpuTensor<E, D>>;

fn matmul_candidates<G: GraphicsApi, E, const D: usize>(
    shape_lhs: Shape<D>,
    shape_rhs: Shape<D>,
) -> Vec<MatmulTunable<G, E, D>>
where
    E: WgpuElement + FloatElement,
{
    let matmul_benchmark =
        |func: AutoTuneFunction<(WgpuTensor<E, D>, WgpuTensor<E, D>), WgpuTensor<E, D>>| {
            Tunable::<G, _, _>::new(
                func.clone(),
                Arc::new(MatmulBenchmark::new(
                    shape_lhs.clone(),
                    shape_rhs.clone(),
                    5,
                    func.clone(),
                )),
            )
        };

    let mut candidates = Vec::new();

    for block_size in [64, 128] {
        for block_size_k in [16, 32] {
            for tile_size in [4, 8] {
                candidates.push(matmul_benchmark(Arc::new(
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
                candidates.push(matmul_benchmark(Arc::new(
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
                candidates.push(matmul_benchmark(Arc::new(Tiling2DTileLoad::<E, D>::new(
                    block_size,
                    block_size,
                    block_size_k,
                    tile_size,
                    tile_size,
                    block_size / tile_size,
                    block_size / tile_size,
                ))));
                candidates.push(matmul_benchmark(Arc::new(
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

        for workgroup_size in [8, 16, 32] {
            candidates.push(matmul_benchmark(Arc::new(MemoryCoalescing::new(
                workgroup_size,
                workgroup_size,
            ))));
        }
    }
    candidates
}
