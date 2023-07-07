use std::marker::PhantomData;

use burn_tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_wgpu::{
    benchmark::Benchmark,
    kernel::{
        continuous, continuous_vectorized, matmul_mem_coalescing_default, matmul_naive_default,
        tile, tile_vectorized,
    },
    run_benchmark, GraphicsApi, WgpuBackend, WgpuDevice,
};

trait MatmulFunction<B: Backend, const D: usize> {
    fn run(lhs: Tensor<B, D>, rhs: Tensor<B, D>) -> Tensor<B, D>;
}

struct MatmulBenchmark<F, const D: usize> {
    shape_lhs: Shape<D>,
    shape_rhs: Shape<D>,
    num_repeats: usize,
    matmul: PhantomData<F>,
}

impl<F, const D: usize, G> Benchmark<G> for MatmulBenchmark<F, D>
where
    F: MatmulFunction<WgpuBackend<G, f32, i32>, D>,
    G: GraphicsApi,
{
    type Args = (
        Tensor<WgpuBackend<G, f32, i32>, D>,
        Tensor<WgpuBackend<G, f32, i32>, D>,
    );

    fn name(&self) -> String {
        format!(
            "{:?} {:?} x {:?}",
            std::any::type_name::<F>(),
            self.shape_lhs.dims,
            self.shape_rhs.dims
        )
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        for _ in 0..self.num_repeats {
            F::run(lhs.clone(), rhs.clone());
        }
    }

    fn prepare(&self, device: &WgpuDevice) -> Self::Args {
        let lhs = Tensor::random(self.shape_lhs.clone(), Distribution::Default).to_device(device);
        let rhs = Tensor::random(self.shape_rhs.clone(), Distribution::Default).to_device(device);

        (lhs, rhs)
    }
}

/// TODO MACRO

struct NaiveMatmul;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D> for NaiveMatmul {
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(matmul_naive_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}

struct MemCoalescingMatmul;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for MemCoalescingMatmul
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(matmul_mem_coalescing_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}

struct Tiling2DMatmulContinuous;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for Tiling2DMatmulContinuous
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(continuous::matmul_tiling_2d_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}
struct Tiling2DMatmulTile;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for Tiling2DMatmulTile
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(tile::matmul_tiling_2d_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}

struct Tiling2DMatmulTileVectorized;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for Tiling2DMatmulTileVectorized
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(tile_vectorized::matmul_tiling_2d_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}

struct Tiling2DMatmulContinuousVectorized;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for Tiling2DMatmulContinuousVectorized
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(continuous_vectorized::matmul_tiling_2d_default(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}

fn main() {
    let num_repeats = 3;
    let batch_size = 3;
    let matrix_size = 1000;
    run_benchmark!(MatmulBenchmark::<MemCoalescingMatmul, 3> {
        shape_lhs: [batch_size, matrix_size, matrix_size].into(),
        shape_rhs: [batch_size, matrix_size, matrix_size].into(),
        num_repeats,
        matmul: PhantomData::default()
    });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulContinuous, 3> {
        shape_lhs: [batch_size, matrix_size, matrix_size].into(),
        shape_rhs: [batch_size, matrix_size, matrix_size].into(),
        num_repeats,
        matmul: PhantomData::default()
    });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulContinuousVectorized, 3> {
        shape_lhs: [batch_size, matrix_size, matrix_size].into(),
        shape_rhs: [batch_size, matrix_size, matrix_size].into(),
        num_repeats,
        matmul: PhantomData::default()
    });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulTile, 3> {
        shape_lhs: [batch_size, matrix_size, matrix_size].into(),
        shape_rhs: [batch_size, matrix_size, matrix_size].into(),
        num_repeats,
        matmul: PhantomData::default()
    });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulTileVectorized, 3> {
        shape_lhs: [batch_size, matrix_size, matrix_size].into(),
        shape_rhs: [batch_size, matrix_size, matrix_size].into(),
        num_repeats,
        matmul: PhantomData::default()
    });
}
