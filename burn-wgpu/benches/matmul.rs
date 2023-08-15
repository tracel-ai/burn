use burn_tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_wgpu::{
    benchmark::Benchmark,
    kernel::matmul::{
        contiguous, contiguous_vectorized, matmul_mem_coalescing_default, matmul_naive_default,
        tile, tile_vectorized, tune,
    },
    run_benchmark, GraphicsApi, WgpuBackend, WgpuDevice,
};
use std::marker::PhantomData;

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

macro_rules! benchmark {
    ($name:ident, $func:expr) => {
        struct $name;

        impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D> for $name {
            fn run(
                lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
                rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
            ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
                Tensor::from_primitive($func(lhs.into_primitive(), rhs.into_primitive()))
            }
        }
    };
}

benchmark!(NaiveMatmul, matmul_naive_default);
benchmark!(MemCoalescingMatmul, matmul_mem_coalescing_default);
benchmark!(
    Tiling2DMatmulContiguous,
    contiguous::matmul_tiling_2d_default
);
benchmark!(Tiling2DMatmulTile, tile::matmul_tiling_2d_default);
benchmark!(
    Tiling2DMatmulTileVectorized,
    tile_vectorized::matmul_tiling_2d_default
);
benchmark!(
    Tiling2DMatmulContiguousVectorized,
    contiguous_vectorized::matmul_tiling_2d_default
);

struct MatmulAutotune;

impl<const D: usize, G: GraphicsApi> MatmulFunction<WgpuBackend<G, f32, i32>, D>
    for MatmulAutotune
{
    fn run(
        lhs: Tensor<WgpuBackend<G, f32, i32>, D>,
        rhs: Tensor<WgpuBackend<G, f32, i32>, D>,
    ) -> Tensor<WgpuBackend<G, f32, i32>, D> {
        Tensor::from_primitive(tune::<G, f32, D>(
            lhs.into_primitive(),
            rhs.into_primitive(),
        ))
    }
}

fn main() {
    let num_repeats = 3;
    let batch_size = 3;
    let m = 1024;
    let k = 2048;
    let n = 1024;

    run_benchmark!(MatmulBenchmark::<MemCoalescingMatmul, 3> {
        shape_lhs: [batch_size, m, k].into(),
        shape_rhs: [batch_size, k, n].into(),
        num_repeats,
        matmul: PhantomData
    });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulContiguous, 3> {
        shape_lhs: [batch_size, m, k].into(),
        shape_rhs: [batch_size, k, n].into(),
        num_repeats,
        matmul: PhantomData
    });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulContiguousVectorized, 3> {
        shape_lhs: [batch_size, m, k].into(),
        shape_rhs: [batch_size, k, n].into(),
        num_repeats,
        matmul: PhantomData
    });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulTile, 3> {
        shape_lhs: [batch_size, m, k].into(),
        shape_rhs: [batch_size, k, n].into(),
        num_repeats,
        matmul: PhantomData
    });
    run_benchmark!(MatmulBenchmark::<Tiling2DMatmulTileVectorized, 3> {
        shape_lhs: [batch_size, m, k].into(),
        shape_rhs: [batch_size, k, n].into(),
        num_repeats,
        matmul: PhantomData
    });
    run_benchmark!(MatmulBenchmark::<MatmulAutotune, 3> {
        shape_lhs: [batch_size, m, k].into(),
        shape_rhs: [batch_size, k, n].into(),
        num_repeats,
        matmul: PhantomData
    });
}
