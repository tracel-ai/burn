use burn_wgpu::WgpuDevice;

use std::marker::PhantomData;

use burn_tensor::{
    benchmark::{run_benchmark, Benchmark},
    Distribution, Shape, Tensor,
};
use derive_new::new;

use burn_wgpu::{
    kernel::matmul::{
        contiguous, contiguous_vectorized, matmul_mem_coalescing_default, matmul_naive_default,
        tile, tile_vectorized,
    },
    AutoGraphicsApi, GraphicsApi, WgpuBackend,
};

type WTensor<G, const D: usize> = Tensor<WgpuBackend<G, f32, i32>, D>;

#[derive(new)]
struct MatmulBenchmark<F, const D: usize> {
    shape_lhs: Shape<D>,
    shape_rhs: Shape<D>,
    num_repeats: usize,
    matmul: PhantomData<F>,
}

trait MatmulFunction<G: GraphicsApi, const D: usize> {
    fn run(lhs: WTensor<G, D>, rhs: WTensor<G, D>) -> WTensor<G, D>;
}

impl<F, const D: usize, G> Benchmark<WgpuBackend<G, f32, i32>> for MatmulBenchmark<F, D>
where
    F: MatmulFunction<G, D>,
    G: GraphicsApi,
{
    type Args = (WTensor<G, D>, WTensor<G, D>);

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
        let lhs = WTensor::random_device(self.shape_lhs.clone(), Distribution::Default, device);
        let rhs = WTensor::random_device(self.shape_rhs.clone(), Distribution::Default, device);

        (lhs, rhs)
    }
}

macro_rules! bench_matmul {
    ($benchmark:ident, $matmul_name:ident, $func:expr) => {
        struct $matmul_name {}
        impl<G: GraphicsApi, const D: usize> MatmulFunction<G, D> for $matmul_name {
            fn run(lhs: WTensor<G, D>, rhs: WTensor<G, D>) -> WTensor<G, D> {
                Tensor::from_primitive($func(lhs.into_primitive(), rhs.into_primitive()))
            }
        }
        type $benchmark<const D: usize> = MatmulBenchmark<$matmul_name, D>;
    };
}

bench_matmul!(NaiveMatmulBenchmark, NaiveMatmul, matmul_naive_default);
bench_matmul!(
    MemCoalescingMatmulBenchmark,
    MemCoalescingMatmul,
    matmul_mem_coalescing_default
);
bench_matmul!(
    Tiling2DMatmulContiguousBenchmark,
    Tiling2DMatmulContiguous,
    contiguous::matmul_tiling_2d_default
);
bench_matmul!(
    Tiling2DMatmulTileBenchmark,
    Tiling2DMatmulTile,
    tile::matmul_tiling_2d_default
);
bench_matmul!(
    Tiling2DMatmulTileVectorizedBenchmark,
    Tiling2DMatmulTileVectorized,
    tile_vectorized::matmul_tiling_2d_default
);
bench_matmul!(
    Tiling2DMatmulContiguousVectorizedBenchmark,
    Tiling2DMatmulContiguousVectorized,
    contiguous_vectorized::matmul_tiling_2d_default
);

#[allow(dead_code)]
/// Runs the benchmarks for wgpu matmul implementations
pub fn bench(device: &WgpuDevice) {
    const D: usize = 3;
    let num_repeats = 3;
    let batch_size = 3;
    let m = 1024;
    let k = 2048;
    let n = 1024;
    let shape_lhs = Shape::new([batch_size, m, k]);
    let shape_rhs = Shape::new([batch_size, k, n]);

    macro_rules! run_matmul_benchmark {
        ($benchmark:ident) => {
            run_benchmark::<WgpuBackend<AutoGraphicsApi, f32, i32>, $benchmark<D>>(
                $benchmark::new(shape_lhs.clone(), shape_rhs.clone(), num_repeats),
                device,
            );
        };
    }
    run_matmul_benchmark!(NaiveMatmulBenchmark);
    run_matmul_benchmark!(MemCoalescingMatmulBenchmark);
    run_matmul_benchmark!(Tiling2DMatmulContiguousBenchmark);
    run_matmul_benchmark!(Tiling2DMatmulTileBenchmark);
    run_matmul_benchmark!(Tiling2DMatmulTileVectorizedBenchmark);
    run_matmul_benchmark!(Tiling2DMatmulContiguousVectorizedBenchmark);
}

fn main() {
    bench(&WgpuDevice::BestAvailable)
}
