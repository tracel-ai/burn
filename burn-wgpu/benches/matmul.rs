use burn_common::benchmark::{run_benchmark, Benchmark};
use burn_tensor::backend::Backend;
use burn_tensor::{Distribution, Shape, Tensor};
use burn_wgpu::kernel::matmul::init_matmul_output;
use burn_wgpu::{kernel::matmul::vec4_primitive, WgpuDevice};
use burn_wgpu::{AutoGraphicsApi, Wgpu};
use derive_new::new;
use std::marker::PhantomData;

use burn_wgpu::{
    kernel::matmul::{
        contiguous, contiguous_vectorized, matmul_mem_coalescing_default, matmul_naive_default,
        tile, tile_vectorized,
    },
    GraphicsApi,
};

type WTensor<G, const D: usize> = Tensor<Wgpu<G, f32, i32>, D>;

#[derive(new)]
struct MatmulBenchmark<B: Backend, F, const D: usize> {
    shape_lhs: Shape<D>,
    shape_rhs: Shape<D>,
    num_repeats: usize,
    device: B::Device,
    matmul: PhantomData<F>,
}

trait MatmulFunction<G: GraphicsApi, const D: usize> {
    fn run(lhs: WTensor<G, D>, rhs: WTensor<G, D>) -> WTensor<G, D>;
}

impl<F, const D: usize, G> Benchmark for MatmulBenchmark<Wgpu<G, f32, i32>, F, D>
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

    fn prepare(&self) -> Self::Args {
        let lhs =
            WTensor::random_device(self.shape_lhs.clone(), Distribution::Default, &self.device);
        let rhs =
            WTensor::random_device(self.shape_rhs.clone(), Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        Wgpu::<G, f32, i32>::sync(&self.device)
    }
}

macro_rules! bench_matmul {
    ($benchmark:ident, $matmul_name:ident, $func:expr) => {
        struct $matmul_name {}
        impl<G: GraphicsApi, const D: usize> MatmulFunction<G, D> for $matmul_name {
            fn run(lhs: WTensor<G, D>, rhs: WTensor<G, D>) -> WTensor<G, D> {
                let lhs = lhs.into_primitive();
                let rhs = rhs.into_primitive();
                let output = init_matmul_output(&lhs, &rhs);
                Tensor::from_primitive($func(lhs, rhs, output))
            }
        }
        type $benchmark<const D: usize> =
            MatmulBenchmark<Wgpu<AutoGraphicsApi, f32, i32>, $matmul_name, D>;
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
bench_matmul!(
    Tiling2DMatmulVec4PrimitiveBenchmark,
    Tiling2DMatmulVec4Primitive,
    vec4_primitive::matmul_tiling_2d_vec4_primitive_default
);

#[allow(dead_code)]
/// Runs the benchmarks for wgpu matmul implementations
pub fn bench(device: &WgpuDevice) {
    const D: usize = 3;
    let num_repeats = 3;
    let batch_size = 3;
    let m = 2048;
    let k = 2048;
    let n = 1024;
    let shape_lhs = Shape::new([batch_size, m, k]);
    let shape_rhs = Shape::new([batch_size, k, n]);

    macro_rules! run_matmul_benchmark {
        ($benchmark:ident) => {
            run_benchmark($benchmark::new(
                shape_lhs.clone(),
                shape_rhs.clone(),
                num_repeats,
                device.clone(),
            ));
        };
    }
    run_matmul_benchmark!(NaiveMatmulBenchmark);
    run_matmul_benchmark!(MemCoalescingMatmulBenchmark);
    run_matmul_benchmark!(Tiling2DMatmulContiguousBenchmark);
    run_matmul_benchmark!(Tiling2DMatmulTileBenchmark);
    run_matmul_benchmark!(Tiling2DMatmulTileVectorizedBenchmark);
    run_matmul_benchmark!(Tiling2DMatmulContiguousVectorizedBenchmark);
    run_matmul_benchmark!(Tiling2DMatmulVec4PrimitiveBenchmark);
}

fn main() {
    bench(&WgpuDevice::BestAvailable)
}
