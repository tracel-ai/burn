use burn_common::benchmark::{run_benchmark, Benchmark};
use burn_tensor::backend::Backend;
use burn_tensor::{Distribution, Shape, Tensor};
use burn_wgpu::kernel::matmul::init_matmul_output;
use burn_wgpu::kernel::matmul::unpadded::matmul_tiling_2d_unpadded;
use burn_wgpu::kernel::matmul::vec4::matmul_tiling_2d_vec4;
use burn_wgpu::kernel::matmul::vec4_lhs::matmul_tiling_2d_vec4_lhs;
use burn_wgpu::WgpuDevice;
use burn_wgpu::{AutoGraphicsApi, Wgpu};
use derive_new::new;
use std::marker::PhantomData;

use burn_wgpu::{
    kernel::matmul::{matmul_mem_coalescing_default, matmul_naive_default},
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
        let lhs = WTensor::random(self.shape_lhs.clone(), Distribution::Default, &self.device);
        let rhs = WTensor::random(self.shape_rhs.clone(), Distribution::Default, &self.device);

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
    Tiling2DMatmulVec4LHSBenchmark,
    Tiling2DMatmulVec4LHS,
    matmul_tiling_2d_vec4_lhs
);
bench_matmul!(
    Tiling2DMatmulVec4Benchmark,
    Tiling2DMatmulVec4,
    matmul_tiling_2d_vec4
);
bench_matmul!(
    Tiling2DMatmulUnpaddedBenchmark,
    Tiling2DMatmulUnpadded,
    matmul_tiling_2d_unpadded
);

#[allow(dead_code)]
/// Runs the benchmarks for wgpu matmul implementations
pub fn bench(device: &WgpuDevice) {
    const D: usize = 3;
    let num_repeats = 3;
    let batch_size = 3;
    let m = 1007;
    let k = 1023;
    let n = 1005;
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
    run_matmul_benchmark!(Tiling2DMatmulUnpaddedBenchmark);
    run_matmul_benchmark!(Tiling2DMatmulVec4LHSBenchmark);
    run_matmul_benchmark!(Tiling2DMatmulVec4Benchmark);
}

fn main() {
    bench(&WgpuDevice::BestAvailable)
}
