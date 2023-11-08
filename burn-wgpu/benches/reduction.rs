use burn_common::benchmark::{run_benchmark, Benchmark};
use burn_tensor::backend::Backend;
use burn_tensor::{Distribution, Shape, Tensor};
use burn_wgpu::kernel::{sum_dim, sum_dim_alt};
use burn_wgpu::WgpuDevice;
use burn_wgpu::{AutoGraphicsApi, Wgpu};
use derive_new::new;
use std::marker::PhantomData;

use burn_wgpu::GraphicsApi;

type WTensor<G, const D: usize> = Tensor<Wgpu<G, f32, i32>, D>;

#[derive(new)]
struct ReduceBenchmark<B: Backend, F, const D: usize> {
    shape: Shape<D>,
    dim: usize,
    num_repeats: usize,
    device: B::Device,
    reduce: PhantomData<F>,
}

trait ReduceFunction<G: GraphicsApi, const D: usize> {
    fn run(input: WTensor<G, D>, dim: usize) -> WTensor<G, D>;
}

impl<F, const D: usize, G> Benchmark for ReduceBenchmark<Wgpu<G, f32, i32>, F, D>
where
    F: ReduceFunction<G, D>,
    G: GraphicsApi,
{
    type Args = WTensor<G, D>;

    fn name(&self) -> String {
        format!(
            "{:?} {:?} dim={:?}",
            std::any::type_name::<F>(),
            self.shape.dims,
            self.dim
        )
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn execute(&self, input: Self::Args) {
        for _ in 0..self.num_repeats {
            F::run(input.clone(), self.dim);
        }
    }

    fn prepare(&self) -> Self::Args {
        WTensor::random_device(self.shape.clone(), Distribution::Default, &self.device)
    }

    fn sync(&self) {
        Wgpu::<G, f32, i32>::sync(&self.device)
    }
}

macro_rules! bench_reduce {
    ($benchmark:ident, $reduce_name:ident, $func:expr) => {
        struct $reduce_name {}
        impl<G: GraphicsApi, const D: usize> ReduceFunction<G, D> for $reduce_name {
            fn run(input: WTensor<G, D>, dim: usize) -> WTensor<G, D> {
                Tensor::from_primitive($func(input.into_primitive(), dim))
            }
        }
        type $benchmark<const D: usize> =
            ReduceBenchmark<Wgpu<AutoGraphicsApi, f32, i32>, $reduce_name, D>;
    };
}

bench_reduce!(FormerSumDimBenchmark, FormerSumDim, sum_dim);
bench_reduce!(NewSumDimBenchmark, NewSumDim, sum_dim_alt);

#[allow(dead_code)]
/// Runs the benchmarks for wgpu matmul implementations
pub fn bench(device: &WgpuDevice) {
    const D: usize = 3;
    let num_repeats = 3;
    let shape = Shape::new([5, 20000, 5]);
    // let shape = Shape::new([2000, 2000, 2000]);
    let dim = 1;

    macro_rules! run_reduce_benchmark {
        ($benchmark:ident) => {
            run_benchmark($benchmark::new(
                shape.clone(),
                dim,
                num_repeats,
                device.clone(),
            ));
        };
    }

    run_reduce_benchmark!(NewSumDimBenchmark);
    run_reduce_benchmark!(FormerSumDimBenchmark);
}

fn main() {
    bench(&WgpuDevice::BestAvailable)
}
