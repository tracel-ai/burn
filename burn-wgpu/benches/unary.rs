use burn_tensor::{Distribution, Shape, Tensor};
use burn_wgpu::{
    benchmark::Benchmark,
    kernel::{unary_default, unary_inplace_default},
    run_benchmark, unary, unary_inplace, GraphicsApi, WgpuBackend, WgpuDevice,
};

unary!(TestKernel, func "log");
unary_inplace!(TestKernelInplace, func "log");

struct UnaryBenchmark<const D: usize> {
    inplace: bool,
    shape: Shape<D>,
    num_repeats: usize,
}

impl<const D: usize, G: GraphicsApi> Benchmark<G> for UnaryBenchmark<D> {
    type Args = Tensor<WgpuBackend<G, f32, i32>, D>;

    fn name(&self) -> String {
        match self.inplace {
            true => "Unary Inplace Ops",
            false => "Unary Ops",
        }
        .into()
    }

    fn execute(&self, args: Self::Args) {
        for _ in 0..self.num_repeats {
            if self.inplace {
                unary_inplace_default::<TestKernelInplace, f32, D>(args.clone().into_primitive());
            } else {
                unary_default::<TestKernel, f32, D>(args.clone().into_primitive());
            }
        }
    }

    fn prepare(&self, device: &WgpuDevice) -> Self::Args {
        Tensor::random(self.shape.clone(), Distribution::Default).to_device(device)
    }
}

fn main() {
    run_benchmark!(UnaryBenchmark::<3> {
        inplace: false,
        shape: [32, 512, 1024].into(),
        num_repeats: 10,
    });
    run_benchmark!(UnaryBenchmark::<3> {
        inplace: true,
        shape: [32, 512, 1024].into(),
        num_repeats: 10,
    });
}
