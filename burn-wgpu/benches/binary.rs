use burn_tensor::{Distribution, Shape, Tensor};
use burn_wgpu::{
    benchmark::Benchmark,
    binary_elemwise, binary_elemwise_inplace,
    kernel::{binary_elemwise_default, binary_elemwise_inplace_default},
    run_benchmark, GraphicsApi, WgpuBackend, WgpuDevice,
};

binary_elemwise!(TestKernel, "+");
binary_elemwise_inplace!(TestKernelInplace, "+");

struct BinaryBenchmark<const D: usize> {
    inplace: bool,
    shape: Shape<D>,
    num_repeats: usize,
}

impl<const D: usize, G: GraphicsApi> Benchmark<G> for BinaryBenchmark<D> {
    type Args = (
        Tensor<WgpuBackend<G, f32, i32>, D>,
        Tensor<WgpuBackend<G, f32, i32>, D>,
    );

    fn name(&self) -> String {
        match self.inplace {
            true => "Binary Inplace Ops",
            false => "Binary Ops",
        }
        .into()
    }
    fn execute(&self, (lhs, rhs): Self::Args) {
        for _ in 0..self.num_repeats {
            if self.inplace {
                binary_elemwise_inplace_default::<TestKernelInplace, f32, D>(
                    lhs.clone().into_primitive(),
                    rhs.clone().into_primitive(),
                );
            } else {
                binary_elemwise_default::<TestKernel, f32, D>(
                    lhs.clone().into_primitive(),
                    rhs.clone().into_primitive(),
                );
            }
        }
    }

    fn prepare(&self, device: &WgpuDevice) -> Self::Args {
        let lhs = Tensor::random(self.shape.clone(), Distribution::Default).to_device(device);
        let rhs = Tensor::random(self.shape.clone(), Distribution::Default).to_device(device);

        (lhs, rhs)
    }
}

fn main() {
    run_benchmark!(BinaryBenchmark::<3> {
        inplace: false,
        shape: [32, 512, 1024].into(),
        num_repeats: 10,
    });
    run_benchmark!(BinaryBenchmark::<3> {
        inplace: true,
        shape: [32, 512, 1024].into(),
        num_repeats: 10,
    });
}
