use burn_tensor::{Distribution, Shape, Tensor};
use burn_wgpu::{
    benchmark::Benchmark,
    binary_elemwise, binary_elemwise_inplace,
    kernel::{binary_elemwise_default, binary_elemwise_inplace_default},
    GraphicsApi, OpenGl, Vulkan, WgpuBackend, WgpuDevice,
};

binary_elemwise!(TestKernel, "+");
binary_elemwise_inplace!(TestKernelInplace, "+");

struct BinaryBenchmark<const D: usize> {
    device: WgpuDevice,
    inplace: bool,
    shape: Shape<D>,
}

impl<const D: usize, G: GraphicsApi> Benchmark<G> for BinaryBenchmark<D> {
    type Args = (
        Tensor<WgpuBackend<G, f32, i32>, D>,
        Tensor<WgpuBackend<G, f32, i32>, D>,
    );

    fn execute(&self, (lhs, rhs): Self::Args) {
        if self.inplace {
            binary_elemwise_inplace_default::<TestKernelInplace, f32, D>(
                lhs.into_primitive(),
                rhs.into_primitive(),
            );
        } else {
            binary_elemwise_default::<TestKernel, f32, D>(
                lhs.into_primitive(),
                rhs.into_primitive(),
            );
        }
    }

    fn prepare(&self) -> Self::Args {
        let lhs =
            Tensor::random(self.shape.clone(), Distribution::Standard).to_device(&self.device);
        let rhs =
            Tensor::random(self.shape.clone(), Distribution::Standard).to_device(&self.device);

        (lhs, rhs)
    }

    fn device(&self) -> WgpuDevice {
        self.device.clone()
    }
}

fn main() {
    let mut benchmark = BinaryBenchmark::<3> {
        device: WgpuDevice::DiscreteGpu(0),
        inplace: true,
        shape: [32, 512, 1024].into(),
    };

    let durations_inplace_opengl = Benchmark::<OpenGl>::run(&benchmark, 10);
    let durations_inplace_vulkan = Benchmark::<Vulkan>::run(&benchmark, 10);

    benchmark.inplace = false;

    let durations_opengl = Benchmark::<OpenGl>::run(&benchmark, 10);
    let durations_vulkan = Benchmark::<Vulkan>::run(&benchmark, 10);

    println!("Vulkan {}", durations_vulkan);
    println!("Vulkan inplace {}", durations_inplace_vulkan);

    println!("OpenGL {}", durations_opengl);
    println!("OpenGL inplace {}", durations_inplace_opengl);
}
