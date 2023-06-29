use burn_tensor::{Distribution, Shape, Tensor};
use burn_wgpu::{
    benchmark::Benchmark,
    kernel::{unary_default, unary_inplace_default},
    unary, unary_inplace, GraphicsApi, OpenGl, Vulkan, WgpuBackend, WgpuDevice,
};

unary!(TestKernel, func "log");
unary_inplace!(TestKernelInplace, func "log");

struct UnaryBenchmark<const D: usize> {
    device: WgpuDevice,
    inplace: bool,
    shape: Shape<D>,
}

impl<const D: usize, G: GraphicsApi> Benchmark<G> for UnaryBenchmark<D> {
    type Args = Tensor<WgpuBackend<G, f32, i32>, D>;

    fn execute(&self, args: Self::Args) {
        if self.inplace {
            unary_inplace_default::<TestKernelInplace, f32, D>(args.into_primitive());
        } else {
            unary_default::<TestKernel, f32, D>(args.into_primitive());
        }
    }

    fn prepare(&self) -> Self::Args {
        Tensor::random(self.shape.clone(), Distribution::Standard).to_device(&self.device)
    }

    fn device(&self) -> WgpuDevice {
        self.device.clone()
    }
}

fn main() {
    let mut benchmark = UnaryBenchmark::<3> {
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
