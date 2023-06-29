use burn_tensor::Tensor;
use burn_wgpu::{benchmark::Benchmark, GraphicsApi, OpenGl, Vulkan, WGPUBackend, WgpuDevice};

struct UnaryBenchmark {
    device: WgpuDevice,
}

impl<G: GraphicsApi> Benchmark<G> for UnaryBenchmark {
    type Args = Tensor<WGPUBackend<G, f32, i32>, 2>;

    fn execute(&self, args: Self::Args) {
        args.log();
    }

    fn prepare(&self) -> Self::Args {
        Tensor::random([1024, 1024], burn_tensor::Distribution::Standard).to_device(&self.device)
    }

    fn device(&self) -> WgpuDevice {
        self.device.clone()
    }
}

fn main() {
    let benchmark = UnaryBenchmark {
        device: WgpuDevice::DiscreteGpu(0),
    };

    let durations_vulkan = Benchmark::<Vulkan>::run(&benchmark, 100);
    let durations_opengl = Benchmark::<OpenGl>::run(&benchmark, 100);

    println!("Vulkan {}", durations_vulkan);
    println!("OpenGL {}", durations_opengl);
}
