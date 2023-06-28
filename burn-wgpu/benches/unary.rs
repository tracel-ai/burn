use burn_tensor::Tensor;
use burn_wgpu::{benchmark::Benchmark, GraphicsApi, Vulkan, WGPUBackend, WgpuDevice};

struct UnaryBenchmark {
    device: WgpuDevice,
}

impl<G: GraphicsApi> Benchmark<G> for UnaryBenchmark {
    type Args = Tensor<WGPUBackend<G, f32, i32>, 2>;

    fn execute(&self, args: Self::Args) {
        // args.clone().add_scalar(5.0);
        args.clone().matmul(args);
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

    let durations = Benchmark::<Vulkan>::run(&benchmark, 10);

    println!("{:?}", durations);
}
