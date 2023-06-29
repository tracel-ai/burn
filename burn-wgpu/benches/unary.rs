use burn_tensor::{Distribution, Shape, Tensor};
use burn_wgpu::{benchmark::Benchmark, GraphicsApi, OpenGl, Vulkan, WgpuBackend, WgpuDevice};

struct UnaryBenchmark<const D: usize> {
    device: WgpuDevice,
    inplace: bool,
    shape: Shape<D>,
}

impl<const D: usize, G: GraphicsApi> Benchmark<G> for UnaryBenchmark<D> {
    type Args = Tensor<WgpuBackend<G, f32, i32>, D>;

    fn execute(&self, args: Self::Args) {
        if self.inplace {
            args.log();
        } else {
            #[allow(clippy::redundant_clone)]
            args.clone().log();
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
        shape: [32, 256, 256].into(),
    };

    let durations_inplace_opengl = Benchmark::<OpenGl>::run(&benchmark, 200);
    let durations_inplace_vulkan = Benchmark::<Vulkan>::run(&benchmark, 200);

    benchmark.inplace = false;

    let durations_opengl = Benchmark::<OpenGl>::run(&benchmark, 200);
    let durations_vulkan = Benchmark::<Vulkan>::run(&benchmark, 200);

    println!("Vulkan {}", durations_vulkan);
    println!("Vulkan inplace {}", durations_inplace_vulkan);

    println!("OpenGL {}", durations_opengl);
    println!("OpenGL inplace {}", durations_inplace_opengl);
}
