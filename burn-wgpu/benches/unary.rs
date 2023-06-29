use burn_tensor::{Distribution, Shape, Tensor};
use burn_wgpu::{
    benchmark::Benchmark,
    kernel::{unary_default, unary_inplace_default},
    unary, unary_inplace, GraphicsApi, WgpuBackend, WgpuDevice,
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
    let num_iter = 10;
    let benchmark_inplace = UnaryBenchmark::<3> {
        device: WgpuDevice::DiscreteGpu(0),
        inplace: true,
        shape: [32, 512, 1024].into(),
    };
    let benchmark = UnaryBenchmark::<3> {
        device: WgpuDevice::DiscreteGpu(0),
        inplace: false,
        shape: [32, 512, 1024].into(),
    };

    println!(
        "OpenGL {}",
        Benchmark::<burn_wgpu::OpenGl>::run(&benchmark, num_iter)
    );
    println!(
        "OpenGL inplace {}",
        Benchmark::<burn_wgpu::OpenGl>::run(&benchmark_inplace, num_iter)
    );

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    {
        println!(
            "Vulkan {}",
            Benchmark::<burn_wgpu::Vulkan>::run(&benchmark, num_iter)
        );
        println!(
            "Vulkan inplace {}",
            Benchmark::<burn_wgpu::Vulkan>::run(&benchmark_inplace, num_iter)
        );
    }

    #[cfg(target_os = "macos")]
    {
        println!(
            "Metal {}",
            Benchmark::<burn_wgpu::Metal>::run(&benchmark, num_iter)
        );
        println!(
            "Metal inplace {}",
            Benchmark::<burn_wgpu::Metal>::run(&benchmark_inplace, num_iter)
        );
    }
}
