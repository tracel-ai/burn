use burn_tensor::{Data, Distribution, Shape, Tensor};
use burn_wgpu::{benchmark::Benchmark, run_benchmark, GraphicsApi, WgpuBackend, WgpuDevice};

struct ToDataBenchmark<const D: usize> {
    shape: Shape<D>,
    num_repeats: usize,
}

impl<const D: usize, G: GraphicsApi> Benchmark<G> for ToDataBenchmark<D> {
    type Args = Tensor<WgpuBackend<G, f32, i32>, D>;

    fn name(&self) -> String {
        format!("to-data-{:?}-{}", self.shape.dims, self.num_repeats)
    }

    fn execute(&self, args: Self::Args) {
        for _ in 0..self.num_repeats {
            let _data = args.to_data();
        }
    }

    fn prepare(&self, device: &WgpuDevice) -> Self::Args {
        Tensor::random(self.shape.clone(), Distribution::Default).to_device(device)
    }
}

struct FromDataBenchmark<const D: usize> {
    shape: Shape<D>,
    num_repeats: usize,
}

impl<const D: usize, G: GraphicsApi> Benchmark<G> for FromDataBenchmark<D> {
    type Args = (Data<f32, D>, WgpuDevice);

    fn name(&self) -> String {
        format!("from-data-{:?}-{}", self.shape.dims, self.num_repeats)
    }

    fn execute(&self, (data, device): Self::Args) {
        for _ in 0..self.num_repeats {
            let _data =
                Tensor::<WgpuBackend<G, f32, i32>, D>::from_data_device(data.clone(), &device);
        }
    }

    fn prepare(&self, device: &WgpuDevice) -> Self::Args {
        (
            Data::random(
                self.shape.clone(),
                Distribution::Default,
                &mut rand::thread_rng(),
            ),
            device.clone(),
        )
    }
}

fn main() {
    let num_repeats = 3;

    run_benchmark!(ToDataBenchmark::<3> {
        shape: [32, 256, 512].into(),
        num_repeats,
    });
    run_benchmark!(ToDataBenchmark::<3> {
        shape: [32, 512, 1024].into(),
        num_repeats,
    });
    run_benchmark!(FromDataBenchmark::<3> {
        shape: [32, 256, 512].into(),
        num_repeats,
    });
    run_benchmark!(FromDataBenchmark::<3> {
        shape: [32, 512, 1024].into(),
        num_repeats,
    });
}
