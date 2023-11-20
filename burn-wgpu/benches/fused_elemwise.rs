use burn_common::benchmark::{run_benchmark, Benchmark};
use burn_fusion::Fusion;
use burn_tensor::backend::Backend;
use burn_tensor::{Distribution, Shape, Tensor};
use burn_wgpu::Wgpu;
use burn_wgpu::WgpuDevice;
use derive_new::new;
use std::marker::PhantomData;

#[derive(new)]
struct ElemWiseBenchmark<B: Backend> {
    shape: Shape<3>,
    device: B::Device,
    repeat: usize,
    _b: PhantomData<B>,
}

impl<B: Backend> Benchmark for ElemWiseBenchmark<B> {
    type Args = (Tensor<B, 3>, Tensor<B, 3>);

    fn name(&self) -> String {
        format!(
            "Backend {} Shape {:?} Repeat {}",
            B::name(),
            self.shape.dims,
            self.repeat
        )
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn execute(&self, (lhs, rhs): Self::Args) {
        for _ in 0..self.repeat {
            let tmp_0 = lhs.clone() + rhs.clone();
            let tmp_1 = rhs.clone() * tmp_0.clone();
            let tmp_2 = rhs.clone().exp();
            let tmp_3 = tmp_0 * tmp_1;
            let _tmp_4 = tmp_2 / tmp_3;
        }
    }

    fn prepare(&self) -> Self::Args {
        B::seed(10);
        let lhs = Tensor::random_device(self.shape.clone(), Distribution::Default, &self.device);
        let rhs = Tensor::random_device(self.shape.clone(), Distribution::Default, &self.device);

        (lhs, rhs)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
/// Runs the benchmarks for wgpu matmul implementations
pub fn bench(device: &WgpuDevice) {
    run_benchmark(ElemWiseBenchmark::<Wgpu>::new(
        Shape::new([256, 256, 1024]),
        device.clone(),
        10,
    ));
    run_benchmark(ElemWiseBenchmark::<Fusion<Wgpu>>::new(
        Shape::new([256, 256, 1024]),
        device.clone(),
        10,
    ));
}

fn main() {
    bench(&WgpuDevice::BestAvailable)
}
