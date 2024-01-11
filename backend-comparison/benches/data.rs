use backend_comparison::persistence::save;
use burn::tensor::{backend::Backend, Data, Distribution, Shape, Tensor};
use burn_common::benchmark::{run_benchmark, Benchmark};
use derive_new::new;

#[derive(new)]
struct ToDataBenchmark<B: Backend, const D: usize> {
    shape: Shape<D>,
    num_repeats: usize,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for ToDataBenchmark<B, D> {
    type Args = Tensor<B, D>;

    fn name(&self) -> String {
        "to_data".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec!(self.shape.dims.into())
    }

    fn num_repeats(&self) -> usize {
        self.num_repeats
    }

    fn execute(&self, args: Self::Args) {
        for _ in 0..self.num_repeats() {
            let _data = args.to_data();
        }
    }

    fn prepare(&self) -> Self::Args {
        Tensor::random(self.shape.clone(), Distribution::Default, &self.device)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[derive(new)]
struct FromDataBenchmark<B: Backend, const D: usize> {
    shape: Shape<D>,
    num_repeats: usize,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for FromDataBenchmark<B, D> {
    type Args = (Data<B::FloatElem, D>, B::Device);

    fn name(&self) -> String {
        "from_data".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec!(self.shape.dims.into())
    }

    fn num_repeats(&self) -> usize {
        self.num_repeats
    }

    fn execute(&self, (data, device): Self::Args) {
        for _ in 0..self.num_repeats() {
            let _data = Tensor::<B, D>::from_data(data.clone(), &device);
        }
    }

    fn prepare(&self) -> Self::Args {
        (
            Data::random(
                self.shape.clone(),
                Distribution::Default,
                &mut rand::thread_rng(),
            ),
            self.device.clone(),
        )
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) {
    const D: usize = 3;
    let shape: Shape<D> = [32, 512, 1024].into();
    let num_repeats = 10;

    let to_benchmark = ToDataBenchmark::<B, D>::new(shape.clone(), num_repeats, device.clone());
    let from_benchmark = FromDataBenchmark::<B, D>::new(shape, num_repeats, device.clone());

    save::<B>(
        vec![run_benchmark(to_benchmark), run_benchmark(from_benchmark)],
        device,
    )
    .unwrap();
}

fn main() {
    backend_comparison::bench_on_backend!();
}
