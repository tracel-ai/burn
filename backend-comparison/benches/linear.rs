use backend_comparison::persistence::Persistence;
use burn::{
    nn::{LinearConfig, LinearTConfig},
    tensor::{backend::Backend, Distribution, Shape, Tensor},
};
use burn_common::benchmark::{run_benchmark, Benchmark};
use derive_new::new;

#[derive(new)]
struct LinearBenchmark<B: Backend, const D: usize> {
    shape: Shape<D>,
    weight_shape: [usize; 2],
    num_repeats: usize,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for LinearBenchmark<B, D> {
    type Args = (Box<dyn Fn(Tensor<B, D>) -> ()>, Tensor<B, D>);

    fn name(&self) -> String {
        "Linear".to_string()
    }

    fn execute(&self, args: Self::Args) {
        for _ in 0..self.num_repeats {
            args.0(args.1.clone())
        }
    }

    fn prepare(&self) -> Self::Args {
        let conf = LinearConfig::new(self.weight_shape[0], self.weight_shape[1]);
        let lin = conf.init();
        let f = Box::new(move |x| {
            lin.forward(x);
        });

        let input = Tensor::random_device(self.shape.clone(), Distribution::Default, &self.device);

        (f, input)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[derive(new)]
struct LinearTBenchmark<B: Backend, const D: usize> {
    shape: Shape<D>,
    weight_shape: [usize; 2],
    num_repeats: usize,
    device: B::Device,
}

impl<B: Backend, const D: usize> Benchmark for LinearTBenchmark<B, D> {
    type Args = (Box<dyn Fn(Tensor<B, D>) -> ()>, Tensor<B, D>);

    fn name(&self) -> String {
        "LinearT".to_string()
    }

    fn execute(&self, args: Self::Args) {
        for _ in 0..self.num_repeats {
            args.0(args.1.clone())
        }
    }

    fn prepare(&self) -> Self::Args {
        let conf = LinearTConfig::new(self.weight_shape[0], self.weight_shape[1]);
        let lint = conf.init();
        let f = Box::new(move |x| {
            lint.forward(x);
        });

        let input = Tensor::random_device(self.shape.clone(), Distribution::Default, &self.device);

        (f, input)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(device: &B::Device) {
    const D: usize = 3;
    let weight_shape = [1024, 1024];
    let shape: Shape<D> = [32, 512, weight_shape[0]].into();
    let num_repeats = 10;

    let lin =
        LinearBenchmark::<B, D>::new(shape.clone(), weight_shape, num_repeats, device.clone());

    let lint =
        LinearTBenchmark::<B, D>::new(shape.clone(), weight_shape, num_repeats, device.clone());

    Persistence::persist::<B>(
        vec![
            run_benchmark(lin),  //
            run_benchmark(lint), //
        ],
        device,
    )
}

fn main() {
    backend_comparison::bench_on_backend!();
}
