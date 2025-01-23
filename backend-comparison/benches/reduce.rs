use backend_comparison::persistence::save;
use burn::tensor::{backend::Backend, Distribution, Shape, Tensor};
use burn_common::benchmark::{run_benchmark, Benchmark};

enum Instruction {
    ArgMin(usize),
    SumDim(usize),
    Sum,
}

struct ReduceBenchmark<B: Backend> {
    instruction: Instruction,
    shape: Shape,
    device: B::Device,
    tensor: Tensor<B, 3>,
}

impl<B: Backend> ReduceBenchmark<B> {
    pub fn new(instruction: Instruction, device: B::Device) -> Self {
        let shape = Shape::new([4096, 512, 64]);
        // let shape = Shape::new([128, 128, 64]);
        let tensor = Tensor::random(shape.clone(), Distribution::Default, &device);
        Self {
            instruction,
            shape,
            device,
            tensor,
        }
    }
}

impl<B: Backend> Benchmark for ReduceBenchmark<B> {
    type Args = ();

    fn prepare(&self) -> Self::Args {}

    fn execute(&self, _: Self::Args) {
        match self.instruction {
            Instruction::ArgMin(axis) => {
                self.tensor.clone().argmin(axis);
            }
            Instruction::SumDim(axis) => {
                self.tensor.clone().sum_dim(axis);
            }
            Instruction::Sum => {
                self.tensor.clone().sum();
            }
        }
    }

    fn name(&self) -> String {
        match self.instruction {
            Instruction::ArgMin(axis) => format!("reduce-argmin-{axis}"),
            Instruction::SumDim(axis) => format!("reduce-sum-{axis}"),
            Instruction::Sum => String::from("reduce-sum-full"),
        }
    }

    fn sync(&self) {
        B::sync(&self.device)
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![self.shape.dims.clone()]
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(
    device: &B::Device,
    feature_name: &str,
    url: Option<&str>,
    token: Option<&str>,
) {
    let mut benchmarks = Vec::new();

    for axis in 0..3 {
        benchmarks.push(ReduceBenchmark::<B>::new(
            Instruction::ArgMin(axis),
            device.clone(),
        ));

        benchmarks.push(ReduceBenchmark::<B>::new(
            Instruction::SumDim(axis),
            device.clone(),
        ));
    }

    benchmarks.push(ReduceBenchmark::<B>::new(Instruction::Sum, device.clone()));

    save::<B>(
        benchmarks.into_iter().map(run_benchmark).collect(),
        device,
        feature_name,
        url,
        token,
    )
    .unwrap();
}

fn main() {
    backend_comparison::bench_on_backend!();
}
