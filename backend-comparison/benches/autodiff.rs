use backend_comparison::persistence::save;
use burn::{
    module::Module,
    nn,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Distribution, Tensor,
    },
};
use burn_common::benchmark::{run_benchmark, Benchmark};

pub struct AutodiffOverheadBenchmark<B: AutodiffBackend> {
    config: nn::LstmConfig,
    lstm: nn::Lstm<B>,
    device: B::Device,
}

impl<B: AutodiffBackend> Benchmark for AutodiffOverheadBenchmark<B> {
    type Args = Tensor<B, 3>;

    fn name(&self) -> String {
        "autodiff_overhead".into()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![]
    }

    fn execute(&self, input: Self::Args) {
        for _ in 0..20 {
            let input = input.clone().detach();
            let mut cell = input.clone();
            let lstm = self.lstm.clone().fork(&input.device());

            for _ in 0..10 {
                let (cells, _) = lstm.forward(input.clone(), None);
                cell = cell + cells;
            }

            cell.backward();
        }
    }

    fn prepare(&self) -> Self::Args {
        let shape = [1, 3, self.config.d_hidden];
        Tensor::random(shape, Distribution::Default, &self.device)
    }

    fn sync(&self) {
        B::sync(&self.device)
    }
}

#[allow(dead_code)]
fn bench<B: Backend>(
    device: &B::Device,
    feature_name: &str,
    url: Option<&str>,
    token: Option<&str>,
) {
    let config = nn::LstmConfig::new(3, 3, true);
    let lstm = config.init(device);
    let benchmark = AutodiffOverheadBenchmark::<burn::backend::Autodiff<B>> {
        lstm,
        config,
        device: device.clone(),
    };

    save::<B>(
        vec![run_benchmark(benchmark)],
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
