use backend_comparison::persistence::save;
use burn::tensor::backend::Backend;
use burn::tensor::Device;
use burn::{config::Config, module::Module, nn};
use burn_common::benchmark::{run_benchmark, Benchmark};
use derive_new::new;

#[derive(Module, Debug)]
struct BenchmarkModule<B: Backend> {
    linears: Vec<nn::Linear<B>>,
}

#[derive(Config, Debug)]
struct BenchmarkConfig {
    linear: nn::LinearConfig,
    num_layers: usize,
}

impl BenchmarkConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BenchmarkModule<B> {
        BenchmarkModule {
            linears: (0..self.num_layers)
                .map(|_| self.linear.init(device))
                .collect(),
        }
    }
    pub fn init_with<B: Backend>(&self, record: BenchmarkModuleRecord<B>) -> BenchmarkModule<B> {
        BenchmarkModule {
            linears: record
                .linears
                .into_iter()
                .map(|record| nn::Linear {
                    weight: record.weight,
                    bias: record.bias,
                })
                .collect(),
        }
    }
}

#[derive(Debug)]
enum Kind {
    Lazy,
    Sync,
    Manual,
}

#[derive(new)]
struct LoadRecordBenchmark<B: Backend> {
    config: BenchmarkConfig,
    device: Device<B>,
    kind: Kind,
}

impl<B: Backend> Benchmark for LoadRecordBenchmark<B> {
    type Args = BenchmarkModule<B>;

    fn name(&self) -> String {
        format!("load_record_{:?}", self.kind).to_lowercase()
    }

    fn shapes(&self) -> Vec<Vec<usize>> {
        vec![]
    }

    fn num_samples(&self) -> usize {
        10
    }

    fn execute(&self, module: Self::Args) {
        let record = module.into_record();

        let _ = match self.kind {
            Kind::Lazy => {
                let module = self.config.init(&self.device);
                module.load_record(record)
            }
            Kind::Sync => {
                let module = self.config.init(&self.device);
                // Force sync.
                let _ = module.clone();
                module.load_record(record)
            }
            Kind::Manual => self.config.init_with(record),
        };
    }

    fn prepare(&self) -> Self::Args {
        let module = self.config.init(&self.device);
        // Force sync.

        module.clone()
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
    let config = BenchmarkConfig::new(nn::LinearConfig::new(2048, 2048), 12);

    let benchmark_lazy = LoadRecordBenchmark::<B>::new(config.clone(), device.clone(), Kind::Lazy);
    let benchmark_sync = LoadRecordBenchmark::<B>::new(config.clone(), device.clone(), Kind::Sync);
    let benchmark_manual =
        LoadRecordBenchmark::<B>::new(config.clone(), device.clone(), Kind::Manual);

    save::<B>(
        vec![run_benchmark(benchmark_lazy)],
        device,
        feature_name,
        url,
        token,
    )
    .unwrap();
    save::<B>(
        vec![run_benchmark(benchmark_manual)],
        device,
        feature_name,
        url,
        token,
    )
    .unwrap();
    save::<B>(
        vec![run_benchmark(benchmark_sync)],
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
