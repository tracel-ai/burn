use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::source::huggingface::{MNISTDataset, MNISTItem};
use burn::module::{Forward, Module, Param, State};
use burn::nn;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::momentum::MomentumConfig;
use burn::optim::{Optimizer, Sgd, SgdConfig};
use burn::tensor::backend::{ADBackend, Backend};
use burn::tensor::loss::cross_entropy_with_logits;
use burn::tensor::{Data, ElementConversion, Shape, Tensor};
use burn::train::metric::{AccuracyMetric, CUDAMetric, LossMetric};
use burn::train::{ClassificationLearner, ClassificationOutput, Train};
use burn::train::{SupervisedData, SupervisedTrainerBuilder};
use std::sync::Arc;

static MODEL_STATE_PATH: &str = "/tmp/mnist_state_model.json.gz";
static OPTIMIZER_STATE_PATH: &str = "/tmp/mnist_state_optim.json.gz";
static CONFIG_PATH: &str = "/tmp/mnist_config.yaml";

#[derive(Config)]
struct MnistConfig {
    #[config(default = 15)]
    num_epochs: usize,
    #[config(default = 128)]
    batch_size: usize,
    #[config(default = 8)]
    num_workers: usize,
    #[config(default = 42)]
    seed: u64,
    optimizer: SgdConfig,
    mlp: MlpConfig,
}

#[derive(Module, Debug)]
struct Model<B: Backend> {
    mlp: Param<Mlp<B>>,
    input: Param<nn::Linear<B>>,
    output: Param<nn::Linear<B>>,
}

#[derive(Config)]
struct MlpConfig {
    #[config(default = 6)]
    num_layers: usize,
    #[config(default = 0.5)]
    dropout: f64,
    #[config(default = 1024)]
    dim: usize,
}

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    linears: Param<Vec<nn::Linear<B>>>,
    dropout: nn::Dropout,
    activation: nn::ReLU,
}

impl<B: Backend> Forward<Tensor<B, 2>, Tensor<B, 2>> for Mlp<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        for linear in self.linears.iter() {
            x = linear.forward(x);
            x = self.dropout.forward(x);
            x = self.activation.forward(x);
        }

        x
    }
}

impl<B: Backend> Forward<Tensor<B, 2>, Tensor<B, 2>> for Model<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        x = self.input.forward(x);
        x = self.mlp.forward(x);
        x = self.output.forward(x);

        x
    }
}

impl<B: Backend> Forward<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn forward(&self, item: MNISTBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.images);
        let loss = cross_entropy_with_logits(&output, &targets);

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: Backend> Mlp<B> {
    fn new(config: &MlpConfig) -> Self {
        let mut linears = Vec::with_capacity(config.num_layers);

        for _ in 0..config.num_layers {
            let linear = nn::Linear::new(&nn::LinearConfig::new(config.dim, config.dim));
            linears.push(linear);
        }

        Self {
            linears: Param::new(linears),
            dropout: nn::Dropout::new(&nn::DropoutConfig::new(0.3)),
            activation: nn::ReLU::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    fn new(config: &MnistConfig, d_input: usize, num_classes: usize) -> Self {
        let mlp = Mlp::new(&config.mlp);
        let output = nn::Linear::new(&nn::LinearConfig::new(config.mlp.dim, num_classes));
        let input = nn::Linear::new(&nn::LinearConfig::new(d_input, config.mlp.dim));

        Self {
            mlp: Param::new(mlp),
            output: Param::new(output),
            input: Param::new(input),
        }
    }
}

struct MNISTBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
struct MNISTBatch<B: Backend> {
    images: Tensor<B, 2>,
    targets: Tensor<B, 2>,
}

impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .map(|tensor| tensor.reshape(Shape::new([1, 784])))
            .map(|tensor| tensor.div_scalar(&255.to_elem()))
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 2>::one_hot(item.label, 10))
            .collect();

        let images = Tensor::cat(images, 0).to_device(self.device).detach();
        let targets = Tensor::cat(targets, 0).to_device(self.device).detach();

        MNISTBatch { images, targets }
    }
}

fn run<B: ADBackend>(device: B::Device) {
    // Config
    let config_optimizer = SgdConfig::new()
        .with_learning_rate(2.5e-2)
        .with_weight_decay(Some(WeightDecayConfig::new(0.05)))
        .with_momentum(Some(MomentumConfig::new().with_nesterov(true)));
    let config_mlp = MlpConfig::new();
    let config = MnistConfig::new(config_optimizer, config_mlp);
    B::seed(config.seed);

    // Data
    let batcher_train = Arc::new(MNISTBatcher::<B> { device });
    let batcher_valid = Arc::new(MNISTBatcher::<B::InnerBackend> { device });
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(Arc::new(MNISTDataset::train()));
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(Arc::new(MNISTDataset::test()));
    let data = SupervisedData::new(dataloader_train, dataloader_test);

    // Model
    let optim = Sgd::new(&config.optimizer);
    let mut model = Model::new(&config, 784, 10);
    model.to_device(device);
    let learner = ClassificationLearner::new(model, optim);

    // Training
    let trainer = SupervisedTrainerBuilder::default()
        .metric_train(CUDAMetric::new())
        .metric_train_plot(AccuracyMetric::new())
        .metric_valid_plot(AccuracyMetric::new())
        .metric_train_plot(LossMetric::new())
        .metric_valid_plot(LossMetric::new())
        .num_epochs(config.num_epochs)
        .build();
    let trained = trainer.train(learner, data);

    // Saving
    let state_model: State<f32> = trained.model.state().convert();
    let state_optim: State<f32> = trained.optim.state(&trained.model).convert();

    state_model.save(MODEL_STATE_PATH).unwrap();
    state_optim.save(OPTIMIZER_STATE_PATH).unwrap();
    config.save(CONFIG_PATH).unwrap();
}

fn main() {
    use burn::tensor::backend::{TchADBackend, TchDevice};
    use burn::tensor::f16;

    let device = TchDevice::Cuda(0);
    run::<TchADBackend<f16>>(device);
    println!("Done.");
}
