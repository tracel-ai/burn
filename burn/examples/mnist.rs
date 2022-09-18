use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::source::huggingface::{MNISTDataset, MNISTItem};
use burn::module::{Forward, Module, Param, State};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::momentum::MomentumConfig;
use burn::optim::{Optimizer, Sgd, SgdConfig};
use burn::tensor::backend::{ADBackend, Backend};
use burn::tensor::loss::cross_entropy_with_logits;
use burn::tensor::{Data, ElementConversion, Shape, Tensor};
use burn::train::logger::{AsyncLogger, CLILogger};
use burn::train::metric::{AccuracyMetric, CUDAMetric, LossMetric};
use burn::train::{ClassificationLearner, ClassificationOutput, SupervisedTrainer};
use burn::{config, nn};
use std::sync::Arc;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    mlp: Param<Mlp<B>>,
    input: Param<nn::Linear<B>>,
    output: Param<nn::Linear<B>>,
}

config!(
    struct MlpConfig {
        #[config(default = 4)]
        num_layers: usize,
        #[config(default = 0.2)]
        dropout: f64,
        #[config(default = 1024)]
        dim: usize,
    }
);

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
    fn new(d_input: usize, num_classes: usize) -> Self {
        let mlp_config = MlpConfig::new();
        let mlp = Mlp::new(&mlp_config);
        let output = nn::Linear::new(&nn::LinearConfig::new(mlp_config.dim, num_classes));
        let input = nn::Linear::new(&nn::LinearConfig::new(d_input, mlp_config.dim));

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
    let batch_size = 128;
    let num_epochs = 15;
    let num_workers = 8;
    let seed = 42;

    let state_model = State::<f32>::load("/tmp/mnist_state_model").ok();
    let state_optim = State::<f32>::load("/tmp/mnist_state_optim").ok();

    let mut model = Model::new(784, 10);
    model.to_device(device);

    if let Some(state) = state_model {
        println!("Loading model state");
        model.load(&state.convert()).unwrap();
    }

    let optim_config = SgdConfig::new()
        .with_learning_rate(2.5e-2)
        .with_weight_decay(Some(WeightDecayConfig::new(0.05)))
        .with_momentum(Some(MomentumConfig::new().with_nesterov(true)));

    let mut optim = Sgd::new(&optim_config);
    if let Some(state) = state_optim {
        println!("Loading optimizer state");
        optim.load(&model, &state.convert()).unwrap();
    }

    println!(
        "Training '{}' with {} params on backend {} {:?}",
        model.name(),
        model.num_params(),
        B::name(),
        device,
    );

    let batcher_train = Arc::new(MNISTBatcher::<B> { device });
    let batcher_valid = Arc::new(MNISTBatcher::<B::InnerBackend> { device });
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(num_workers)
        .build(Arc::new(MNISTDataset::train()));
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(batch_size)
        .num_workers(num_workers)
        .build(Arc::new(MNISTDataset::test()));

    let learner = ClassificationLearner::new(model, optim);

    let logger_train = Box::new(AsyncLogger::new(Box::new(CLILogger::new(
        vec![
            Box::new(LossMetric::new()),
            Box::new(AccuracyMetric::new()),
            Box::new(CUDAMetric::new()),
        ],
        "Train".to_string(),
    ))));
    let logger_valid = Box::new(AsyncLogger::new(Box::new(CLILogger::new(
        vec![
            Box::new(LossMetric::new()),
            Box::new(AccuracyMetric::new()),
            Box::new(CUDAMetric::new()),
        ],
        "Valid".to_string(),
    ))));

    let trainer = SupervisedTrainer::new(
        dataloader_train.clone(),
        dataloader_test.clone(),
        logger_train,
        logger_valid,
        learner,
    );

    let learned = trainer.run(num_epochs);
    let state_model: State<f32> = learned.model.state().convert();
    let state_optim: State<f32> = learned.optim.state(&learned.model).convert();

    state_model.save("/tmp/mnist_state_model").unwrap();
    state_optim.save("/tmp/mnist_state_optim").unwrap();
}

fn main() {
    use burn::tensor::backend::{TchADBackend, TchDevice};
    use burn::tensor::f16;

    let device = TchDevice::Cuda(0);
    run::<TchADBackend<f16>>(device);
}
