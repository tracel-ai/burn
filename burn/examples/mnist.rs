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
use burn::train::logger::{AsyncLogger, CLILogger};
use burn::train::metric::{AccuracyMetric, CUDAMetric, LossMetric};
use burn::train::{ClassificationLearner, ClassificationOutput, SupervisedTrainer};
use std::sync::Arc;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    mlp: Param<Mlp<B>>,
    input: Param<nn::Linear<B>>,
    output: Param<nn::Linear<B>>,
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
    fn new(dim: usize, num_layers: usize) -> Self {
        let mut linears = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let config = nn::LinearConfig {
                d_input: dim,
                d_output: dim,
                bias: true,
            };
            let linear = nn::Linear::new(&config);
            linears.push(linear);
        }

        Self {
            linears: Param::new(linears),
            dropout: nn::Dropout::new(&nn::DropoutConfig { prob: 0.3 }),
            activation: nn::ReLU::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    fn new(d_input: usize, d_hidden: usize, num_layers: usize, num_classes: usize) -> Self {
        let mlp = Mlp::new(d_hidden, num_layers);
        let config_input = nn::LinearConfig {
            d_input,
            d_output: d_hidden,
            bias: true,
        };
        let config_output = nn::LinearConfig {
            d_input: d_hidden,
            d_output: num_classes,
            bias: true,
        };
        let output = nn::Linear::new(&config_output);
        let input = nn::Linear::new(&config_input);

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
    let num_layers = 4;
    let hidden_dim = 1024;
    let seed = 42;

    let state_model = State::<f32>::load("/tmp/mnist_state_model").ok();
    let state_optim = State::<f32>::load("/tmp/mnist_state_optim").ok();

    let mut model = Model::new(784, hidden_dim, num_layers, 10);
    model.to_device(device);

    if let Some(state) = state_model {
        println!("Loading model state");
        model.load(&state.convert()).unwrap();
    }

    let mut optim = Sgd::new(&SgdConfig {
        learning_rate: 2.5e-2,
        weight_decay: Some(WeightDecayConfig { penalty: 0.01 }),
        momentum: Some(MomentumConfig {
            momentum: 0.9,
            dampening: 0.1,
            nesterov: true,
        }),
    });

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
