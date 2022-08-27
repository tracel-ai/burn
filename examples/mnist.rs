use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::source::huggingface::{MNISTDataset, MNISTItem};
use burn::module::{Forward, Module, Param};
use burn::nn;
use burn::optim::SGDOptimizer;
use burn::tensor::af::relu;
use burn::tensor::back::{ad, Backend};
use burn::tensor::losses::cross_entropy_with_logits;
use burn::tensor::{Data, ElementConversion, Shape, Tensor};
use burn::train::logger::{AsyncLogger, CLILogger};
use burn::train::metric::{AccuracyMetric, CUDAMetric, LossMetric, Metric};
use burn::train::{ClassificationLearner, ClassificationOutput, SupervisedTrainer};
use std::sync::Arc;

#[derive(Module, Debug)]
struct Model<B: Backend> {
    mlp: Param<MLP<B>>,
    input: Param<nn::Linear<B>>,
    output: Param<nn::Linear<B>>,
}

#[derive(Module, Debug)]
struct MLP<B: Backend> {
    linears: Param<Vec<nn::Linear<B>>>,
}

impl<B: Backend> Forward<Tensor<B, 2>, Tensor<B, 2>> for MLP<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        for linear in self.linears.iter() {
            x = linear.forward(x);
            x = relu(&x);
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

impl<B: Backend> MLP<B> {
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
        }
    }
}

impl<B: Backend> Model<B> {
    fn new(d_input: usize, d_hidden: usize, num_layers: usize, num_classes: usize) -> Self {
        let mlp = MLP::new(d_hidden, num_layers);
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

impl<B: ad::Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
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

fn run<B: ad::Backend>(device: B::Device) {
    let batch_size = 128;
    let learning_rate = 5.5e-2;
    let num_epochs = 10;
    let num_workers = 8;
    let num_layers = 4;
    let hidden_dim = 5560;
    let seed = 42;
    let metrics = || -> Vec<Box<dyn Metric<ClassificationOutput<B>>>> {
        vec![
            Box::new(LossMetric::new()),
            Box::new(AccuracyMetric::new()),
            Box::new(CUDAMetric::new()),
        ]
    };

    let mut model: Model<B> = Model::new(784, hidden_dim, num_layers, 10);
    model.to_device(device);
    println!(
        "Training '{}' with {} params on backend {} {:?}",
        model.name(),
        model.num_params(),
        B::name(),
        device,
    );

    let optim: SGDOptimizer<B> = SGDOptimizer::new(learning_rate);
    let batcher = Arc::new(MNISTBatcher::<B> { device });
    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(batch_size)
        .shuffle(seed)
        .num_workers(num_workers)
        .build(Arc::new(MNISTDataset::train()));
    let dataloader_test = DataLoaderBuilder::new(batcher.clone())
        .batch_size(batch_size)
        .num_workers(num_workers)
        .build(Arc::new(MNISTDataset::test()));

    let learner = ClassificationLearner::new(model);

    let logger_train = Box::new(AsyncLogger::new(Box::new(CLILogger::new(
        metrics(),
        "Train".to_string(),
    ))));
    let logger_valid = Box::new(AsyncLogger::new(Box::new(CLILogger::new(
        metrics(),
        "Valid".to_string(),
    ))));
    let logger_test = Box::new(CLILogger::new(metrics(), "Test".to_string()));

    let trainer = SupervisedTrainer::new(
        dataloader_train.clone(),
        dataloader_test.clone(),
        dataloader_test.clone(),
        logger_train,
        logger_valid,
        logger_test,
        learner,
        optim,
    );

    trainer.run(num_epochs);
}

fn main() {
    let device = burn::tensor::back::TchDevice::Cuda(0);
    run::<ad::Tch<burn::tensor::f16>>(device);
}
