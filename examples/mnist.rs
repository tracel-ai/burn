use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::BasicDataLoader;
use burn::data::dataset::source::huggingface::{MNISTDataset, MNISTItem};
use burn::data::dataset::transform::ShuffledDataset;
use burn::module::{Forward, Module, Param};
use burn::nn;
use burn::optim::SGDOptimizer;
use burn::tensor::af::relu;
use burn::tensor::back::{ad, Backend};
use burn::tensor::losses::cross_entropy_with_logits;
use burn::tensor::{Data, ElementConversion, Shape, Tensor};
use burn::train::logger::{CLILogger, TextPlot};
use burn::train::metric::{AccuracyMetric, CUDAMetric, LossMetric, Metric};
use burn::train::{ClassificationLearner, ClassificationOutput, SupervisedTrainer};
use std::sync::Arc;

#[derive(Module, Debug)]
struct Model<B>
where
    B: Backend,
{
    mlp: Param<MLP<B>>,
    input: Param<nn::Linear<B>>,
    output: Param<nn::Linear<B>>,
}

#[derive(Module, Debug)]
struct MLP<B>
where
    B: Backend,
{
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
        let mut images_list = Vec::with_capacity(items.len());
        let mut targets_list = Vec::with_capacity(items.len());

        for item in items {
            let data: Data<f32, 2> = Data::from(item.image);
            let image = Tensor::<B, 2>::from_data(data.convert());
            let image = image
                .reshape(Shape::new([1, 784]))
                .div_scalar(&255.to_elem());
            let target = Tensor::<B, 2>::zeros(Shape::new([1, 10]));
            let target = target.index_assign(
                [0..1, item.label..(item.label + 1)],
                &Tensor::ones(Shape::new([1, 1])),
            );

            images_list.push(image);
            targets_list.push(target);
        }

        let images = images_list.iter().collect();
        let images = Tensor::cat(images, 0).to_device(self.device).detach();

        let targets = targets_list.iter().collect();
        let targets = Tensor::cat(targets, 0).to_device(self.device).detach();

        MNISTBatch { images, targets }
    }
}

fn run<B: ad::Backend>(device: B::Device) {
    let batch_size = 64;
    let learning_rate = 9.5e-2;
    let num_epochs = 20;
    let num_workers = 8;
    let num_layers = 4;
    let hidden_dim = 1024;
    let seed = 42;
    let metrics = || -> Vec<Box<dyn Metric<ClassificationOutput<B>>>> {
        vec![
            Box::new(TextPlot::new(LossMetric::new())),
            Box::new(TextPlot::new(AccuracyMetric::new())),
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
    let dataset_train = Arc::new(ShuffledDataset::with_seed(
        Arc::new(MNISTDataset::train()),
        seed,
    ));
    let dataset_test = Arc::new(MNISTDataset::test());
    let dataloader_train = Arc::new(BasicDataLoader::multi_thread(
        batch_size,
        dataset_train,
        batcher.clone(),
        num_workers,
    ));
    let dataloader_test = Arc::new(BasicDataLoader::multi_thread(
        batch_size,
        dataset_test,
        batcher.clone(),
        num_workers,
    ));

    let learner = ClassificationLearner::new(model);

    let logger_train = Box::new(CLILogger::new(metrics(), "Train".to_string()));
    let logger_valid = Box::new(CLILogger::new(metrics(), "Valid".to_string()));
    let logger_test = Box::new(CLILogger::new(metrics(), "Test".to_string()));

    let trainer = SupervisedTrainer::new(
        dataloader_train.clone(),
        dataloader_train.clone(),
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
