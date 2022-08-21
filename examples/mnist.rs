use burn::data::dataset::source::huggingface::MNISTDataset;
use burn::data::dataset::Dataset;
use burn::module::{Forward, Module, Param};
use burn::nn;
use burn::optim::SGDOptimizer;
use burn::tensor::af::relu;
use burn::tensor::back::{ad, Backend};
use burn::tensor::losses::cross_entropy_with_logits;
use burn::tensor::{Data, Distribution, Shape, Tensor};
use num_traits::FromPrimitive;

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

fn run<B: ad::Backend>(device: B::Device) {
    let mut model: Model<B> = Model::new(784, 2024, 4, 10);
    model.to_device(device);

    let mut optim: SGDOptimizer<B> = SGDOptimizer::new(2.5e-2);
    let dataset = MNISTDataset::train();

    let mut batch = Vec::new();

    for epoch in 0..20 {
        for item in dataset.iter() {
            let data: Data<f32, 2> = Data::from(item.image);
            let input = Tensor::<B, 2>::from_data(data.from_f32()).to_device(device);
            let input = input
                .reshape(Shape::new([1, 784]))
                .div_scalar(&B::Elem::from_f32(255.0).unwrap());
            let targets = Tensor::<B, 2>::zeros(Shape::new([1, 10]));
            let targets = targets
                .index_assign(
                    [0..1, item.label..(item.label + 1)],
                    &Tensor::ones(Shape::new([1, 1])),
                )
                .to_device(device);
            let item = (input, targets);
            batch.push(item);

            if batch.len() < 128 {
                continue;
            }

            let inputs: Vec<Tensor<B, 2>> = batch.iter().map(|b| b.0.clone()).collect();
            let inputs = inputs.iter().collect();
            let inputs = Tensor::cat(inputs, 0);

            let targets: Vec<Tensor<B, 2>> = batch.iter().map(|b| b.1.clone()).collect();
            let targets = targets.iter().collect();
            let targets = Tensor::cat(targets, 0);

            let output = model.forward(inputs);
            let loss = cross_entropy_with_logits(&output, &targets);
            let grads = loss.backward();

            model.update_params(&grads, &mut optim);

            println!("Epoch {}; loss {}", epoch, loss.to_data());

            batch.clear();
        }
    }
}

fn main() {
    let device = burn::tensor::back::TchDevice::Cuda(0);
    run::<ad::Tch<f32>>(device);
}
