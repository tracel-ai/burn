use burn::{
    module::pipeline::{Pipeline, PipelineLayout, Stage, StageMap},
    nn::{
        Linear, LinearConfig,
        loss::{MseLoss, Reduction},
    },
    optim::{AdamConfig, GradientsParams},
    prelude::*,
    tensor::{Distribution, activation::relu},
};

const FEATURES: usize = 16;
const HIDDEN: usize = 256;
const BLOCKS: usize = 8;
const BATCH: usize = 64;
const STEPS: usize = 200;
const LEARNING_RATE: f64 = 1e-3;

#[derive(Module, Debug)]
pub struct Model {
    input: Linear,
    blocks: Vec<Linear>,
    output: Linear,
}

impl Model {
    pub fn new(device: &Device) -> Self {
        Self {
            input: LinearConfig::new(FEATURES, HIDDEN).init(device),
            blocks: (0..BLOCKS)
                .map(|_| LinearConfig::new(HIDDEN, HIDDEN).init(device))
                .collect(),
            output: LinearConfig::new(HIDDEN, 1).init(device),
        }
    }
}

impl Pipeline for Model {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type Activations = Tensor<2>;

    fn layout(&self) -> PipelineLayout {
        PipelineLayout::new()
            .input(&self.input)
            .blocks(&self.blocks)
            .output(&self.output)
    }

    fn forward_input(&self, features: Tensor<2>) -> Tensor<2> {
        relu(self.input.forward(features))
    }

    fn forward_block(&self, index: usize, hidden: Tensor<2>) -> Tensor<2> {
        relu(self.blocks[index].forward(hidden.clone())) + hidden
    }

    fn forward_output(&self, hidden: Tensor<2>) -> Tensor<2> {
        self.output.forward(hidden)
    }
}

/// The blocks shared out in order, as evenly as the count allows.
pub fn even_stages(devices: &[Device], blocks: usize) -> Vec<Stage> {
    devices
        .iter()
        .enumerate()
        .map(|(index, device)| Stage {
            device: device.clone(),
            blocks: blocks / devices.len() + usize::from(index < blocks % devices.len()),
        })
        .collect()
}

pub fn run(devices: Vec<Device>) {
    assert!(!devices.is_empty(), "no device to place the model on");
    let devices: Vec<Device> = devices.into_iter().map(Device::autodiff).collect();

    let stages = StageMap::new(&even_stages(&devices, BLOCKS));
    let mut model = Model::new(&devices[0]).place(&stages);

    println!("stages:");
    println!(
        "  input    {:>8} params  {:?}",
        model.input.num_params(),
        stages.input
    );
    for (index, device) in stages.blocks.iter().enumerate() {
        let params = model.blocks[index].num_params();
        println!("  block {index:<2} {params:>8} params  {device:?}");
    }
    println!(
        "  output   {:>8} params  {:?}",
        model.output.num_params(),
        stages.output
    );

    let mut optimizer = AdamConfig::new().init();
    let loss_fn = MseLoss::new();
    let input_device = stages.input.clone();
    let output_device = stages.output.clone();
    let teacher = Tensor::<2>::random([FEATURES, 1], Distribution::Normal(0.0, 1.0), &input_device);

    for step in 1..=STEPS {
        let features = Tensor::<2>::random([BATCH, FEATURES], Distribution::Default, &input_device);
        let targets = features
            .clone()
            .matmul(teacher.clone())
            .sin()
            .to_device(&output_device);

        let predictions = model.forward_on(&stages, features);
        let loss = loss_fn.forward(predictions, targets, Reduction::Mean);
        if step == 1 || step % 20 == 0 {
            println!(
                "step {step:>3}: loss {:.5}",
                loss.clone().into_scalar::<f32>()
            );
        }

        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(LEARNING_RATE, model, grads);
    }
}
