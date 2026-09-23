use burn::{
    module::parallel::{DistributedLayeredModel, LayerPlacement},
    nn::loss::{MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams},
    prelude::*,
    tensor::Distribution,
};

use crate::{layered::LayeredModel, model::ModelConfig};

const BATCH: usize = 64;
const STEPS: usize = 200;
const LEARNING_RATE: f64 = 1e-3;
const LOG_EVERY: usize = 20;

/// Train the split model from scratch to fit a fixed random function.
pub fn train(devices: Vec<Device>) {
    assert!(!devices.is_empty(), "no device to place the model on");

    // Every device needs autodiff, since a gradient has to survive the hop back to the layer before.
    let devices: Vec<Device> = devices.into_iter().map(Device::autodiff).collect();
    let config = ModelConfig::new();
    let placement = LayerPlacement::even(&devices, config.blocks);

    let mut model =
        DistributedLayeredModel::new(LayeredModel::new(&config, &placement), &placement);
    model.print_placement(model.placement());

    let teacher = Teacher::new(&config, &placement);
    let mut optimizer = AdamConfig::new().init();
    for step in 1..=STEPS {
        let (features, targets) = teacher.batch(BATCH);
        let loss = MseLoss::new().forward(model.forward(features), targets, Reduction::Mean);
        if step == 1 || step % LOG_EVERY == 0 {
            println!(
                "step {step:>3}: loss {:.5}",
                loss.clone().into_scalar::<f32>()
            );
        }

        // Each parameter is updated on its own device: the optimizer follows the gradients.
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(LEARNING_RATE, model, grads);
    }
}

/// The function the model learns: random features, and the targets a fixed random map gives them.
struct Teacher {
    map: Tensor<2>,
    features: usize,
    output: Device,
}

impl Teacher {
    fn new(config: &ModelConfig, placement: &LayerPlacement) -> Self {
        Self {
            map: Tensor::random(
                [config.features, config.outputs],
                Distribution::Normal(0.0, 1.0),
                &placement.input,
            ),
            features: config.features,
            output: placement.output.clone(),
        }
    }

    /// Features on the input layer's device, and their targets on the output layer's, where the
    /// loss runs.
    fn batch(&self, size: usize) -> (Tensor<2>, Tensor<2>) {
        let features = Tensor::random(
            [size, self.features],
            Distribution::Default,
            &self.map.device(),
        );
        let targets = features
            .clone()
            .matmul(self.map.clone())
            .sin()
            .to_device(&self.output);
        (features, targets)
    }
}
