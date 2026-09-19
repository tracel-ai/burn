use burn::{
    module::pipeline::{Pipeline, PipelinePlacement},
    nn::loss::{MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams},
    prelude::*,
    tensor::Distribution,
};

use crate::model::ModelConfig;

const BATCH: usize = 64;
const STEPS: usize = 200;
const LEARNING_RATE: f64 = 1e-3;

pub fn train(devices: Vec<Device>) {
    assert!(!devices.is_empty(), "no device to place the model on");

    // Every stage needs autodiff, since a gradient has to survive the hop back to the stage before.
    let devices: Vec<Device> = devices.into_iter().map(Device::autodiff).collect();
    let config = ModelConfig::new();
    let placement = PipelinePlacement::even(&devices, config.blocks);

    let mut model = config.init(&devices[0]).place(&placement);
    model.print_placement(model.placement());

    let mut optimizer = AdamConfig::new().init();
    let loss_fn = MseLoss::new();
    let teacher = Tensor::random(
        [config.features, config.outputs],
        Distribution::Normal(0.0, 1.0),
        &placement.input,
    );

    for step in 1..=STEPS {
        let features = Tensor::random(
            [BATCH, config.features],
            Distribution::Default,
            &placement.input,
        );
        let targets = features
            .clone()
            .matmul(teacher.clone())
            .sin()
            .to_device(&placement.output);

        let loss = loss_fn.forward(model.forward(features), targets, Reduction::Mean);
        if step == 1 || step % 20 == 0 {
            println!(
                "step {step:>3}: loss {:.5}",
                loss.clone().into_scalar::<f32>()
            );
        }

        // Each parameter is updated on its own stage: the optimizer follows the gradients.
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        model = optimizer.step(LEARNING_RATE, model, grads);
    }
}
