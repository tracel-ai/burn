//! A split model trained as a pipeline updates as one step on the whole batch does.
//!
//! Two fixed CPU backends stand in for two cards. Run with
//! `cargo test -p burn-optim --features flex,cpu --test training_pipeline`.
#![cfg(all(feature = "flex", feature = "cpu"))]

use burn_core as burn;

use burn::module::parallel::{
    DistributedLayer, DistributedLayeredModel, LayerParallelism, LayerPlacement, LayerSequence,
    LayerStage, PipelineSchedule,
};
use burn::module::{Module, Param};
use burn::tensor::{Device, Distribution, Tensor, Tolerance};
use burn_optim::parallel::TrainingPipeline;
use burn_optim::{AdamConfig, GradientsParams, ModuleOptimizer, SgdConfig};

const WIDTH: usize = 6;
const ROWS: usize = 8;
const MICROBATCHES: usize = 4;
const LEARNING_RATE: f64 = 0.05;
const STEPS: usize = 2;

const SCHEDULES: [PipelineSchedule; 2] = [
    PipelineSchedule::GPipe,
    PipelineSchedule::OneForwardOneBackward,
];

fn devices() -> (Device, Device) {
    (Device::flex().autodiff(), Device::cpu().autodiff())
}

fn placement<const N: usize>(stages: [(Device, usize); N]) -> LayerPlacement {
    LayerPlacement::new(&stages.map(|(device, num_hidden_layers)| LayerStage {
        device,
        num_hidden_layers,
    }))
}

fn loss(output: Tensor<2>, target: Tensor<2>) -> Tensor<1> {
    (output - target).powi_scalar(2).mean()
}

/// Trains `placement`'s split for `STEPS` steps and the same model on one device, and checks their
/// weights end up the same.
fn assert_trains_as_one_device(
    placement: &LayerPlacement,
    schedule: PipelineSchedule,
    optimizer: impl Fn() -> ModuleOptimizer,
) {
    let device = placement.input.clone();
    let stack = Stack::new(placement);
    let mut reference = stack.clone().fork(&device);
    let mut reference_optimizer = optimizer();
    let mut training = TrainingPipeline::new(
        DistributedLayeredModel::new(stack, placement),
        schedule,
        optimizer(),
    );

    for _ in 0..STEPS {
        let input = Tensor::random([ROWS, WIDTH], Distribution::Default, &device);
        let target = Tensor::random([ROWS, 2], Distribution::Default, &device);

        let grads = loss(reference.plain_forward(input.clone()), target.clone()).backward();
        let grads = GradientsParams::from_grads(grads, &reference);
        reference = reference_optimizer.step(LEARNING_RATE, reference, grads);

        let losses = training.step(
            input.chunk(MICROBATCHES, 0),
            target.chunk(MICROBATCHES, 0),
            loss,
            LEARNING_RATE,
        );
        assert_eq!(losses.len(), MICROBATCHES);
    }

    let trained = training.into_model().into_inner().fork(&device);
    for ((name, got), (_, want)) in trained.weights().into_iter().zip(reference.weights()) {
        assert_eq!(got.shape(), want.shape(), "{name}");
        got.into_data()
            .assert_approx_eq::<f32>(&want.into_data(), Tolerance::permissive());
    }
}

#[test]
fn a_pipelined_step_with_sgd_updates_as_a_full_batch_step() {
    let (flex, cpu) = devices();
    for schedule in SCHEDULES {
        assert_trains_as_one_device(
            &placement([(flex.clone(), 2), (cpu.clone(), 1), (flex.clone(), 1)]),
            schedule,
            || SgdConfig::new().init(),
        );
    }
}

#[test]
fn a_pipelined_step_with_adam_updates_as_a_full_batch_step() {
    let (flex, cpu) = devices();
    for schedule in SCHEDULES {
        assert_trains_as_one_device(
            &placement([(flex.clone(), 2), (cpu.clone(), 1), (flex.clone(), 1)]),
            schedule,
            || AdamConfig::new().init(),
        );
    }
}

#[test]
fn outer_layers_on_devices_of_their_own_train_as_parts_of_their_own() {
    let (flex, cpu) = devices();
    for schedule in SCHEDULES {
        assert_trains_as_one_device(
            &placement([(flex.clone(), 0), (cpu.clone(), 3), (flex.clone(), 0)]),
            schedule,
            || SgdConfig::new().init(),
        );
    }
}

#[test]
#[should_panic(expected = "the loss failed")]
fn a_panic_in_the_loss_stops_the_step_with_its_own_message() {
    let (flex, cpu) = devices();
    let placement = placement([(flex.clone(), 1), (cpu, 1)]);
    let mut training = TrainingPipeline::new(
        DistributedLayeredModel::new(Stack::new(&placement), &placement),
        PipelineSchedule::GPipe,
        SgdConfig::new().init(),
    );
    let input = Tensor::random([ROWS, WIDTH], Distribution::Default, &flex);

    training.step(
        input.chunk(MICROBATCHES, 0),
        vec![(); MICROBATCHES],
        |_, ()| -> Tensor<1> { panic!("the loss failed") },
        LEARNING_RATE,
    );
}

#[test]
fn evaluation_between_steps_runs_the_model_as_it_stands() {
    let (flex, cpu) = devices();
    let placement = placement([(flex.clone(), 2), (cpu, 2)]);
    let stack = Stack::new(&placement);
    let reference = stack.clone().fork(&flex);
    let training = TrainingPipeline::new(
        DistributedLayeredModel::new(stack, &placement),
        PipelineSchedule::OneForwardOneBackward,
        SgdConfig::new().init(),
    );
    let input = Tensor::random([ROWS, WIDTH], Distribution::Default, &flex);

    let outputs = training.forward(input.clone().chunk(MICROBATCHES, 0));

    Tensor::cat(
        outputs
            .into_iter()
            .map(|output| output.to_device(&flex.clone().inner()))
            .collect(),
        0,
    )
    .into_data()
    .assert_approx_eq::<f32>(
        &reference.plain_forward(input).into_data(),
        Tolerance::default(),
    );
}

#[derive(Module, Debug)]
struct Stack {
    input: Linear,
    hidden: Vec<Tanh>,
    output: Linear,
}

impl Stack {
    fn new(placement: &LayerPlacement) -> Self {
        Self {
            input: Linear::new(WIDTH, WIDTH, &placement.input),
            hidden: placement
                .hidden
                .iter()
                .map(|device| Tanh {
                    linear: Linear::new(WIDTH, WIDTH, device),
                })
                .collect(),
            output: Linear::new(WIDTH, 2, &placement.output),
        }
    }

    fn plain_forward(&self, input: Tensor<2>) -> Tensor<2> {
        let hidden = self
            .hidden
            .iter()
            .fold(self.input.forward(input), |x, layer| layer.forward(x));
        self.output.forward(hidden)
    }

    fn weights(&self) -> Vec<(String, Tensor<2>)> {
        core::iter::once(("input".to_string(), &self.input))
            .chain(
                self.hidden
                    .iter()
                    .enumerate()
                    .map(|(index, layer)| (format!("hidden {index}"), &layer.linear)),
            )
            .chain(core::iter::once(("output".to_string(), &self.output)))
            .map(|(name, linear)| (name, linear.weight.val()))
            .collect()
    }
}

impl LayerParallelism for Stack {
    type InputLayer = Linear;
    type HiddenLayer = Tanh;
    type OutputLayer = Linear;

    fn layer_input(&self) -> &Linear {
        &self.input
    }

    fn layers_hidden(&self) -> impl Iterator<Item = &Tanh> {
        self.hidden.iter()
    }

    fn layer_output(&self) -> &Linear {
        &self.output
    }

    fn into_layers(self) -> LayerSequence<Self> {
        LayerSequence {
            input: self.input,
            hidden: self.hidden,
            output: self.output,
        }
    }

    fn from_layers(layers: LayerSequence<Self>) -> Self {
        Self {
            input: layers.input,
            hidden: layers.hidden,
            output: layers.output,
        }
    }
}

#[derive(Module, Debug)]
struct Tanh {
    linear: Linear,
}

impl DistributedLayer for Tanh {
    type Input = Tensor<2>;
    type Output = Tensor<2>;

    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        self.linear.forward(input).tanh()
    }
}

#[derive(Module, Debug)]
struct Linear {
    weight: Param<Tensor<2>>,
}

impl Linear {
    fn new(inputs: usize, outputs: usize, device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::random(
                [outputs, inputs],
                Distribution::Default,
                device,
            )),
        }
    }
}

impl DistributedLayer for Linear {
    type Input = Tensor<2>;
    type Output = Tensor<2>;

    fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        input.matmul(self.weight.val().transpose())
    }
}
