//! Where the layers of a split model must be, and what the split forward computes.
//!
//! Two fixed CPU backends stand in for two cards, so a split is observable without one.
//!
//! Run with `cargo test -p burn-core --features flex,cpu,autodiff --test layer_placement`.
#![cfg(all(feature = "flex", feature = "cpu"))]

use burn_core as burn;
use burn_core::module::{
    Module, Param,
    parallel::{
        DistributedLayer, DistributedLayeredModel, InferencePipeline, LayerParallelism,
        LayerPlacement, LayerSequence, LayerStage, PipelineParts,
    },
};
use burn_tensor::{Device, Distribution, Tensor, Tolerance};

const WIDTH: usize = 8;

fn devices() -> (Device, Device) {
    (Device::flex(), Device::cpu())
}

#[test]
#[should_panic(expected = "hidden layer 2 must be built on")]
fn a_layer_built_off_its_placed_device_is_refused() {
    let (flex, cpu) = devices();
    let stack = Stack::new(&LayerPlacement::even(core::slice::from_ref(&flex), 4));

    DistributedLayeredModel::new(stack, &LayerPlacement::even(&[flex, cpu], 4));
}

#[test]
fn a_device_returned_to_later_starts_a_stage_of_its_own() {
    let (flex, cpu) = devices();
    let stages =
        [(flex.clone(), 2), (cpu, 1), (flex, 1)].map(|(device, num_hidden_layers)| LayerStage {
            device,
            num_hidden_layers,
        });

    assert_eq!(LayerPlacement::new(&stages).stages(), stages);
}

#[test]
fn each_stage_holds_the_layers_placed_on_its_device() {
    let (flex, cpu) = devices();
    let stages = [(flex.clone(), 2), (cpu.clone(), 1), (flex.clone(), 1)].map(
        |(device, num_hidden_layers)| LayerStage {
            device,
            num_hidden_layers,
        },
    );
    let placement = LayerPlacement::new(&stages);

    let staged = DistributedLayeredModel::new(Stack::new(&placement), &placement).into_stages();

    let held: Vec<(Device, usize)> = staged
        .stages
        .iter()
        .map(|stage| (stage.device.clone(), stage.layers.len()))
        .collect();
    assert_eq!(held, [(flex.clone(), 2), (cpu, 1), (flex, 1)]);
    for stage in &staged.stages {
        for layer in &stage.layers {
            assert_eq!(layer.devices(), core::slice::from_ref(&stage.device));
        }
    }
}

#[test]
fn a_model_split_across_two_devices_computes_what_one_device_does() {
    let (flex, cpu) = devices();
    let placement = LayerPlacement::even(&[flex.clone(), cpu.clone()], 4);
    let stack = Stack::new(&placement);
    let input = Tensor::random([4, WIDTH], Distribution::Default, &flex);
    let expected = stack.clone().fork(&flex).plain_forward(input.clone());

    let output = DistributedLayeredModel::new(stack, &placement).forward(input);

    assert_eq!(output.device(), cpu);
    output
        .to_device(&flex)
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
}

#[test]
fn a_pipeline_returns_each_batch_in_order_as_the_plain_forward_computes_it() {
    let (flex, cpu) = devices();
    let placement = placement([(flex.clone(), 2), (cpu, 1), (flex, 1)]);

    assert_pipeline_matches_forward(&placement);
}

#[test]
fn outer_layers_on_devices_of_their_own_run_as_parts_of_their_own() {
    let (flex, cpu) = devices();
    let placement = placement([(flex.clone(), 0), (cpu, 1), (flex, 0)]);
    let model = DistributedLayeredModel::new(Stack::new(&placement), &placement);

    let parts = PipelineParts::new(model.into_stages(), &placement);

    assert!(parts.input.stage.is_none() && parts.output.stage.is_none());
    assert_eq!(parts.middle.len(), 1);
    assert_pipeline_matches_forward(&placement);
}

fn placement<const N: usize>(stages: [(Device, usize); N]) -> LayerPlacement {
    LayerPlacement::new(&stages.map(|(device, num_hidden_layers)| LayerStage {
        device,
        num_hidden_layers,
    }))
}

fn assert_pipeline_matches_forward(placement: &LayerPlacement) {
    let model = DistributedLayeredModel::new(Stack::new(placement), placement);
    let inputs: Vec<Tensor<2>> = (0..5)
        .map(|_| Tensor::random([3, WIDTH], Distribution::Default, &placement.input))
        .collect();
    let expected: Vec<Tensor<2>> = inputs
        .iter()
        .map(|input| model.forward(input.clone()))
        .collect();

    let outputs = InferencePipeline::new(model).forward(inputs);

    assert_eq!(outputs.len(), expected.len());
    for (output, expected) in outputs.into_iter().zip(expected) {
        output
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }
}

#[cfg(feature = "autodiff")]
#[test]
fn gradients_of_a_split_model_match_the_gradients_on_one_device() {
    let (flex, cpu) = devices();
    let (flex, cpu) = (flex.autodiff(), cpu.autodiff());
    let placement = LayerPlacement::even(&[flex.clone(), cpu.clone()], 4);
    let stack = Stack::new(&placement);
    let input = Tensor::random([4, WIDTH], Distribution::Default, &flex);

    let reference = stack.clone().fork(&flex);
    let expected = reference.plain_forward(input.clone()).sum().backward();
    let expected_first = reference.hidden[0].linear.weight.grad(&expected).unwrap();
    let expected_last = reference.hidden[3].linear.weight.grad(&expected).unwrap();

    let stack = DistributedLayeredModel::new(stack, &placement);
    let grads = stack.forward(input).sum().backward();

    let first = stack.model().hidden[0].linear.weight.grad(&grads).unwrap();
    let last = stack.model().hidden[3].linear.weight.grad(&grads).unwrap();
    assert_eq!(first.device(), flex.clone().inner());
    assert_eq!(last.device(), cpu.inner());
    first
        .to_device(&flex.clone().inner())
        .into_data()
        .assert_approx_eq::<f32>(&expected_first.into_data(), Tolerance::default());
    last.to_device(&flex.inner())
        .into_data()
        .assert_approx_eq::<f32>(&expected_last.into_data(), Tolerance::default());
}

#[test]
#[should_panic(expected = "`into_layers` must give the hidden layers in the order")]
fn hidden_layers_handed_over_out_of_order_are_refused() {
    let (flex, cpu) = devices();
    let placement = LayerPlacement::even(&[flex, cpu], 4);
    let model = Reversed {
        stack: Stack::new(&placement),
    };

    DistributedLayeredModel::new(model, &placement).into_stages();
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

    /// The reference the split forward has to reproduce.
    fn plain_forward(&self, input: Tensor<2>) -> Tensor<2> {
        let hidden = self
            .hidden
            .iter()
            .fold(self.input.forward(input), |x, layer| layer.forward(x));
        self.output.forward(hidden)
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

/// A model whose `into_layers` disagrees with `layers_hidden` on the order of the hidden layers.
#[derive(Module, Debug)]
struct Reversed {
    stack: Stack,
}

impl LayerParallelism for Reversed {
    type InputLayer = Linear;
    type HiddenLayer = Tanh;
    type OutputLayer = Linear;

    fn layer_input(&self) -> &Linear {
        self.stack.layer_input()
    }

    fn layers_hidden(&self) -> impl Iterator<Item = &Tanh> {
        self.stack.layers_hidden()
    }

    fn layer_output(&self) -> &Linear {
        self.stack.layer_output()
    }

    fn into_layers(self) -> LayerSequence<Self> {
        let LayerSequence {
            input,
            mut hidden,
            output,
        } = self.stack.into_layers();
        hidden.reverse();
        LayerSequence {
            input,
            hidden,
            output,
        }
    }

    fn from_layers(layers: LayerSequence<Self>) -> Self {
        let LayerSequence {
            input,
            mut hidden,
            output,
        } = layers;
        hidden.reverse();
        Self {
            stack: Stack {
                input,
                hidden,
                output,
            },
        }
    }
}
