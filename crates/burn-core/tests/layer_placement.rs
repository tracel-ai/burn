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
        DeviceShare, DistributedLayer, DistributedLayeredModel, LayerParallelism, LayerPlacement,
        LayerStack, LayerStage,
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
fn each_device_is_handed_the_layers_placed_on_it() {
    let (flex, cpu) = devices();
    let stages = [(flex.clone(), 2), (cpu.clone(), 1), (flex.clone(), 1)].map(
        |(device, num_hidden_layers)| LayerStage {
            device,
            num_hidden_layers,
        },
    );
    let placement = LayerPlacement::new(&stages);

    let shares =
        DistributedLayeredModel::new(Stack::new(&placement), &placement).into_device_shares();

    let [on_flex, on_cpu] = shares.as_slice() else {
        panic!("one share per device, got {}", shares.len());
    };
    assert_eq!((&on_flex.device, &on_cpu.device), (&flex, &cpu));
    assert!(on_flex.input.is_some() && on_flex.output.is_some());
    assert!(on_cpu.input.is_none() && on_cpu.output.is_none());
    let stages = |share: &DeviceShare<Stack>| {
        share
            .stages
            .iter()
            .map(|stage| {
                (
                    stage.stage,
                    stage.first_layer,
                    stage.layers.len(),
                    stage.next.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        stages(on_flex),
        [(0, 0, 2, cpu.clone()), (2, 3, 1, flex.clone())]
    );
    assert_eq!(stages(on_cpu), [(1, 2, 1, flex.clone())]);
    for share in &shares {
        for layer in share.stages.iter().flat_map(|stage| &stage.layers) {
            assert_eq!(layer.devices(), [share.device.clone()]);
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

    let first = stack.hidden[0].linear.weight.grad(&grads).unwrap();
    let last = stack.hidden[3].linear.weight.grad(&grads).unwrap();
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

    fn into_layers(self) -> LayerStack<Self> {
        LayerStack {
            input: self.input,
            hidden: self.hidden,
            output: self.output,
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
