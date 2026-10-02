//! Where the layers of a split model must be, and what the split forward computes.
//!
//! Two fixed CPU backends stand in for two cards, so a split is observable without one.
//!
//! Run with `cargo test -p burn-core --features flex,ndarray,autodiff --test layer_placement`.
#![cfg(all(feature = "flex", feature = "ndarray"))]
#![allow(deprecated)]

use burn_core as burn;
use burn_core::module::{
    Module, Param,
    parallel::{DistributedLayer, DistributedLayeredModel, LayerParallelism, LayerPlacement},
};
use burn_tensor::{Device, Distribution, Tensor, Tolerance};

const WIDTH: usize = 8;

fn devices() -> (Device, Device) {
    (Device::flex(), Device::ndarray())
}

#[test]
#[should_panic(expected = "hidden layer 2 must be built on")]
fn a_layer_built_off_its_placed_device_is_refused() {
    let (flex, ndarray) = devices();
    let stack = Stack::new(&LayerPlacement::even(core::slice::from_ref(&flex), 4));

    DistributedLayeredModel::new(stack, &LayerPlacement::even(&[flex, ndarray], 4));
}

#[test]
fn a_model_split_across_two_devices_computes_what_one_device_does() {
    let (flex, ndarray) = devices();
    let placement = LayerPlacement::even(&[flex.clone(), ndarray.clone()], 4);
    let stack = Stack::new(&placement);
    let input = Tensor::random([4, WIDTH], Distribution::Default, &flex);
    let expected = stack.clone().fork(&flex).plain_forward(input.clone());

    let output = DistributedLayeredModel::new(stack, &placement).forward(input);

    assert_eq!(output.device(), ndarray);
    output
        .to_device(&flex)
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
}

#[cfg(feature = "autodiff")]
#[test]
fn gradients_of_a_split_model_match_the_gradients_on_one_device() {
    let (flex, ndarray) = devices();
    let (flex, ndarray) = (flex.autodiff(), ndarray.autodiff());
    let placement = LayerPlacement::even(&[flex.clone(), ndarray.clone()], 4);
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
    assert_eq!(last.device(), ndarray.inner());
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

    fn layer_hidden(&self, index: usize) -> Option<&Tanh> {
        self.hidden.get(index)
    }

    fn layer_output(&self) -> &Linear {
        &self.output
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
