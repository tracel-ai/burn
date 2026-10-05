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
        AutoregressiveLayer, DistributedLayer, DistributedLayeredModel, LayerParallelism,
        LayerPlacement,
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

/// Generation runs a prompt, then one position at a time, and must produce what a forward over
/// the whole sequence would. Each hidden layer's cache is reused on its own device, so a cache
/// handed to the wrong layer or left on another device fails here.
#[test]
fn a_sequence_run_a_few_positions_at_a_time_matches_one_forward_over_it() {
    let (flex, cpu) = devices();
    let placement = LayerPlacement::even(&[flex.clone(), cpu], 4);
    let model = DistributedLayeredModel::new(CausalStack::new(&placement), &placement);
    let input = Tensor::random([2, 5, WIDTH], Distribution::Default, &flex);
    let expected = model.forward(input.clone());

    let mut cache = model.new_autoregressive_cache();
    let steps = [(0, 3), (3, 1), (4, 1)].map(|(start, length)| {
        model.forward_autoregressive_inference(input.clone().narrow(1, start, length), &mut cache)
    });

    Tensor::cat(steps.to_vec(), 1)
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

#[derive(Module, Debug)]
struct CausalStack {
    input: Scale,
    hidden: Vec<PrefixSum>,
    output: Scale,
}

impl CausalStack {
    fn new(placement: &LayerPlacement) -> Self {
        Self {
            input: Scale::new(&placement.input),
            hidden: placement
                .hidden
                .iter()
                .map(|device| PrefixSum {
                    scale: Scale::new(device),
                })
                .collect(),
            output: Scale::new(&placement.output),
        }
    }
}

impl LayerParallelism for CausalStack {
    type InputLayer = Scale;
    type HiddenLayer = PrefixSum;
    type OutputLayer = Scale;

    fn layer_input(&self) -> &Scale {
        &self.input
    }

    fn layers_hidden(&self) -> impl Iterator<Item = &PrefixSum> {
        self.hidden.iter()
    }

    fn layer_output(&self) -> &Scale {
        &self.output
    }
}

#[derive(Module, Debug)]
struct Scale {
    weight: Param<Tensor<1>>,
}

impl Scale {
    fn new(device: &Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::random([WIDTH], Distribution::Default, device)),
        }
    }
}

impl DistributedLayer for Scale {
    type Input = Tensor<3>;
    type Output = Tensor<3>;

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        input * self.weight.val().unsqueeze()
    }
}

/// Each position's output depends on every position up to it, as in a causal decoder block.
#[derive(Module, Debug)]
struct PrefixSum {
    scale: Scale,
}

impl DistributedLayer for PrefixSum {
    type Input = Tensor<3>;
    type Output = Tensor<3>;

    fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        self.scale.forward(input.cumsum(1)).tanh()
    }
}

impl AutoregressiveLayer for PrefixSum {
    /// The sum of every position seen so far, `[batch, 1, width]`.
    type Cache = Option<Tensor<3>>;

    fn new_autoregressive_cache(&self) -> Self::Cache {
        None
    }

    fn forward_autoregressive_inference(
        &self,
        input: Tensor<3>,
        cache: &mut Self::Cache,
    ) -> Tensor<3> {
        let [_, positions, _] = input.dims();
        let sums = match cache.take() {
            Some(seen) => input.cumsum(1) + seen,
            None => input.cumsum(1),
        };
        *cache = Some(sums.clone().narrow(1, positions - 1, 1));
        self.scale.forward(sums).tanh()
    }
}
