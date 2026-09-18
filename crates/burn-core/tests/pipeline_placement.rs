//! Where the parameters of a placed pipeline land, and what the split forward computes.
//!
//! Two fixed CPU backends stand in for two cards, so a split is observable without one.
//!
//! Run with `cargo test -p burn-core --features flex,ndarray --test pipeline_placement`.
#![cfg(all(feature = "flex", feature = "ndarray"))]
#![allow(deprecated)]

use burn_core as burn;
use burn_core::module::{
    Module, Param, ParamId,
    pipeline::{Pipeline, PipelineLayout, PipelinePlacement},
};
use burn_tensor::{Device, Distribution, Tensor, Tolerance};

const WIDTH: usize = 8;

fn devices() -> (Device, Device) {
    (Device::flex(), Device::ndarray())
}

#[test]
fn each_lazy_parameter_takes_the_device_of_its_segment() {
    let (flex, ndarray) = devices();
    let placement = PipelinePlacement {
        input: flex.clone(),
        blocks: vec![ndarray.clone(), flex.clone(), ndarray.clone()],
        output: ndarray.clone(),
    };

    let stack = Stack::lazy(3, &flex).place(&placement);

    let weights: Vec<_> = stack.weights().collect();
    assert!(weights.iter().all(|weight| !weight.is_initialized()));
    let placed: Vec<_> = weights.iter().map(|weight| weight.lazy_device()).collect();
    assert_eq!(placed, [ndarray.clone(), flex, ndarray.clone(), ndarray]);
}

#[test]
fn a_model_split_across_two_devices_computes_what_one_device_does() {
    let (flex, ndarray) = devices();
    let stack = Stack::new(4, &flex);
    let input = Tensor::random([4, WIDTH], Distribution::Default, &flex);
    let expected = stack.plain_forward(input.clone());

    let placement = PipelinePlacement::even(&[flex.clone(), ndarray.clone()], 4);
    let output = stack.place(&placement).forward(input);

    assert_eq!(output.device(), ndarray);
    output
        .to_device(&flex)
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
}

#[derive(Module, Debug)]
struct Stack {
    layers: Vec<Linear>,
    head: Linear,
}

impl Stack {
    fn new(blocks: usize, device: &Device) -> Self {
        Self {
            layers: (0..blocks)
                .map(|_| Linear::new(WIDTH, WIDTH, device))
                .collect(),
            head: Linear::new(WIDTH, 2, device),
        }
    }

    fn lazy(blocks: usize, device: &Device) -> Self {
        Self {
            layers: (0..blocks)
                .map(|_| Linear::lazy(WIDTH, WIDTH, device))
                .collect(),
            head: Linear::lazy(WIDTH, 2, device),
        }
    }

    fn weights(&self) -> impl Iterator<Item = &Param<Tensor<2>>> {
        self.layers
            .iter()
            .chain([&self.head])
            .map(|layer| &layer.weight)
    }

    fn plain_forward(&self, input: Tensor<2>) -> Tensor<2> {
        let hidden = self
            .layers
            .iter()
            .fold(input, |x, layer| layer.forward(x).tanh());
        self.head.forward(hidden)
    }
}

impl Pipeline for Stack {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type Carry = Tensor<2>;

    fn layout(&self) -> PipelineLayout {
        PipelineLayout::new()
            .blocks(&self.layers)
            .output(&self.head)
    }

    fn forward_input(&self, input: Tensor<2>) -> Tensor<2> {
        input
    }

    fn forward_block(&self, index: usize, carry: Tensor<2>) -> Tensor<2> {
        self.layers[index].forward(carry).tanh()
    }

    fn forward_output(&self, carry: Tensor<2>) -> Tensor<2> {
        self.head.forward(carry)
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

    fn lazy(inputs: usize, outputs: usize, device: &Device) -> Self {
        Self {
            weight: Param::uninitialized(
                ParamId::new(),
                move |device, require_grad| {
                    Tensor::random([outputs, inputs], Distribution::Default, device)
                        .set_require_grad(require_grad)
                },
                device.clone(),
                true,
                [outputs, inputs].into(),
            ),
        }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        x.matmul(self.weight.val().transpose())
    }
}
