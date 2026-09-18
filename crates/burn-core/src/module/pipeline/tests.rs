use alloc::vec::Vec;

use burn_tensor::{Device, Distribution, Tensor, Tolerance};

use super::*;
use crate as burn;
#[cfg(feature = "autodiff")]
use crate::module::{Param, ParamId};
use crate::{module::Module, test_device, test_utils::SimpleLinear};

#[test]
fn the_forward_on_stages_is_the_plain_forward() {
    let device = test_device();
    let stages = StageMap::single(&device, 3);
    let stack = Stack::new(3, &device).place(&stages);
    let input = Tensor::random([4, WIDTH], Distribution::Default, &device);
    let expected = stack.plain_forward(input.clone());

    stack
        .forward_on(&stages, input)
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
}

#[test]
#[should_panic(expected = "every block")]
fn a_stage_map_short_of_a_block_is_refused() {
    let device = test_device();
    Stack::new(3, &device).place(&StageMap::single(&device, 2));
}

#[test]
#[should_panic(expected = "belong to no segment")]
fn a_parameter_no_segment_owns_is_refused() {
    let device = test_device();
    let headless = Headless {
        stack: Stack::new(3, &device),
    };
    headless.place(&StageMap::single(&device, 3));
}

/// A parameter moved instead of forked gets no gradient, and the optimizer skips it.
#[cfg(feature = "autodiff")]
#[test]
fn every_parameter_receives_a_gradient() {
    let device = test_device().autodiff();
    let stages = StageMap::single(&device, 3);
    let stack = Stack::new(3, &device).place(&stages);
    let input = Tensor::random([4, WIDTH], Distribution::Default, &device);

    let grads = stack.forward_on(&stages, input).sum().backward();

    for (i, layer) in stack.layers.iter().enumerate() {
        assert!(
            layer.weight.grad(&grads).is_some(),
            "layer {i} has no gradient"
        );
    }
    assert!(
        stack.head.weight.grad(&grads).is_some(),
        "head has no gradient"
    );
}

#[cfg(feature = "autodiff")]
#[test]
fn a_lazy_parameter_initializes_on_its_segment_device() {
    let device = test_device();
    let segment_device = device.clone().autodiff();

    let stack = Stack::lazy(3, &device).place(&StageMap::single(&segment_device, 3));

    for weight in [&stack.layers[0].weight, &stack.head.weight] {
        assert!(!weight.is_initialized());
        assert_eq!(weight.lazy_device(), segment_device);
    }
}

/// `cargo test -p burn-core --features autodiff,cuda --lib splits_across_two_devices -- --ignored`,
/// or `vulkan` in place of `cuda`.
#[cfg(all(feature = "autodiff", any(feature = "cuda", feature = "vulkan")))]
#[test]
#[ignore = "needs two devices of one runtime"]
fn splits_across_two_devices() {
    let (first, second) = two_devices();
    let (first, second) = (first.autodiff(), second.autodiff());
    let host = test_device().autodiff();

    let stack = Stack::new(4, &host);
    let input = Tensor::random([4, WIDTH], Distribution::Default, &host);
    let expected = stack.plain_forward(input.clone());
    let expected_grads = expected.clone().sum().backward();
    let expected_first = stack.layers[0].weight.grad(&expected_grads).unwrap();
    let expected_last = stack.layers[3].weight.grad(&expected_grads).unwrap();

    let stages = StageMap::new(&[
        Stage {
            device: first.clone(),
            blocks: 2,
        },
        Stage {
            device: second.clone(),
            blocks: 2,
        },
    ]);
    let stack = stack.place(&stages);
    let output = stack.forward_on(&stages, input);
    assert_eq!(output.device(), second);
    output
        .clone()
        .to_device(&host)
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());

    let grads = output.sum().backward();
    let first_grad = stack.layers[0].weight.grad(&grads).unwrap();
    let last_grad = stack.layers[3].weight.grad(&grads).unwrap();
    assert_eq!(first_grad.device(), first.clone().inner());
    assert_eq!(last_grad.device(), second.clone().inner());
    first_grad
        .to_device(&host.clone().inner())
        .into_data()
        .assert_approx_eq::<f32>(&expected_first.into_data(), Tolerance::default());
    last_grad
        .to_device(&host.clone().inner())
        .into_data()
        .assert_approx_eq::<f32>(&expected_last.into_data(), Tolerance::default());
}

#[cfg(all(feature = "autodiff", any(feature = "cuda", feature = "vulkan")))]
fn two_devices() -> (Device, Device) {
    #[cfg(feature = "cuda")]
    let devices = (Device::cuda(0), Device::cuda(1));
    #[cfg(all(feature = "vulkan", not(feature = "cuda")))]
    let devices = {
        use burn_tensor::DeviceKind;
        (
            Device::vulkan(DeviceKind::DiscreteGpu(0)),
            Device::vulkan(DeviceKind::DiscreteGpu(1)),
        )
    };
    devices
}

const WIDTH: usize = 8;

#[derive(Module, Debug)]
struct Stack {
    layers: Vec<SimpleLinear>,
    head: SimpleLinear,
}

impl Stack {
    fn new(blocks: usize, device: &Device) -> Self {
        Self {
            layers: (0..blocks)
                .map(|_| SimpleLinear::new(WIDTH, WIDTH, device))
                .collect(),
            head: SimpleLinear::new(WIDTH, 2, device),
        }
    }

    #[cfg(feature = "autodiff")]
    fn lazy(blocks: usize, device: &Device) -> Self {
        Self {
            layers: (0..blocks)
                .map(|_| lazy_linear(WIDTH, WIDTH, device))
                .collect(),
            head: lazy_linear(WIDTH, 2, device),
        }
    }

    fn plain_forward(&self, input: Tensor<2>) -> Tensor<2> {
        let hidden = self
            .layers
            .iter()
            .fold(input, |x, layer| linear(layer, x).tanh());
        linear(&self.head, hidden)
    }
}

impl Pipeline for Stack {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type Activations = Tensor<2>;

    fn layout(&self) -> PipelineLayout {
        PipelineLayout::new()
            .blocks(&self.layers)
            .output(&self.head)
    }

    fn forward_input(&self, input: Tensor<2>) -> Tensor<2> {
        input
    }

    fn forward_block(&self, index: usize, activations: Tensor<2>) -> Tensor<2> {
        linear(&self.layers[index], activations).tanh()
    }

    fn forward_output(&self, activations: Tensor<2>) -> Tensor<2> {
        linear(&self.head, activations)
    }
}

#[derive(Module, Debug)]
struct Headless {
    stack: Stack,
}

impl Pipeline for Headless {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type Activations = Tensor<2>;

    fn layout(&self) -> PipelineLayout {
        PipelineLayout::new().blocks(&self.stack.layers)
    }

    fn forward_input(&self, input: Tensor<2>) -> Tensor<2> {
        self.stack.forward_input(input)
    }

    fn forward_block(&self, index: usize, activations: Tensor<2>) -> Tensor<2> {
        self.stack.forward_block(index, activations)
    }

    fn forward_output(&self, activations: Tensor<2>) -> Tensor<2> {
        self.stack.forward_output(activations)
    }
}

#[cfg(feature = "autodiff")]
fn lazy_linear(inputs: usize, outputs: usize, device: &Device) -> SimpleLinear {
    SimpleLinear {
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
        bias: None,
    }
}

fn linear(layer: &SimpleLinear, x: Tensor<2>) -> Tensor<2> {
    let out = x.matmul(layer.weight.val().transpose());
    match &layer.bias {
        Some(bias) => out + bias.val().unsqueeze(),
        None => out,
    }
}
