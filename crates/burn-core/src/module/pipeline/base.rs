use alloc::{string::String, vec::Vec};

use burn_tensor::{Bool, Int, Tensor};

use super::{PipelineLayout, PipelinePlacement, PlacedPipeline};
use crate::module::{Module, ModuleMapper, Param, ParameterValue};

/// A model whose forward pass runs in segments, so its layers can live on different devices: the
/// input segment `forward_input`, one segment per block `forward_block`, in order, then the output
/// segment `forward_output`.
///
/// [`PipelineLayout`] says which submodules each segment runs: a module tree does not say in what
/// order forward runs.
///
/// ```rust,ignore
/// impl Pipeline for Model {
///     type Input = Tensor<2>;
///     type Output = Tensor<2>;
///     type Carry = Tensor<2>;
///
///     fn layout(&self) -> PipelineLayout {
///         PipelineLayout::new()
///             .input(&self.input)
///             .blocks(&self.blocks)
///             .output(&self.output)
///     }
///
///     fn forward_input(&self, features: Tensor<2>) -> Tensor<2> {
///         self.input.forward(features)
///     }
///
///     fn forward_block(&self, index: usize, hidden: Tensor<2>) -> Tensor<2> {
///         self.blocks[index].forward(hidden)
///     }
///
///     fn forward_output(&self, hidden: Tensor<2>) -> Tensor<2> {
///         self.output.forward(hidden)
///     }
/// }
///
/// let placement = PipelinePlacement::even(&devices, model.layout().num_blocks());
/// let model = model.place(&placement);
/// let predictions = model.forward(features);
/// ```
pub trait Pipeline: Module {
    /// What `forward_input` consumes.
    type Input: Module;
    /// What `forward_output` produces, left on the output segment's device.
    type Output;
    /// Everything a segment passes to the next: the hidden state, and whatever rides along with
    /// it, such as masks.
    type Carry: Module;

    /// Which submodules each segment runs.
    fn layout(&self) -> PipelineLayout;

    /// Everything before the first block: embeddings, input projection.
    fn forward_input(&self, input: Self::Input) -> Self::Carry;

    /// One block. `index` is the block's position in the layout.
    fn forward_block(&self, index: usize, carry: Self::Carry) -> Self::Carry;

    /// Everything after the last block: final norm, output head.
    fn forward_output(&self, carry: Self::Carry) -> Self::Output;

    /// Fork every parameter onto the device `placement` gives the segment that owns it, giving a
    /// [`PlacedPipeline`] that runs its forward pass there.
    ///
    /// Forking rather than moving keeps each parameter a leaf, so the model still trains. A
    /// parameter not initialized yet only takes the device, so it initializes there and a record
    /// loaded afterwards loads there. One that a live clone of the model shares initializes where
    /// it is and is copied instead, so drop other clones first. Only parameters move: a tensor a
    /// module holds directly stays where the model was built.
    ///
    /// # Panics
    ///
    /// Panics when `placement` does not give a device to every block of the layout, or when a
    /// parameter belongs to no segment, since nothing would say which device it goes to.
    fn place(self, placement: &PipelinePlacement) -> PlacedPipeline<Self> {
        let layout = self.layout();
        placement.assert_covers(&layout);

        let model = Module::map(
            self,
            &mut PlaceOnOwningSegment {
                placement,
                layout: &layout,
                path: Vec::new(),
            },
        );

        PlacedPipeline::new(model, placement.clone())
    }
}

/// `layout` says which segment owns a parameter, `placement` says where that segment runs.
struct PlaceOnOwningSegment<'a> {
    placement: &'a PipelinePlacement,
    layout: &'a PipelineLayout,
    /// Follows the walk, so a parameter no segment owns can be named in the panic.
    path: Vec<String>,
}

impl PlaceOnOwningSegment<'_> {
    /// A parameter no segment owns is a hole in `layout`, never a device to guess at.
    fn fork<P: ParameterValue>(&self, param: Param<P>) -> Param<P>
    where
        Param<P>: Module,
    {
        match self.layout.segment(param.id) {
            Some(segment) => param.fork(self.placement.device(segment)),
            None => panic!(
                "parameter `{}` belongs to no segment of the layout; claim its module in `layout`",
                self.path.join(".")
            ),
        }
    }
}

impl ModuleMapper for PlaceOnOwningSegment<'_> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.into());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        self.fork(param)
    }

    fn map_int<const D: usize>(&mut self, param: Param<Tensor<D, Int>>) -> Param<Tensor<D, Int>> {
        self.fork(param)
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<D, Bool>>,
    ) -> Param<Tensor<D, Bool>> {
        self.fork(param)
    }
}

#[cfg(test)]
mod tests {
    use burn_tensor::{Device, Distribution, Tolerance};

    use super::*;
    use crate as burn;
    use crate::{test_device, test_utils::SimpleLinear};

    #[test]
    fn the_forward_on_one_device_is_the_plain_forward() {
        let device = test_device();
        let placement = PipelinePlacement::even(core::slice::from_ref(&device), 3);
        let stack = Stack::new(3, &device).place(&placement);
        let input = Tensor::random([4, WIDTH], Distribution::Default, &device);
        let expected = stack.plain_forward(input.clone());

        stack
            .forward(input)
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    #[should_panic(expected = "every block")]
    fn a_placement_short_of_a_block_is_refused() {
        let device = test_device();
        Stack::new(3, &device).place(&PipelinePlacement::even(&[device], 2));
    }

    #[test]
    #[should_panic(expected = "`stack.head.weight` belongs to no segment")]
    fn a_parameter_no_segment_owns_is_refused_by_its_path() {
        let device = test_device();
        let model = StackWithUnclaimedHead {
            stack: Stack::new(3, &device),
        };
        model.place(&PipelinePlacement::even(&[device], 3));
    }

    /// A parameter moved instead of forked gets no gradient, and the optimizer skips it.
    #[cfg(feature = "autodiff")]
    #[test]
    fn every_parameter_receives_a_gradient() {
        let device = test_device().autodiff();
        let placement = PipelinePlacement::even(core::slice::from_ref(&device), 3);
        let stack = Stack::new(3, &device).place(&placement);
        let input = Tensor::random([4, WIDTH], Distribution::Default, &device);

        let grads = stack.forward(input).sum().backward();

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

        /// The reference the split forward has to reproduce.
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
            linear(&self.layers[index], carry).tanh()
        }

        fn forward_output(&self, carry: Tensor<2>) -> Tensor<2> {
            linear(&self.head, carry)
        }
    }

    /// `layout` never claims the head that `forward_output` runs.
    #[derive(Module, Debug)]
    struct StackWithUnclaimedHead {
        stack: Stack,
    }

    impl Pipeline for StackWithUnclaimedHead {
        type Input = Tensor<2>;
        type Output = Tensor<2>;
        type Carry = Tensor<2>;

        fn layout(&self) -> PipelineLayout {
            PipelineLayout::new().blocks(&self.stack.layers)
        }

        fn forward_input(&self, input: Tensor<2>) -> Tensor<2> {
            self.stack.forward_input(input)
        }

        fn forward_block(&self, index: usize, carry: Tensor<2>) -> Tensor<2> {
            self.stack.forward_block(index, carry)
        }

        fn forward_output(&self, carry: Tensor<2>) -> Tensor<2> {
            self.stack.forward_output(carry)
        }
    }

    /// `SimpleLinear` is a bare pair of parameters, with no forward of its own.
    fn linear(layer: &SimpleLinear, x: Tensor<2>) -> Tensor<2> {
        let out = x.matmul(layer.weight.val().transpose());
        match &layer.bias {
            Some(bias) => out + bias.val().unsqueeze(),
            None => out,
        }
    }
}
