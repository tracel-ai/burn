use alloc::vec::Vec;

use burn_tensor::Device;

use super::{
    AutoregressiveLayer, DistributedLayer, HiddenLayerSignal, LayerParallelism, LayerPlacement,
    LayerStage,
};
use crate::module::{Devices, Module, ModuleMapper, ModuleVisitor};

/// What a [`LayerParallelism`] model takes: its input layer's input.
pub type LayerParallelismInput<M> =
    <<M as LayerParallelism>::InputLayer as DistributedLayer>::Input;

/// What a [`LayerParallelism`] model returns: its output layer's output.
pub type LayerParallelismOutput<M> =
    <<M as LayerParallelism>::OutputLayer as DistributedLayer>::Output;

/// A [`LayerParallelism`] model split across devices, holding where each of its layers runs so
/// the forward reads it rather than working it out again on every call. Dereferences to the model,
/// and is itself a [`Module`], so records, devices and training flags work as they do on the model
/// alone.
#[derive(Clone, Debug)]
pub struct DistributedLayeredModel<M: LayerParallelism> {
    model: M,
    placement: LayerPlacement,
}

impl<M: LayerParallelism> DistributedLayeredModel<M> {
    /// Split `model` across the devices `placement` gives its layers. Nothing moves: each layer is
    /// built where it runs, and this checks it is.
    ///
    /// # Panics
    ///
    /// Panics when `placement` does not give a device to every hidden layer, or when a layer holds
    /// a tensor off the device `placement` gives it.
    pub fn new(model: M, placement: &LayerPlacement) -> Self {
        Self::assert_placed(&model, placement);

        Self {
            model,
            placement: placement.clone(),
        }
    }

    /// Where each layer runs, which is where its own parameters are.
    pub fn placement(&self) -> &LayerPlacement {
        &self.placement
    }

    /// Run each layer on its device, moving what passes between them.
    pub fn forward(&self, input: LayerParallelismInput<M>) -> LayerParallelismOutput<M> {
        self.forward_with(input, |layer, signal| layer.forward(signal))
    }

    /// The model on its own, no longer carrying where its layers run.
    pub fn into_inner(self) -> M {
        self.model
    }

    /// The forward, with `forward_hidden` running each hidden layer, in order, on the signal
    /// already moved to its device.
    fn forward_with(
        &self,
        input: LayerParallelismInput<M>,
        mut forward_hidden: impl FnMut(&M::HiddenLayer, HiddenLayerSignal<M>) -> HiddenLayerSignal<M>,
    ) -> LayerParallelismOutput<M> {
        let placement = &self.placement;
        let mut signal = self
            .model
            .layer_input()
            .forward(input.to_device(&placement.input));
        for (layer, device) in self.model.layers_hidden().zip(&placement.hidden) {
            signal = forward_hidden(layer, signal.to_device(device));
        }
        self.model
            .layer_output()
            .forward(signal.to_device(&placement.output))
    }

    fn assert_placed(model: &M, placement: &LayerPlacement) {
        let count = model.layers_hidden().count();
        assert!(
            count == placement.hidden.len(),
            "the placement must give a device to every hidden layer: the model has {count}, the \
             placement gives {}",
            placement.hidden.len()
        );

        assert!(
            Self::layer_is_on(model.layer_input(), &placement.input),
            "the input layer must be built on {:?}",
            placement.input
        );
        for (index, (layer, device)) in model.layers_hidden().zip(&placement.hidden).enumerate() {
            assert!(
                Self::layer_is_on(layer, device),
                "hidden layer {index} must be built on {device:?}"
            );
        }
        assert!(
            Self::layer_is_on(model.layer_output(), &placement.output),
            "the output layer must be built on {:?}",
            placement.output
        );
    }

    fn layer_is_on(layer: &impl Module, device: &Device) -> bool {
        layer.devices().iter().all(|found| found == device)
    }

    /// A move of the whole model puts every layer on one device, the hidden layer count unchanged.
    fn on_one_device(&self, device: &Device) -> LayerPlacement {
        LayerPlacement::new(&[LayerStage {
            device: device.clone(),
            num_hidden_layers: self.placement.hidden.len(),
        }])
    }
}

impl<M: LayerParallelism> DistributedLayeredModel<M>
where
    M::HiddenLayer: AutoregressiveLayer,
{
    /// A cache for every hidden layer, each made by its layer.
    pub fn new_autoregressive_cache(&self) -> DistributedAutoregressiveCache<M> {
        DistributedAutoregressiveCache {
            hidden: self
                .model
                .layers_hidden()
                .map(AutoregressiveLayer::new_autoregressive_cache)
                .collect(),
        }
    }

    /// Run the positions that follow the ones `cache` has seen, each hidden layer reading and
    /// extending its own cache on its own device. Only the signal moves between devices.
    ///
    /// # Panics
    ///
    /// Panics when `cache` was made for a model with another number of hidden layers.
    pub fn forward_autoregressive_inference(
        &self,
        input: LayerParallelismInput<M>,
        cache: &mut DistributedAutoregressiveCache<M>,
    ) -> LayerParallelismOutput<M> {
        assert_eq!(
            cache.hidden.len(),
            self.placement.hidden.len(),
            "the cache must hold one entry per hidden layer of this model"
        );
        let mut caches = cache.hidden.iter_mut();
        self.forward_with(input, |layer, signal| {
            let cache = caches
                .next()
                .expect("one cache per hidden layer, checked above");
            layer.forward_autoregressive_inference(signal, cache)
        })
    }
}

/// The caches of a [`DistributedLayeredModel`]'s hidden layers, made by
/// [`new_autoregressive_cache`](DistributedLayeredModel::new_autoregressive_cache).
pub struct DistributedAutoregressiveCache<M: LayerParallelism>
where
    M::HiddenLayer: AutoregressiveLayer,
{
    hidden: Vec<<M::HiddenLayer as AutoregressiveLayer>::Cache>,
}

impl<M: LayerParallelism> core::ops::Deref for DistributedLayeredModel<M> {
    type Target = M;

    fn deref(&self) -> &M {
        &self.model
    }
}

/// A mapper is trusted to leave each parameter where it found it, so only a move of the whole
/// model rewrites the placement.
impl<M: LayerParallelism> Module for DistributedLayeredModel<M> {
    fn collect_devices(&self, devices: Devices) -> Devices {
        self.model.collect_devices(devices)
    }

    fn fork(self, device: &Device) -> Self {
        let placement = self.on_one_device(device);
        Self {
            model: self.model.fork(device),
            placement,
        }
    }

    fn to_device(self, device: &Device) -> Self {
        let placement = self.on_one_device(device);
        Self {
            model: self.model.to_device(device),
            placement,
        }
    }

    fn train(self) -> Self {
        Self {
            model: self.model.train(),
            placement: self.placement,
        }
    }

    fn valid(&self) -> Self {
        Self {
            model: self.model.valid(),
            placement: self.placement.clone(),
        }
    }

    fn visit<Visitor: ModuleVisitor>(&self, visitor: &mut Visitor) {
        self.model.visit(visitor);
    }

    fn map<Mapper: ModuleMapper>(self, mapper: &mut Mapper) -> Self {
        Self {
            model: self.model.map(mapper),
            placement: self.placement,
        }
    }

    fn materialize(self) -> Self {
        Self {
            model: self.model.materialize(),
            placement: self.placement,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use burn_tensor::{Distribution, Tensor, Tolerance};

    use super::*;
    use crate as burn;
    use crate::{test_device, test_utils::SimpleLinear};

    const WIDTH: usize = 8;

    #[test]
    fn the_forward_on_one_device_is_the_plain_forward() {
        let device = test_device();
        let stack = Stack::new(3, &device);
        let input = Tensor::random([4, WIDTH], Distribution::Default, &device);
        let expected = stack.plain_forward(input.clone());

        let placement = LayerPlacement::even(core::slice::from_ref(&device), 3);
        DistributedLayeredModel::new(stack, &placement)
            .forward(input)
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    #[should_panic(expected = "every hidden layer")]
    fn a_placement_short_of_a_hidden_layer_is_refused() {
        let device = test_device();
        let placement = LayerPlacement::even(core::slice::from_ref(&device), 2);
        DistributedLayeredModel::new(Stack::new(3, &device), &placement);
    }

    #[test]
    #[should_panic(expected = "every hidden layer")]
    fn a_placement_past_the_last_hidden_layer_is_refused() {
        let device = test_device();
        let placement = LayerPlacement::even(core::slice::from_ref(&device), 4);
        DistributedLayeredModel::new(Stack::new(3, &device), &placement);
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn every_parameter_receives_a_gradient() {
        let device = test_device().autodiff();
        let placement = LayerPlacement::even(core::slice::from_ref(&device), 3);
        let stack = DistributedLayeredModel::new(Stack::new(3, &device), &placement);
        let input = Tensor::random([4, WIDTH], Distribution::Default, &device);

        let grads = stack.forward(input).sum().backward();

        let hidden = stack.hidden.iter().map(|layer| &layer.linear);
        for (index, linear) in [&stack.input.linear, &stack.output.linear]
            .into_iter()
            .chain(hidden)
            .enumerate()
        {
            assert!(
                linear.weight.grad(&grads).is_some(),
                "layer {index} has no gradient"
            );
        }
    }

    #[derive(Module, Debug)]
    struct Stack {
        input: Tanh,
        hidden: Vec<Tanh>,
        output: Head,
    }

    impl Stack {
        fn new(hidden: usize, device: &Device) -> Self {
            Self {
                input: Tanh::new(device),
                hidden: (0..hidden).map(|_| Tanh::new(device)).collect(),
                output: Head {
                    linear: SimpleLinear::new(WIDTH, 2, device),
                },
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
        type InputLayer = Tanh;
        type HiddenLayer = Tanh;
        type OutputLayer = Head;

        fn layer_input(&self) -> &Tanh {
            &self.input
        }

        fn layers_hidden(&self) -> impl Iterator<Item = &Tanh> {
            self.hidden.iter()
        }

        fn layer_output(&self) -> &Head {
            &self.output
        }
    }

    #[derive(Module, Debug)]
    struct Tanh {
        linear: SimpleLinear,
    }

    impl Tanh {
        fn new(device: &Device) -> Self {
            Self {
                linear: SimpleLinear::new(WIDTH, WIDTH, device),
            }
        }
    }

    impl DistributedLayer for Tanh {
        type Input = Tensor<2>;
        type Output = Tensor<2>;

        fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            linear(&self.linear, input).tanh()
        }
    }

    #[derive(Module, Debug)]
    struct Head {
        linear: SimpleLinear,
    }

    impl DistributedLayer for Head {
        type Input = Tensor<2>;
        type Output = Tensor<2>;

        fn forward(&self, input: Tensor<2>) -> Tensor<2> {
            linear(&self.linear, input)
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
