use crate::module::Module;

/// One layer of a model split by [`LayerParallelism`](super::LayerParallelism). It runs on the
/// device its placement gives it, and what it returns moves to the next layer's device.
pub trait DistributedLayer: Module {
    /// What the layer takes, already on its device.
    type Input;
    /// What the layer returns, left on its device.
    type Output;

    /// Run the layer.
    fn forward(&self, input: Self::Input) -> Self::Output;
}
