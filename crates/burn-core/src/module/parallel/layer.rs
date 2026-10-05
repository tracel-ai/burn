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

/// A [`DistributedLayer`] that keeps what it computed for the earlier positions of a sequence, so
/// the sequence can run a few positions at a time: a decoder block keeping the keys and values it
/// has attended to.
pub trait AutoregressiveLayer: DistributedLayer {
    /// What the layer keeps between forwards. It stays on the layer's device, so a split model's
    /// caches never cross between devices.
    type Cache;

    /// A cache that has seen no position yet.
    fn new_autoregressive_cache(&self) -> Self::Cache;

    /// Run the layer on the positions that follow the ones `cache` has seen, and keep them in it.
    fn forward_autoregressive_inference(
        &self,
        input: <Self as DistributedLayer>::Input,
        cache: &mut Self::Cache,
    ) -> <Self as DistributedLayer>::Output;
}
