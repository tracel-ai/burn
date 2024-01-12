use super::OptimizationStore;
use crate::{FusionBackend, Optimization};

impl<O> OptimizationStore<O> {
    #[allow(dead_code)]
    /// TODO: save the cache state.
    pub(crate) fn save<B: FusionBackend>(&self, _device: &B::Device)
    where
        O: Optimization<B>,
    {
        todo!("Save the state");
    }

    #[allow(dead_code)]
    #[allow(unreachable_code)]
    #[allow(unused_variables)]
    #[allow(clippy::diverging_sub_expression)]
    /// TODO: load the cache state.
    pub(crate) fn load<B: FusionBackend>(_device: &B::Device) -> Self
    where
        O: Optimization<B>,
    {
        todo!("Load the state");
    }
}
