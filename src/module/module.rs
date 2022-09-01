use super::State;
use crate::optim::Optimizer;
use crate::tensor::{back, Gradients};
pub use burn_derive::Module;

pub trait Module: Send + Sync + std::fmt::Debug + std::fmt::Display {
    type Backend: back::Backend;

    fn update_params<O: Optimizer<Backend = Self::Backend>>(
        &mut self,
        grads: &Gradients,
        optim: &mut O,
    ) where
        Self::Backend: back::ad::Backend;
    fn devices(&self) -> Vec<<Self::Backend as back::Backend>::Device>;
    fn to_device(&mut self, device: <Self::Backend as back::Backend>::Device);
    fn name(&self) -> &str;
    fn load(&mut self, state: &State<Self::Backend>);
    fn state(&self) -> State<Self::Backend>;
    fn num_params(&self) -> usize;
}

pub trait ADModule: Module + Send + Sync + std::fmt::Debug + std::fmt::Display {
    type ADBackend: back::ad::Backend;
    type InnerModule: Module<Backend = <Self::ADBackend as back::ad::Backend>::InnerBackend>;

    fn inner(&self) -> Self::InnerModule;
}

pub trait Forward<In, Out> {
    fn forward(&self, input: In) -> Out;
}
