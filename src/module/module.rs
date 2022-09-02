use super::State;
use crate::optim::Optimizer;
use crate::tensor::{back, Gradients};
pub use burn_derive::Module;

#[derive(Debug, new)]
pub struct LoadingError {
    message: String,
}

impl std::fmt::Display for LoadingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("Loading error: {}", self.message).as_str())
    }
}

impl std::error::Error for LoadingError {}

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
    fn load(
        &mut self,
        state: &State<<Self::Backend as back::Backend>::Elem>,
    ) -> Result<(), LoadingError>;
    fn state(&self) -> State<<Self::Backend as back::Backend>::Elem>;
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
