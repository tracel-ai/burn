use super::FloatOps;
use burn_tensor::{backend::Backend, container::TensorContainer};

pub struct FusionServer<B: Backend, E> {
    candidates: Vec<Box<dyn FusedOps<B, E>>>,
    current_ops: Vec<FloatOps<B, E>>,
    tensors: TensorContainer<B>,
}

impl<B: Backend, E> FusionServer<B, E> {
    pub fn register(&mut self, ops: FloatOps<B, E>) {}
}

pub enum RegisterResult {
    Rejected(FusionProperties),
    Accepted(FusionProperties),
}

#[derive(Debug, Clone, Copy)]
pub struct FusionProperties {
    pub score: u64,
    pub ready: bool,
}

pub trait FusedOps<B: Backend, E> {
    fn score(&mut self, graph: Vec<FloatOps<B, E>>) -> FusionProperties;
    fn register(
        &mut self,
        ops: &FloatOps<B, E>,
        tensors: &mut TensorContainer<B>,
    ) -> RegisterResult;
    fn new_empty(&self) -> Box<dyn FusedOps<B, E>>;
    fn execute(&mut self, tensors: &mut TensorContainer<B>);
    fn len(&self) -> usize;
}

pub trait FusedBackend<B: Backend, E> {
    fn operations() -> Vec<Box<dyn FusedOps<B, E>>>;
    fn execute_ops(ops: FloatOps<B, E>, tensor: &mut TensorContainer<B>);
}
