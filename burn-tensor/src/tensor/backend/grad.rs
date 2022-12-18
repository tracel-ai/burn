use crate::backend::ADBackend;
use crate::Tensor;

pub trait Gradients<B: ADBackend>: Send + Sync + std::fmt::Debug {
    fn empty() -> Self;
    fn get<const D: usize>(&self, id: &str) -> Option<Tensor<B::InnerBackend, D>>;
    fn register<const D: usize>(&mut self, id: String, value: Tensor<B::InnerBackend, D>);
    fn remove<const D: usize>(&mut self, id: &str) -> Option<Tensor<B::InnerBackend, D>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
