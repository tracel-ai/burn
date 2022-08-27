use crate::module::Module;
use crate::tensor::back::Backend;
use burn_tensor::Tensor;

pub trait Loss<B: Backend, T>: Module<Backend = B> {
    fn loss(&self, item: T) -> Tensor<B, 1>;
}

pub trait Learner<B: Backend, T, V, O, TO, VO> {
    fn train(&mut self, item: T, optim: &mut O) -> TO;
    fn valid(&self, item: V) -> VO;
}
