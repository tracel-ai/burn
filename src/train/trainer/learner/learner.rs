use crate::module::Module;
use crate::tensor::back::Backend;
use burn_tensor::Tensor;

pub trait Loss: Module {
    type Item;

    fn loss(&self, item: Self::Item) -> Tensor<Self::Backend, 1>;
}

pub trait Learner<T, V, TO, VO> {
    type Backend: Backend;

    fn train(&mut self, item: T) -> TO;
    fn valid(&self, item: V) -> VO;
}
