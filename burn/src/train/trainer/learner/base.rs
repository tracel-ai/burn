use crate::module::Module;
use crate::tensor::backend::Backend;
use burn_tensor::Tensor;

pub trait Loss: Module {
    type Item;

    fn loss(&self, item: Self::Item) -> Tensor<Self::Backend, 1>;
}

pub trait Learner<TI, VI, TO, VO> {
    type Backend: Backend;

    fn train(&mut self, item: TI) -> TO;
    fn valid(&self, item: VI) -> VO;
}

pub trait LearnerCheckpoint {
    fn checkpoint(&self, epoch: usize);
    fn load_checkpoint(&mut self, epoch: usize);
}
