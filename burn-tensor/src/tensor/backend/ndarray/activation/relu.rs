use crate::{activation::ReLU, backend::ndarray::NdArrayTensor, Element};
use rand::distributions::Standard;

impl<E: Element, const D: usize> ReLU<E, D> for NdArrayTensor<E, D>
where
    E: Element,
    Standard: rand::distributions::Distribution<E>,
{
    fn relu(&self) -> Self {
        todo!()
    }
}
