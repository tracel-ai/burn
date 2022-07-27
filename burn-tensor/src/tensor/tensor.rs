use crate::tensor::{Data, Shape};

pub trait TensorBase<P, const D: usize> {
    fn shape(&self) -> &Shape<D>;
    fn into_data(self) -> Data<P, D>;
    fn to_data(&self) -> Data<P, D>;
}
