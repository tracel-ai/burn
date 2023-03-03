use crate::{backend::Backend, Float, TensorKind};

#[derive(new)]
pub struct TensorNew<B, const D: usize, K = Float>
where
    B: Backend,
    K: TensorKind<B>,
{
    pub(crate) primitive: K::Primitive<D>,
}
