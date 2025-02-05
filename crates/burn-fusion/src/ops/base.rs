use crate::{stream::Operation, FusionBackend};
use burn_tensor::repr::HandleContainer;
use std::marker::PhantomData;

#[derive(new)]
pub struct NoOp<B: FusionBackend> {
    _b: PhantomData<B>,
}

impl<B: FusionBackend> Operation<B::FusionRuntime> for NoOp<B> {
    fn execute(self: Box<Self>, _handles: &mut HandleContainer<B::Handle>) {}
}
