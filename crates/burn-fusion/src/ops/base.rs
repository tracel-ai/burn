use crate::{FusionBackend, stream::Operation};
use burn_ir::HandleContainer;
use std::marker::PhantomData;

/// A no-operation placeholder for the fusion backend.
///
/// `NoOp` is an implementation of [`Operation`] that doesn't execute anything.
#[derive(new, Clone, Debug)]
pub struct NoOp<B: FusionBackend> {
    _b: PhantomData<B>,
}

impl<B: FusionBackend> Operation<B::FusionRuntime> for NoOp<B> {
    fn execute(&self, _handles: &mut HandleContainer<B::Handle>) {}
}
