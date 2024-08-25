use core::marker::PhantomData;

use burn_tensor::{
    backend::{Backend, BackendBridge},
    ops::FloatTensor,
};

use crate::decorator::SparseDecorator;
use crate::decorator::SparseRepresentation;

#[derive(Debug)]
pub struct FullPrecisionBridge<Bridge> {
    _p: PhantomData<Bridge>,
}

impl<B, R, Bridge> BackendBridge<SparseDecorator<B, R>> for FullPrecisionBridge<Bridge>
where
    B: Backend,
    R: SparseRepresentation,
    Bridge: BackendBridge<B> + 'static,
{
    type Target = SparseDecorator<Bridge::Target, R>;

    fn into_target<const D: usize>(
        tensor: FloatTensor<SparseDecorator<B, R>, D>,
        device: Option<burn_tensor::Device<Self::Target>>,
    ) -> burn_tensor::ops::FloatTensor<Self::Target, D> {
        Bridge::into_target(tensor, device)
    }

    fn from_target<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self::Target, D>,
        device: Option<burn_tensor::Device<SparseDecorator<B, R>>>,
    ) -> burn_tensor::ops::FloatTensor<SparseDecorator<B, R>, D> {
        Bridge::from_target(tensor, device)
    }
}
