use crate::decorator::FullPrecisionBridge;
use crate::decorator::SparseRepresentation;
use burn_tensor::backend::Backend;
use core::marker::PhantomData;
use derive_new::new;

/// Tensor backend that extends existing backends with sparse tensor support.
/// This backend abstracts over all backends, and so lacks the performance of a direct implementation.
/// Backends implementing SparseDecorator should be used directly where possible.
#[derive(new, Clone, Copy, Default, Debug)]
pub struct SparseDecorator<B: Backend, R: SparseRepresentation> {
    _p: PhantomData<B>,
    _r: PhantomData<R>,
}

impl<B: Backend, R: SparseRepresentation> Backend for SparseDecorator<B, R> {
    type Device = B::Device;

    type FullPrecisionBridge = FullPrecisionBridge<B::FullPrecisionBridge>;

    type FloatTensorPrimitive<const D: usize> = B::FloatTensorPrimitive<D>;

    type FloatElem = B::FloatElem;

    type IntTensorPrimitive<const D: usize> = B::IntTensorPrimitive<D>;

    type IntElem = B::IntElem;

    type BoolTensorPrimitive<const D: usize> = B::BoolTensorPrimitive<D>;

    type QuantizedTensorPrimitive<const D: usize> = B::QuantizedTensorPrimitive<D>;

    fn name() -> String {
        format!("SparseDecorator<{}>", B::name())
    }

    fn seed(seed: u64) {
        B::seed(seed)
    }
}

impl<B: Backend, R: SparseRepresentation> SparseDecorator<B, R> {}
