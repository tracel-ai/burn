use burn_tensor::{backend::Backend, ops::QTensorOps, QuantizationStrategy};

use crate::{FloatElement, IntElement, JitBackend, JitRuntime};

impl<R, F, I> QTensorOps<Self> for JitBackend<R, F, I>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
    fn quantize<const D: usize>(
        _tensor: <Self as Backend>::FloatTensorPrimitive<D>,
        _strategy: &QuantizationStrategy,
    ) -> <Self as Backend>::QuantizedTensorPrimitive<D> {
        unimplemented!()
    }

    fn dequantize<const D: usize>(
        _tensor: <Self as Backend>::QuantizedTensorPrimitive<D>,
        _strategy: &QuantizationStrategy,
    ) -> <Self as Backend>::FloatTensorPrimitive<D> {
        unimplemented!()
    }
}
