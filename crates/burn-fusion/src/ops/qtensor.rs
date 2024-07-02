use burn_tensor::{backend::Backend, ops::QTensorOps, QuantizationStrategy};

use crate::{Fusion, FusionBackend};

impl<B: FusionBackend> QTensorOps<Self> for Fusion<B> {
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
