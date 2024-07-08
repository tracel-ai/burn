use burn_tensor::{
    backend::Backend,
    ops::{QTensorOps, QuantizedTensor},
    Device, QuantizationStrategy, Shape,
};

use crate::{client::FusionClient, Fusion, FusionBackend};

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

    fn q_shape<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Shape<D> {
        tensor.shape()
    }

    fn q_device<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Device<Self> {
        tensor.client.device().clone()
    }
}
