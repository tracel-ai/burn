use std::ops::Range;

use burn_tensor::{
    backend::Backend,
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{QuantizationParametersPrimitive, QuantizationScheme},
    Device, Shape, TensorData,
};

use crate::{checkpoint::strategy::CheckpointStrategy, Autodiff};

impl<B: Backend, C: CheckpointStrategy> QTensorOps<Self> for Autodiff<B, C> {
    fn q_from_data<const D: usize>(
        _data: TensorData,
        _device: &Device<Self>,
    ) -> QuantizedTensor<Self, D> {
        todo!()
    }

    fn quantize<const D: usize>(
        _tensor: FloatTensor<Self, D>,
        _scheme: &QuantizationScheme,
        _qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self, D> {
        todo!() // required for QAT
    }

    fn quantize_dynamic<const D: usize>(
        _tensor: FloatTensor<Self, D>,
        _scheme: &QuantizationScheme,
    ) -> QuantizedTensor<Self, D> {
        todo!()
    }

    fn dequantize<const D: usize>(_tensor: QuantizedTensor<Self, D>) -> FloatTensor<Self, D> {
        todo!()
    }

    fn q_shape<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Shape<D> {
        B::q_shape(tensor)
    }

    fn q_device<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Device<Self> {
        B::q_device(tensor)
    }

    fn q_to_device<const D: usize>(
        _tensor: QuantizedTensor<Self, D>,
        _device: &Device<Self>,
    ) -> QuantizedTensor<Self, D> {
        unimplemented!()
    }

    fn q_reshape<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        B::q_reshape(tensor, shape)
    }

    async fn q_into_data<const D: usize>(tensor: QuantizedTensor<Self, D>) -> TensorData {
        B::q_into_data(tensor).await
    }

    fn q_swap_dims<const D: usize>(
        _tensor: QuantizedTensor<Self, D>,
        _dim1: usize,
        _dim2: usize,
    ) -> QuantizedTensor<Self, D> {
        unimplemented!()
    }

    fn q_permute<const D: usize>(
        _tensor: QuantizedTensor<Self, D>,
        _axes: [usize; D],
    ) -> QuantizedTensor<Self, D> {
        unimplemented!()
    }

    fn q_flip<const D: usize>(
        _tensor: QuantizedTensor<Self, D>,
        _axes: &[usize],
    ) -> QuantizedTensor<Self, D> {
        unimplemented!()
    }

    fn q_gather<const D: usize>(
        _dim: usize,
        _tensor: QuantizedTensor<Self, D>,
        _indices: IntTensor<Self, D>,
    ) -> QuantizedTensor<Self, D> {
        unimplemented!()
    }

    fn q_select<const D: usize>(
        _tensor: QuantizedTensor<Self, D>,
        _dim: usize,
        _indices: IntTensor<Self, 1>,
    ) -> QuantizedTensor<Self, D> {
        unimplemented!()
    }

    fn q_slice<const D1: usize, const D2: usize>(
        _tensor: QuantizedTensor<Self, D1>,
        _ranges: [Range<usize>; D2],
    ) -> QuantizedTensor<Self, D1> {
        unimplemented!()
    }

    fn q_argmax<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        B::q_argmax(tensor, dim)
    }

    fn q_argmin<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        B::q_argmin(tensor, dim)
    }

    fn q_expand<const D1: usize, const D2: usize>(
        _tensor: QuantizedTensor<Self, D1>,
        _shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        unimplemented!()
    }
}
