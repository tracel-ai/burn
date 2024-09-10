use std::ops::Range;

use burn_tensor::{
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{QuantizationParametersPrimitive, QuantizationScheme},
    Device, Shape, TensorData,
};

use crate::{tensor::QJitTensor, FloatElement, IntElement, JitBackend, JitRuntime};

impl<R, F, I> QTensorOps<Self> for JitBackend<R, F, I>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
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
        unimplemented!()
    }

    fn dequantize<const D: usize>(_tensor: QuantizedTensor<Self, D>) -> FloatTensor<Self, D> {
        unimplemented!()
    }

    fn q_shape<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Shape<D> {
        tensor.qtensor.shape.clone()
    }

    fn q_device<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Device<Self> {
        tensor.qtensor.device.clone()
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
        QJitTensor {
            qtensor: super::reshape(tensor.qtensor, shape),
            scheme: tensor.scheme,
        }
    }

    async fn q_into_data<const D: usize>(_tensor: QuantizedTensor<Self, D>) -> TensorData {
        unimplemented!()
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

    fn q_expand<const D1: usize, const D2: usize>(
        _tensor: QuantizedTensor<Self, D1>,
        _shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        unimplemented!()
    }
}
