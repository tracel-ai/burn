use burn_tensor::{
    Shape, TensorData,
    backend::ExecutionError,
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{QuantScheme, QuantizationParametersPrimitive},
};

use crate::{LibTorch, LibTorchDevice, TchElement};

impl<E: TchElement> QTensorOps<Self> for LibTorch<E> {
    fn q_from_data(_data: TensorData, _device: &LibTorchDevice) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn quantize(
        _tensor: FloatTensor<Self>,
        _scheme: &QuantScheme,
        _qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn quantize_dynamic(
        _tensor: FloatTensor<Self>,
        _scheme: &QuantScheme,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn dequantize(_tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        unimplemented!()
    }

    fn q_device(_tensor: &QuantizedTensor<Self>) -> LibTorchDevice {
        unimplemented!()
    }

    fn q_to_device(
        _tensor: QuantizedTensor<Self>,
        _device: &burn_tensor::Device<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_reshape(_tensor: QuantizedTensor<Self>, _shape: Shape) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    async fn q_into_data(_tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        unimplemented!()
    }
    fn q_swap_dims(
        _tensor: QuantizedTensor<Self>,
        _dim1: usize,
        _dim2: usize,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_permute(_tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_flip(_tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_select(
        _tensor: QuantizedTensor<Self>,
        _dim: usize,
        _indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_slice(
        _tensor: QuantizedTensor<Self>,
        _slices: &[burn_tensor::Slice],
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_argmax(_tensor: QuantizedTensor<Self>, _dim: usize) -> IntTensor<Self> {
        unimplemented!()
    }

    fn q_argmin(_tensor: QuantizedTensor<Self>, _dim: usize) -> IntTensor<Self> {
        unimplemented!()
    }

    fn q_max_dim_with_indices(
        _tensor: QuantizedTensor<Self>,
        _dim: usize,
    ) -> (QuantizedTensor<Self>, IntTensor<Self>) {
        unimplemented!()
    }

    fn q_max_dim(_tensor: QuantizedTensor<Self>, _dim: usize) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_min_dim(_tensor: QuantizedTensor<Self>, _dim: usize) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_min_dim_with_indices(
        _tensor: QuantizedTensor<Self>,
        _dim: usize,
    ) -> (QuantizedTensor<Self>, IntTensor<Self>) {
        unimplemented!()
    }

    fn q_expand(_tensor: QuantizedTensor<Self>, _shape: Shape) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_sort(
        _tensor: QuantizedTensor<Self>,
        _dim: usize,
        _descending: bool,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_sort_with_indices(
        _tensor: QuantizedTensor<Self>,
        _dim: usize,
        _descending: bool,
    ) -> (QuantizedTensor<Self>, IntTensor<Self>) {
        unimplemented!()
    }

    fn q_argsort(
        _tensor: QuantizedTensor<Self>,
        _dim: usize,
        _descending: bool,
    ) -> IntTensor<Self> {
        unimplemented!()
    }
}
