use std::ops::Range;

use burn_tensor::{
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{QuantizationParametersPrimitive, QuantizationScheme, QuantizationType},
    DType, Device, Shape, TensorData,
};

use crate::{
    element::BoolElement, kernel, tensor::CubeTensor, CubeBackend, CubeRuntime, FloatElement,
    IntElement,
};

/// Create a quantized tensor with packed values (u32).
fn new_qtensor<R: CubeRuntime, S: Into<Shape>>(
    data: &[u8],
    shape: S,
    scheme: QuantizationScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    let client = R::client(device);
    let buffer = client.create(data);

    CubeTensor::new_contiguous(
        client,
        device.clone(),
        shape.into(),
        buffer,
        DType::QFloat(scheme),
    )
}

impl<R, F, I, BT> QTensorOps<Self> for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        match data.dtype {
            DType::QFloat(scheme) => match scheme {
                QuantizationScheme::PerTensor(_mode, QuantizationType::QInt8) => {
                    // TensorData quantized representation is the same, with multiple quantized values
                    // packed into u32 and quantization parameters appended to the bytes
                    new_qtensor(data.as_bytes(), data.shape.clone(), scheme, device)
                }
                QuantizationScheme::PerBlock(_mode, _dtype, _block_layout) => todo!(),
            },
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        }
    }

    // TODO: quantize_dynamic (we can compute min-max on the fly and scale, especially when not per-tensor)

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantizationScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        kernel::quantization::quantize::<R, F, I>(tensor, scheme, qparams.scale, qparams.offset)
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        kernel::quantization::dequantize::<R, F>(tensor)
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        super::to_device(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        super::reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> TensorData {
        let tensor = kernel::into_contiguous(tensor);
        let bytes = tensor.client.read_one_async(tensor.handle.binding()).await;

        TensorData::from_bytes(bytes, tensor.shape, tensor.dtype)
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

    fn q_gather(
        _dim: usize,
        _tensor: QuantizedTensor<Self>,
        _indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_select(
        _tensor: QuantizedTensor<Self>,
        _dim: usize,
        _indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_slice(_tensor: QuantizedTensor<Self>, _ranges: &[Range<usize>]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_expand(_tensor: QuantizedTensor<Self>, _shape: Shape) -> QuantizedTensor<Self> {
        unimplemented!()
    }
}
