use std::ops::Range;

use burn_tensor::{
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{
        QTensorPrimitive, QuantizationParametersPrimitive, QuantizationScheme, QuantizationType,
    },
    DType, Device, Shape, TensorData,
};

use crate::{
    element::BoolElement,
    kernel,
    tensor::{JitQuantizationParameters, JitTensor, QJitTensor},
    FloatElement, IntElement, JitBackend, JitRuntime,
};
use cubecl::CubeElement;

/// Create a quantized tensor with packed values (u32).
fn packed_tensor<R: JitRuntime, S: Into<Shape>>(
    data: &[u8],
    shape: S,
    device: &R::Device,
) -> JitTensor<R> {
    let client = R::client(device);
    let buffer = client.create(data);

    JitTensor::new_contiguous(client, device.clone(), shape.into(), buffer, DType::U32)
}

impl<R, F, I, BT> QTensorOps<Self> for JitBackend<R, F, I, BT>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        match data.dtype {
            DType::QFloat(scheme) => match scheme {
                QuantizationScheme::PerTensorAffine(QuantizationType::QInt8)
                | QuantizationScheme::PerTensorSymmetric(QuantizationType::QInt8) => {
                    // Convert quantized values to packed u32s
                    let qparams = data.get_q_params::<F, i8>().unwrap();
                    QJitTensor {
                        qtensor: packed_tensor(data.values_as_bytes(), data.shape.clone(), device),
                        scheme,
                        qparams: JitQuantizationParameters::new(
                            qparams.scale,
                            qparams.offset,
                            device,
                        ),
                    }
                }
            },
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        }
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantizationScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        kernel::quantization::quantize::<R, F, I>(tensor, scheme, qparams.into())
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        kernel::quantization::dequantize::<R, F, I>(tensor)
    }

    fn q_shape(tensor: &QuantizedTensor<Self>) -> Shape {
        tensor.qtensor.shape.clone()
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        tensor.qtensor.device.clone()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        let mut tensor = tensor;
        tensor.qtensor = super::to_device(tensor.qtensor, device);
        tensor.qparams.scale = super::to_device(tensor.qparams.scale, device);
        tensor.qparams.offset = tensor.qparams.offset.map(|x| super::to_device(x, device));

        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        QJitTensor {
            qtensor: super::reshape(tensor.qtensor, shape),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> TensorData {
        let strategy = tensor.strategy();
        let qtensor = kernel::into_contiguous(tensor.qtensor);

        let bytes = qtensor
            .client
            .read_one_async(qtensor.handle.binding())
            .await;

        // TensorData keeps quantized values packed into 32-bit unsigned integers so we can
        // keep the current representation, just cast the bytes as u32.
        match &tensor.scheme {
            QuantizationScheme::PerTensorAffine(dtype)
            | QuantizationScheme::PerTensorSymmetric(dtype) => match dtype {
                QuantizationType::QInt8 => {
                    TensorData::quantized(u32::from_bytes(&bytes).to_vec(), qtensor.shape, strategy)
                }
            },
        }
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
