use std::ops::Range;

use alloc::vec::Vec;
use burn_tensor::{
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{
        QTensorPrimitive, QuantizationParametersPrimitive, QuantizationScheme,
        QuantizationStrategy, QuantizationType,
    },
    DType, Device, ElementConversion, Shape, TensorData,
};

use crate::{
    kernel,
    tensor::{JitQuantizationParameters, JitTensor, QJitTensor},
    FloatElement, IntElement, JitBackend, JitRuntime,
};
use cubecl::CubeElement;

fn pack_i8s_to_u32s(data: &TensorData) -> Vec<u32> {
    // Shift and combine groups of four 8-bit values into a u32.
    // Same as doing this:
    //     let result = (a_u8 & 0xFF) << 24 | (b_u8 & 0xFF) << 16 | (c_u8 & 0xFF) << 8 | (d_u8 & 0xFF);
    data.as_bytes()
        .chunks(4)
        .map(|x| {
            x.iter().enumerate().fold(0u32, |acc, (i, x)| {
                acc | (*x as i8 as u32 & 0xFF) << ((3 - i) * 8)
            })
        })
        .collect()
}

/// Create a quantized tensor with packed values (u32).
fn packed_tensor<R: JitRuntime, S: Into<Shape>>(
    data: Vec<u32>,
    shape: S,
    device: &R::Device,
) -> JitTensor<R, u32> {
    let client = R::client(device);
    let buffer = client.create(u32::as_bytes(&data));

    JitTensor::new_contiguous(client, device.clone(), shape.into(), buffer)
}

impl<R, F, I> QTensorOps<Self> for JitBackend<R, F, I>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        match data.dtype {
            DType::QFloat(strategy) => match strategy {
                QuantizationStrategy::PerTensorAffineInt8(q) => {
                    // Convert quantized values to packed u32s
                    QJitTensor {
                        qtensor: packed_tensor(pack_i8s_to_u32s(&data), data.shape, device),
                        scheme: strategy.scheme(),
                        qparams: JitQuantizationParameters::new(
                            q.scale.elem(),
                            Some(q.offset.elem()),
                            device,
                        ),
                    }
                }
                QuantizationStrategy::PerTensorSymmetricInt8(q) => {
                    // Convert quantized values to packed u32s
                    QJitTensor {
                        qtensor: packed_tensor(pack_i8s_to_u32s(&data), data.shape, device),
                        scheme: strategy.scheme(),
                        qparams: JitQuantizationParameters::new(q.scale.elem(), None, device),
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
        kernel::quantization::quantize(tensor, scheme, qparams.into())
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        kernel::quantization::dequantize(tensor)
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
        let numel = tensor.qtensor.shape.num_elements();
        let qtensor = kernel::into_contiguous(tensor.qtensor);

        let bytes = qtensor.client.read_async(qtensor.handle.binding()).await;

        // Convert packed bytes to quantized dtype (TensorData can be used with other backends,
        // which don't have the prior knowledge of this packed representation)
        match &tensor.scheme {
            QuantizationScheme::PerTensorAffine(dtype)
            | QuantizationScheme::PerTensorSymmetric(dtype) => match dtype {
                QuantizationType::QInt8 => TensorData::quantized(
                    u32::from_bytes(&bytes)
                        .iter()
                        .enumerate()
                        .flat_map(|(i, packed)| {
                            // A single u32 could contain less than four 8-bit values...
                            let n = core::cmp::min(4, numel - i * 4);
                            // Extract each 8-bit segment from u32 and cast back to i8
                            // Same as doing this (when 4 values are fully packed):
                            //     let a = ((packed >> 24) & 0xFF) as i8;
                            //     let b = ((packed >> 16) & 0xFF) as i8;
                            //     let c = ((packed >> 8) & 0xFF) as i8;
                            //     let d = (packed & 0xFF) as i8;
                            (0..n).map(move |i| (packed >> ((3 - i) * 8) & 0xFF) as i8)
                        })
                        .collect(),
                    qtensor.shape,
                    strategy,
                ),
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
