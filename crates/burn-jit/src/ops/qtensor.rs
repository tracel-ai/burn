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
    if data.bytes.len() % 4 != 0 {
        panic!("Number of elements in the input must be a factor of 4");
    }

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
fn packed_tensor<R: JitRuntime, S: Into<Shape<D>>, const D: usize>(
    data: Vec<u32>,
    shape: S,
    device: &R::Device,
) -> JitTensor<R, u32, D> {
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
    fn q_from_data<const D: usize>(
        data: TensorData,
        device: &Device<Self>,
    ) -> QuantizedTensor<Self, D> {
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

    fn quantize<const D: usize>(
        tensor: FloatTensor<Self, D>,
        scheme: &QuantizationScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self, D> {
        kernel::quantization::quantize(tensor, scheme, qparams.into())
    }

    fn dequantize<const D: usize>(tensor: QuantizedTensor<Self, D>) -> FloatTensor<Self, D> {
        kernel::quantization::dequantize(tensor)
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
            qparams: tensor.qparams,
        }
    }

    async fn q_into_data<const D: usize>(tensor: QuantizedTensor<Self, D>) -> TensorData {
        let strategy = tensor.strategy();
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
                        .flat_map(|packed| {
                            // Extract each 8-bit segment from u32 and cast back to i8
                            // Same as doing this:
                            //     let a = ((packed >> 24) & 0xFF) as i8;
                            //     let b = ((packed >> 16) & 0xFF) as i8;
                            //     let c = ((packed >> 8) & 0xFF) as i8;
                            //     let d = (packed & 0xFF) as i8;
                            (0..4).map(move |i| (packed >> ((3 - i) * 8) & 0xFF) as i8)
                        })
                        .collect(),
                    qtensor.shape,
                    strategy,
                ),
            },
        }
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
