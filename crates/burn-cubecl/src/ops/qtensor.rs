use std::ops::Range;

use burn_tensor::{
    DType, Device, Shape, TensorData, dequant_op_quant,
    ops::{FloatTensor, FloatTensorOps, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{
        BlockLayout, QTensorPrimitive, QuantizationMode, QuantizationParametersPrimitive,
        QuantizationScheme, QuantizationType,
    },
};

use crate::{
    CubeBackend, CubeRuntime, FloatElement, IntElement,
    element::BoolElement,
    kernel::{self, matmul::MatmulStrategy},
    tensor::CubeTensor,
};

use super::{permute, swap_dims};

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
                QuantizationScheme::PerTensor(_mode, QuantizationType::QInt8)
                | QuantizationScheme::PerBlock(
                    _mode,
                    QuantizationType::QInt8,
                    BlockLayout::Flat(..),
                ) => {
                    // TensorData quantized representation is the same, with multiple quantized values
                    // packed into u32 and quantization parameters appended to the bytes
                    new_qtensor(data.as_bytes(), data.shape.clone(), scheme, device)
                }
                QuantizationScheme::PerBlock(
                    _mode,
                    QuantizationType::QInt8,
                    BlockLayout::Grid(..),
                ) => panic!("Per-block quantization is not supported for grid layout"),
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

        // We use the same internal representation
        TensorData::from_bytes(bytes, tensor.shape, tensor.dtype)
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        permute(tensor, axes)
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

    fn q_matmul(lhs: QuantizedTensor<Self>, rhs: QuantizedTensor<Self>) -> QuantizedTensor<Self> {
        if both_matches_symmetric_qint8(lhs.scheme(), rhs.scheme()) {
            let out =
                kernel::matmul::q_matmul(lhs.clone(), rhs.clone(), None, MatmulStrategy::default());
            if let Ok(out) = out {
                // return <Self>::quantize_dynamic(out, lhs.scheme()); // Using lhs.scheme() is similar to the dequant_op_quant macro.
                return out;
            }
        }
        // If the above quantized matmul fail, we fallback to the dequantize-matmul-quantize pattern.
        dequant_op_quant!(
            ty Self,
            float_op Self::float_matmul,
            lhs,
            rhs
        )
    }
}

fn both_matches_symmetric_qint8(lhs: &QuantizationScheme, rhs: &QuantizationScheme) -> bool {
    [lhs, rhs].iter().all(|scheme| {
        matches!(
            scheme,
            QuantizationScheme::PerTensor(QuantizationMode::Symmetric, QuantizationType::QInt8),
        )
    })
}
