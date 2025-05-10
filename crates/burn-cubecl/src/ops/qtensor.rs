use std::ops::Range;

use burn_tensor::{
    DType, Device, Shape, TensorData, TensorPrimitive,
    ops::{FloatTensor, FloatTensorOps, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{
        QTensorPrimitive, QuantInputType, QuantLevel, QuantMode, QuantPropagation, QuantScheme,
        QuantizationParametersPrimitive,
    },
};
use cubecl::{
    Feature, Runtime,
    client::ComputeClient,
    ir::{Elem, IntKind},
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
    scheme: QuantScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    let client = R::client(device);
    let shape: Shape = shape.into();
    let (data, shapes, elem_sizes) = match scheme {
        // Just to ensure we get and error if more modes are added and unhandled
        QuantScheme {
            level: QuantLevel::Tensor,
            mode: QuantMode::Symmetric,
            q_type: QuantInputType::QInt8,
            ..
        } => {
            let data = vec![&data[..shape.num_elements()], &data[shape.num_elements()..]];
            let shapes = vec![shape.dims.as_slice(), &[1]];
            let elem_sizes = vec![size_of::<i8>(), size_of::<f32>()];
            (data, shapes, elem_sizes)
        }
    };

    let (handle, strides) = client.create_tensors(data, shapes, elem_sizes).remove(0);

    CubeTensor::new(
        client,
        handle,
        shape,
        device.clone(),
        strides,
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
                QuantScheme {
                    level: QuantLevel::Tensor,
                    mode: QuantMode::Symmetric,
                    q_type: QuantInputType::QInt8,
                    ..
                } => {
                    // TensorData quantized representation is the same, with multiple quantized values
                    // packed into u32 and quantization parameters appended to the bytes
                    new_qtensor(data.as_bytes(), data.shape.clone(), scheme, device)
                }
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
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        kernel::quantization::quantize::<R, F>(tensor, scheme, qparams.scale)
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
        let client = tensor.client.clone();
        let handle = tensor.handle.clone();
        let data_size = handle.size();
        let dtype = tensor.dtype;

        let tensor = kernel::into_contiguous(tensor);
        let mut bytes = tensor
            .client
            .read_async(vec![handle.clone().binding()])
            .await
            .remove(0);
        bytes.truncate(tensor.shape.num_elements() * tensor.dtype.size());
        if let DType::QFloat(params) = dtype {
            // There's no scale type rn so assume f32. At some point we should add a type to support
            // things like e8m0 scales.
            let (scale_len, scale_ty) = match params.level {
                QuantLevel::Tensor => (1, DType::F32),
            };
            let mut handle = handle.offset_start(data_size);
            handle.offset_end = None;
            let scale_bytes = scale_len * scale_ty.size();
            let data = client.read_one(handle.binding());
            bytes.extend(data[..scale_bytes].iter().copied())
        }

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

    fn q_matmul(lhs: QuantizedTensor<Self>, rhs: QuantizedTensor<Self>) -> TensorPrimitive<Self> {
        if features_enabled::<R>(&lhs.client)
            && both_matches_symmetric_qint8(lhs.scheme(), rhs.scheme())
        {
            let out =
                kernel::matmul::q_matmul(lhs.clone(), rhs.clone(), None, MatmulStrategy::default());
            if let Ok(out) = out {
                return match lhs.scheme().propagation {
                    QuantPropagation::Propagate => {
                        TensorPrimitive::QFloat(Self::quantize_dynamic(out, lhs.scheme()))
                    }
                    QuantPropagation::Inhibit => TensorPrimitive::Float(out),
                };
            }
        }

        // If the above quantized matmul fail, we fallback to the dequantize-then-matmul pattern.
        let scheme = *lhs.scheme();
        let t1_f = <Self>::dequantize(lhs);
        let t2_f = <Self>::dequantize(rhs);
        let out = Self::float_matmul(t1_f, t2_f);

        match scheme.propagation {
            QuantPropagation::Propagate => {
                TensorPrimitive::QFloat(Self::quantize_dynamic(out, &scheme))
            }
            QuantPropagation::Inhibit => TensorPrimitive::Float(out),
        }
    }
}

fn both_matches_symmetric_qint8(lhs: &QuantScheme, rhs: &QuantScheme) -> bool {
    [lhs, rhs].iter().all(|scheme| {
        matches!(
            scheme,
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                q_type: QuantInputType::QInt8,
                ..
            }
        )
    })
}

fn features_enabled<R: Runtime>(client: &ComputeClient<R::Server, R::Channel>) -> bool {
    client
        .properties()
        .feature_enabled(Feature::Type(Elem::Int(IntKind::I8)))
        && client
            .properties()
            .feature_enabled(Feature::DynamicLineSize)
}
