use burn_backend::{
    DType, ExecutionError, FloatDType, Shape, Slice, TensorData, TensorMetadata, TensorPrimitive,
    get_or_init_device_settings,
    ops::{FloatTensorOps, QTensorOps},
    quantization::{QuantPropagation, QuantScheme, QuantizationParametersPrimitive},
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};
use burn_ir::{
    DequantizeOpIr, FloatOperationIr, MatmulOpIr, OperationIr, OperationOutput,
    QuantizationParametersIr, QuantizeOpIr,
};

use crate::{BackendRouter, RouterChannel, RouterClient, RouterTensor};

// A quantized tensor routes as a float one: its IR carries the `QFloat` dtype the server reads.
impl<R: RouterChannel> QTensorOps<Self> for BackendRouter<R> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        Self::float_from_data(data, device)
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        let client = tensor.client.clone();
        let qparams = QuantizationParametersIr {
            scales: qparams.scales.into_ir(),
            global: qparams.global.map(RouterTensor::into_ir),
        };
        let desc = QuantizeOpIr::create(tensor.into_ir(), qparams, *scheme, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Float(
                desc.tensor.dtype,
                FloatOperationIr::Quantize(desc),
            ))
            .output()
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = dtype.into();
        let desc = DequantizeOpIr::create(tensor.into_ir(), dtype, || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                dtype,
                FloatOperationIr::Dequantize(desc),
            ))
            .output()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        Self::float_to_device(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        Self::float_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        tensor.into_data().await
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        Self::float_swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        Self::float_permute(tensor, axes)
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        Self::float_flip(tensor, axes)
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        Self::float_gather(dim, tensor, indices)
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        Self::float_select(tensor, dim, indices)
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        Self::float_slice(tensor, slices)
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        Self::float_expand(tensor, shape)
    }

    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        // rhs's scheme and lhs's float dtype win, as in the default `q_matmul` and fusion.
        let scheme = match (&lhs, &rhs) {
            (_, TensorPrimitive::QFloat(tensor)) | (TensorPrimitive::QFloat(tensor), _) => {
                Some(tensor.scheme())
            }
            _ => None,
        };
        let float_dtype = match (&lhs, &rhs) {
            (TensorPrimitive::Float(tensor), _) | (_, TensorPrimitive::Float(tensor)) => {
                Some(tensor.dtype)
            }
            _ => None,
        };
        let [lhs, rhs] = [lhs, rhs].map(|operand| match operand {
            TensorPrimitive::Float(tensor) | TensorPrimitive::QFloat(tensor) => tensor,
        });

        let client = lhs.client.clone();
        let settings = get_or_init_device_settings::<Self>(&client.device());
        let dtype = match (scheme, settings.quantization.propagation) {
            (Some(scheme), QuantPropagation::Propagate) => DType::QFloat(scheme),
            _ => float_dtype.unwrap_or_else(|| settings.float_dtype.into()),
        };
        let desc = MatmulOpIr::create_mixed(lhs.into_ir(), rhs.into_ir(), dtype, || {
            client.create_empty_handle()
        });

        let out = client
            .register(OperationIr::Float(dtype, FloatOperationIr::Matmul(desc)))
            .output();
        match dtype {
            DType::QFloat(_) => TensorPrimitive::QFloat(out),
            _ => TensorPrimitive::Float(out),
        }
    }
}
