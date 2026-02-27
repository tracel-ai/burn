use crate::tensor::CubeTensor;
use crate::{CubeRuntime, ops::numeric::empty_device_dtype};
use burn_backend::{DType, TensorMetadata};

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R>(tensor: CubeTensor<R>, dtype: DType) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let scheme = match tensor.dtype {
        DType::QFloat(scheme) => scheme,
        _ => return tensor,
    };

    let output = empty_device_dtype(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape(),
        dtype,
    );
    let (values, params) = tensor.quantized_handles().unwrap();

    cubek::quantization::dequantize::launch_ref(
        &output.client,
        values.binding(),
        output.clone().binding(),
        params.binding(),
        &scheme,
        dtype.into(),
    )
    .expect("Kernel to never fail");

    output
}
