use crate::CubeRuntime;
use crate::ops::numeric::empty_device_dtype;
use crate::tensor::CubeTensor;
use burn_tensor::DType;

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R>(tensor: CubeTensor<R>, dtype: DType) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let scheme = match tensor.dtype {
        DType::QFloat(scheme) => scheme,
        _ => return tensor,
    };

    let output = empty_device_dtype::<R>(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
        dtype,
    );
    let (values, params) = tensor.quantized_handles().unwrap();

    cubecl_quant::dequantize::launch_ref::<R>(
        &values.client,
        &values.as_handle_ref(),
        &output.as_handle_ref(),
        &params.as_handle_ref(),
        &scheme,
        dtype.into(),
    );

    output
}
