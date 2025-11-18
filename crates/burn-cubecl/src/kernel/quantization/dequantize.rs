use crate::CubeRuntime;
use crate::ops::numeric::empty_device_dtype;
use crate::tensor::CubeTensor;
use burn_tensor::DType;

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R>(tensor: CubeTensor<R>) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let (shape, dtype) = (tensor.shape.clone(), tensor.dtype);
    let output = empty_device_dtype::<R>(
        tensor.client.clone(),
        tensor.device.clone(),
        shape,
        tensor.dtype,
    );
    let (values, params) = tensor.quantized_handles().unwrap();

    match dtype {
        DType::QFloat(scheme) => {
            cubecl_quant::dequantize::launch_ref::<R>(
                &values.client,
                &values.as_handle_ref(),
                &output.as_handle_ref(),
                &params.as_handle_ref(),
                &scheme,
                tensor.dtype.into(),
            );
        }
        _ => panic!("Expected QFloat dtype"),
    };

    output
}
