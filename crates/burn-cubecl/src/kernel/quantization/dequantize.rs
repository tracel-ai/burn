use crate::ops::numeric::empty_device;
use crate::tensor::CubeTensor;
use crate::{CubeRuntime, FloatElement};
use burn_tensor::DType;

/// Convert the tensor back to a higher precision data type.
pub fn dequantize<R, F>(tensor: CubeTensor<R>) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
{
    let (shape, dtype) = (tensor.shape.clone(), tensor.dtype);
    let output = empty_device::<R, F>(tensor.client.clone(), tensor.device.clone(), shape);
    let (values, params) = tensor.quantized_handles().unwrap();

    match dtype {
        DType::QFloat(scheme) => {
            // TODO: Only works with contiguous tensors.
            cubecl_quant::dequantize::launch_ref::<R, F>(
                &values.client,
                &values.as_handle_ref(),
                &output.as_handle_ref(),
                &params.as_handle_ref(),
                &scheme,
            );
        }
        _ => panic!("Expected QFloat dtype"),
    };

    output
}
