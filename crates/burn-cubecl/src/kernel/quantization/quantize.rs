use crate::{CubeRuntime, FloatElement, ops::empty_qtensor};
use crate::{ops::into_data_sync, tensor::CubeTensor};
use burn_tensor::quantization::QuantScheme;

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R, F>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
    F: FloatElement,
{
    let output = empty_qtensor(tensor.shape.clone(), *scheme, &tensor.device);
    let (out_values, out_params) = output.clone().quantized_handles().unwrap();

    println!(
        "quantize tensor: {}",
        into_data_sync::<R, f32>(tensor.clone())
    );
    println!("scheme: {:?}", scheme);
    println!("scale: {}", into_data_sync::<R, f32>(scale.clone()));

    cubecl_quant::quantize::launch_ref::<R, F>(
        &tensor.client,
        &tensor.as_handle_ref(),
        &out_values.as_handle_ref(),
        &scale.as_handle_ref(),
        &out_params.as_handle_ref(),
        scheme,
    );

    println!("output: {}", into_data_sync::<R, i8>(output.clone()));
    println!(
        "output scale: {}",
        into_data_sync::<R, f32>(output.scales().unwrap())
    );

    output
}
