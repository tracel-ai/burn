//! The burn side of the depthwise convolution routine.
//!
//! `launch_convolution_forward` rejects `groups != 1` outright, so every depthwise layer in a
//! model autotunes against `conv_direct` alone — the accelerated candidates all decline before
//! they are timed. This wrapper reaches the one routine that does accept it.

use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::ops::{ConvOptions, conv::calculate_conv_output_sizes};
use cubek::convolution::{
    ConvolutionArgs, DepthwiseTiling, components::ConvSetupError, launch_depthwise_tiled,
};

/// Perform a depthwise 2D convolution: one filter per channel, `groups == channels`, under the
/// stated [`DepthwiseTiling`].
///
/// The tiling is a parameter rather than the routine's own default because it is what the tuner
/// has to choose: the same routine runs 8% faster over an encoder's depthwise layers when the
/// tile is picked per shape than under any single one of them, and only timing says which.
///
/// A bias is not folded in here. The convolutions this targets carry none — every grouped shape
/// in the model has `has_bias: false` — and adding one would be a second pass over the output
/// that the caller can already express.
pub fn conv_depthwise<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
    tiling: DepthwiseTiling,
) -> Result<CubeTensor<R>, ConvSetupError> {
    if N != 2 {
        return Err(ConvSetupError::Unknown);
    }
    if bias.is_some() {
        return Err(ConvSetupError::Unknown);
    }

    let out_dtype = input.dtype;
    let rank = input.meta.shape().num_dims();
    let batch_size = input.meta.shape()[0];
    let dim_c = rank - 1;
    let shape = &input.meta.shape()[1..dim_c];

    let out_channels = weight.meta.shape()[0];
    let weight_shape = &weight.meta.shape()[1..dim_c];

    let mut out_shape = calculate_conv_output_sizes(
        weight_shape,
        &options.stride,
        &options.padding,
        &options.dilation,
        shape,
    );
    out_shape.insert(0, batch_size);
    out_shape.push(out_channels);

    let out = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        out_shape.clone().into(),
        out_dtype,
    );

    let args = ConvolutionArgs::<2> {
        stride: [options.stride[0], options.stride[1]],
        padding: [options.padding[0], options.padding[1]],
        dilation: [options.dilation[0], options.dilation[1]],
    };

    let client = input.client.clone();
    let in_shape: Vec<usize> = input.meta.shape().iter().copied().collect();
    let w_shape: Vec<usize> = weight.meta.shape().iter().copied().collect();
    let dtype = dtype_to_storage_type(out_dtype);

    launch_depthwise_tiled::<R>(
        &client,
        input.binding(),
        weight.binding(),
        out.clone().binding(),
        &in_shape,
        &w_shape,
        &out_shape,
        args,
        options.groups,
        dtype,
        tiling,
    )?;

    Ok(out)
}
