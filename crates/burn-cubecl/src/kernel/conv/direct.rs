use crate::{
    kernel::into_contiguous_aligned, ops::numeric::empty_device_dtype, tensor::CubeTensor,
};
use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::ops::{ConvOptions, conv::calculate_conv_output_sizes};
use cubek::convolution::{
    ConvolutionArgs, DirectTensors, components::ConvSetupError, launch_direct,
};

/// Perform a convolution using the direct convolution algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_direct<const N: usize>(
    mut input: CubeTensor,
    mut weight: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvOptions<N>,
) -> Result<CubeTensor, ConvSetupError> {
    let out_dtype = input.dtype;
    let rank = input.meta.shape().num_dims();
    let dim_c = rank - 1;

    // We only care about the channels here, everything else can be permuted
    if input.meta.strides()[dim_c] != 1 {
        input = into_contiguous_aligned(input);
    }
    if weight.meta.strides()[dim_c] != 1 {
        weight = into_contiguous_aligned(weight);
    }

    let batch_size = input.meta.shape()[0];
    let in_shape = &input.meta.shape()[1..dim_c];
    let out_channels = weight.meta.shape()[0];
    let kernel_shape = &weight.meta.shape()[1..dim_c];

    let out_size = calculate_conv_output_sizes(
        kernel_shape,
        &options.stride,
        &options.padding,
        &options.dilation,
        in_shape,
    );

    let mut shape_out = vec![batch_size];
    shape_out.extend(out_size.iter().copied());
    shape_out.push(out_channels);

    let out = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        shape_out.into(),
        out_dtype,
    );

    let padding = options.padding_begin();
    let args = ConvolutionArgs::<N> {
        stride: options.stride,
        padding: core::array::from_fn(|i| padding[i]),
        dilation: options.dilation,
    };

    let client = input.client.clone();
    let dtype = dtype_to_storage_type(out_dtype);

    let tensors = DirectTensors {
        input: input.binding(),
        weight: weight.binding(),
        bias: bias.map(|bias| bias.binding()),
        out: out.clone().binding(),
    };

    launch_direct::<N>(&client, tensors, args, options.groups, dtype)?;

    Ok(out)
}
