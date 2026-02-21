use burn_backend::{
    TensorMetadata,
    ops::{ConvOptions, ConvTransposeOptions},
};
use burn_std::Shape;
use cubek::convolution::components::ConvSetupError;

use crate::{
    CubeRuntime,
    kernel::conv::{conv_transpose2d, conv_transpose3d},
    ops::{permute_nchw_to_nhwc, permute_nhwc_to_nchw, reshape},
    tensor::CubeTensor,
};

pub(crate) fn conv_data_backward_fallback<R: CubeRuntime, const N_DIM: usize>(
    out_grad: CubeTensor<R>,
    weights: CubeTensor<R>,
    in_shape: Shape,
    options: ConvOptions<N_DIM>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let dim_c = out_grad.rank();

    let kernel_size = &weights.meta.shape()[1..dim_c];
    let in_shape = &in_shape[1..dim_c];
    let out_shape = &out_grad.meta.shape()[1..dim_c];

    let mut padding_out = [0; N_DIM];

    for i in 0..N_DIM {
        padding_out[i] = calculate_padding_out(
            kernel_size[i],
            options.stride[i],
            options.padding[i],
            options.dilation[i],
            in_shape[i],
            out_shape[i],
        );
    }

    // We don't yet have NHWC kernels for conv_transpose so need to do this.
    // Should eventually use NHWC kernels instead
    let out_grad = permute_nhwc_to_nchw(out_grad);
    let weights = permute_nhwc_to_nchw(weights);

    let in_grad = match N_DIM {
        1 => conv_transpose1d_from_conv_transpose2d(
            out_grad,
            weights,
            ConvTransposeOptions::new(
                [options.stride[0]],
                [options.padding[0]],
                [padding_out[0]],
                [options.dilation[0]],
                options.groups,
            ),
        ),
        2 => conv_transpose2d(
            out_grad,
            weights,
            None,
            ConvTransposeOptions::new(
                [options.stride[0], options.stride[1]],
                [options.padding[0], options.padding[1]],
                [padding_out[0], padding_out[1]],
                [options.dilation[0], options.dilation[1]],
                options.groups,
            ),
            Default::default(),
        ),
        3 => Ok(conv_transpose3d(
            out_grad,
            weights,
            None,
            ConvTransposeOptions::new(
                [options.stride[0], options.stride[1], options.stride[2]],
                [options.padding[0], options.padding[1], options.padding[2]],
                [padding_out[0], padding_out[1], padding_out[2]],
                [
                    options.dilation[0],
                    options.dilation[1],
                    options.dilation[2],
                ],
                options.groups,
            ),
        )
        .unwrap()),
        _ => unimplemented!("Invalid dimensionality"),
    }?;
    Ok(permute_nchw_to_nhwc(in_grad))
}

fn calculate_padding_out(
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    size_in: usize,
    size_out: usize,
) -> usize {
    if stride <= 1 {
        return 0;
    }

    let out = 1
        + ((size_in + 2 * padding - dilation * (kernel_size - 1) - 1) as f64 / stride as f64).ceil()
            as usize;
    i64::max(0, out as i64 - size_out as i64) as usize
}

fn conv_transpose1d_from_conv_transpose2d<R: CubeRuntime>(
    x: CubeTensor<R>,
    weight: CubeTensor<R>,
    options: ConvTransposeOptions<1>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let [channels_in, channels_out, kernel_size] = weight.shape().dims();
    let [batch_size, _channels_in, length_in] = x.shape().dims();

    let weight = reshape(
        weight,
        Shape::new([channels_in, channels_out, kernel_size, 1]),
    );
    let x = reshape(x, Shape::new([batch_size, channels_in, length_in, 1]));

    let tensor = conv_transpose2d(
        x,
        weight,
        None,
        ConvTransposeOptions::new(
            [options.stride[0], 1],
            [options.padding[0], 0],
            [options.padding_out[0], 0],
            [options.dilation[0], 1],
            options.groups,
        ),
        Default::default(),
    )?;
    let [batch_size, channels_out, height_out, _weight_out] = tensor.shape().dims();
    Ok(reshape(
        tensor,
        Shape::from([batch_size, channels_out, height_out]),
    ))
}
