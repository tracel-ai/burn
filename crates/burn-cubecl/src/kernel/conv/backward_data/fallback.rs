use burn_backend::{
    TensorMetadata,
    ops::{ConvOptions, ConvTransposeOptions, conv::calculate_padding_out},
};
use burn_std::Shape;
use cubek::convolution::components::ConvSetupError;

use crate::{
    kernel::{conv::conv_transpose2d, conv::conv_transpose3d, slice},
    ops::{permute_nchw_to_nhwc, permute_nhwc_to_nchw, reshape},
    tensor::CubeTensor,
};

pub(crate) fn conv_data_backward_fallback<const N_DIM: usize>(
    out_grad: CubeTensor,
    weights: CubeTensor,
    in_shape: Shape,
    options: ConvOptions<N_DIM>,
) -> Result<CubeTensor, ConvSetupError> {
    if options.is_asymmetric() {
        let original_shape = in_shape.clone();
        let mut padded_shape = original_shape.to_vec();
        for dim in 0..N_DIM {
            padded_shape[dim + 1] += options.padding[dim].0 + options.padding[dim].1;
        }

        let grad = conv_data_backward_fallback(
            out_grad,
            weights,
            padded_shape.into(),
            ConvOptions::new(options.stride, [0; N_DIM], options.dilation, options.groups),
        )?;
        let mut ranges = original_shape
            .iter()
            .map(|&size| 0..size)
            .collect::<Vec<_>>();
        for dim in 0..N_DIM {
            let begin = options.padding[dim].0;
            ranges[dim + 1] = begin..begin + original_shape[dim + 1];
        }
        return Ok(slice(grad, &ranges));
    }

    let dim_c = out_grad.rank();

    let kernel_size = &weights.meta.shape()[1..dim_c];
    let in_shape = &in_shape[1..dim_c];
    let out_shape = &out_grad.meta.shape()[1..dim_c];

    let mut padding_out = [0; N_DIM];

    for i in 0..N_DIM {
        padding_out[i] = calculate_padding_out(
            kernel_size[i],
            options.stride[i],
            options.padding_begin()[i],
            options.dilation[i],
            in_shape[i],
            out_shape[i],
        );
    }

    // Through `conv_transpose2d` so its autotune keeps both routes: the direct NHWC kernel
    // (its direct candidate) and col2im. 3D has no NHWC kernel and flips to NCHW below.
    match N_DIM {
        1 => {
            let out_grad = permute_nhwc_to_nchw(out_grad);
            let weights = permute_nhwc_to_nchw(weights);

            let in_grad = conv_transpose1d_from_conv_transpose2d(
                out_grad,
                weights,
                ConvTransposeOptions::new(
                    [options.stride[0]],
                    [options.padding_begin()[0]],
                    [padding_out[0]],
                    [options.dilation[0]],
                    options.groups,
                ),
            )?;

            Ok(permute_nchw_to_nhwc(in_grad))
        }
        2 => {
            let out_grad = permute_nhwc_to_nchw(out_grad);
            let weights = permute_nhwc_to_nchw(weights);

            let in_grad = conv_transpose2d(
                out_grad,
                weights,
                None,
                ConvTransposeOptions::new(
                    [options.stride[0], options.stride[1]],
                    [options.padding_begin()[0], options.padding_begin()[1]],
                    [padding_out[0], padding_out[1]],
                    [options.dilation[0], options.dilation[1]],
                    options.groups,
                ),
                Default::default(),
            )?;

            Ok(permute_nchw_to_nhwc(in_grad))
        }
        3 => {
            // `conv_transpose3d` is NCHW-only.
            let out_grad = permute_nhwc_to_nchw(out_grad);
            let weights = permute_nhwc_to_nchw(weights);

            let in_grad = conv_transpose3d(
                out_grad,
                weights,
                None,
                ConvTransposeOptions::new(
                    [options.stride[0], options.stride[1], options.stride[2]],
                    [
                        options.padding_begin()[0],
                        options.padding_begin()[1],
                        options.padding_begin()[2],
                    ],
                    [padding_out[0], padding_out[1], padding_out[2]],
                    [
                        options.dilation[0],
                        options.dilation[1],
                        options.dilation[2],
                    ],
                    options.groups,
                ),
                Default::default(),
            )?;

            Ok(permute_nchw_to_nhwc(in_grad))
        }
        _ => unimplemented!("Invalid dimensionality"),
    }
}

/// Runs a 1D transposition as a 2D one whose width is a single column.
fn conv_transpose1d_from_conv_transpose2d(
    x: CubeTensor,
    weight: CubeTensor,
    options: ConvTransposeOptions<1>,
) -> Result<CubeTensor, ConvSetupError> {
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
    let [batch_size, channels_out, height_out, _width_out] = tensor.shape().dims();
    Ok(reshape(
        tensor,
        Shape::from([batch_size, channels_out, height_out]),
    ))
}
