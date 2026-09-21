use burn_backend::{
    TensorMetadata,
    ops::{ConvOptions, ConvTransposeOptions, conv::calculate_padding_out},
};
use burn_std::Shape;
use cubek::convolution::components::ConvSetupError;

use crate::{
    kernel::{
        conv::{conv_transpose2d_direct_nhwc, conv_transpose3d},
        slice,
    },
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

    // 1D and 2D stay in NHWC, the layout the rest of the convolution stack works in and the one
    // `conv_transpose2d_direct_nhwc` reads coalesced. 3D has no NHWC kernel yet and flips below.
    //
    // The direct kernel rather than the strategy dispatch: the only other transposition
    // `ConvTranspose2dStrategy` offers is col2im, which is NCHW, and the GEMM routes for a data
    // gradient are already candidates one level up in `dgrad_autotune`. Choosing here would only
    // nest a second autotune inside one of that set's candidates.
    match N_DIM {
        1 => conv_transpose1d_from_conv_transpose2d_nhwc(
            out_grad,
            weights,
            ConvTransposeOptions::new(
                [options.stride[0]],
                [options.padding_begin()[0]],
                [padding_out[0]],
                [options.dilation[0]],
                options.groups,
            ),
        ),
        2 => conv_transpose2d_direct_nhwc(
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
        ),
        3 => {
            // `conv_transpose3d` is NCHW-only, so this dimensionality still pays for the round
            // trip. `conv_transpose2d_direct_nhwc` is the shape the fix would take here too.
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
            )
            .unwrap();

            Ok(permute_nchw_to_nhwc(in_grad))
        }
        _ => unimplemented!("Invalid dimensionality"),
    }
}

/// Runs a 1D transposition as a 2D one whose width is a single column.
///
/// The unit axis goes between the length and the channels so the channel axis stays last, which
/// is the order the 2D kernel decomposes. Its `x_start..x_end` loop then runs exactly once per
/// output element.
///
/// Logical order only: these tensors were permuted from NCHW, so the channel axis is still the
/// strided one and the kernel materializes it. Placing the unit axis correctly is what lets that
/// one copy be the whole cost, rather than a copy plus a kernel reading against its layout.
fn conv_transpose1d_from_conv_transpose2d_nhwc(
    x: CubeTensor,
    weight: CubeTensor,
    options: ConvTransposeOptions<1>,
) -> Result<CubeTensor, ConvSetupError> {
    let [channels_in, kernel_size, channels_out] = weight.shape().dims();
    let [batch_size, length_in, _channels_in] = x.shape().dims();

    let weight = reshape(
        weight,
        Shape::new([channels_in, kernel_size, 1, channels_out]),
    );
    let x = reshape(x, Shape::new([batch_size, length_in, 1, channels_in]));

    let tensor = conv_transpose2d_direct_nhwc(
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
    )?;
    let [batch_size, height_out, _width_out, channels_out] = tensor.shape().dims();
    Ok(reshape(
        tensor,
        Shape::from([batch_size, height_out, channels_out]),
    ))
}
