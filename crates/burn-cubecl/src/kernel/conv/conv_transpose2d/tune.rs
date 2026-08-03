use burn_backend::ops::ConvTransposeOptions;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};

use crate::{
    CubeAutotuneKey, CubeRuntime, CubeTuneId,
    kernel::conv::{
        ConvTranspose2dAutotuneKey,
        bounds::{ConvDims, with_conv_transpose2d_bounds},
        conv_transpose2d_col2im, conv_transpose2d_direct,
    },
    tensor::CubeTensor,
};

/// Executes autotune on conv2d operations
pub fn conv_transpose2d_autotune<R: CubeRuntime>(
    input: CubeTensor<R>,
    weights: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvTransposeOptions<2>,
) -> CubeTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tune_set = TUNER.init(|| {
        let set = TunableSet::new(create_key::<R>, create_transpose2d_input::<R>);
        let set = with_conv_transpose2d_bounds(set, |(input, weights, _bias, options)| {
            (
                &input.client,
                ConvDims {
                    batch_size: input.meta.shape()[0],
                    in_channels: input.meta.shape()[1],
                    // The weight is laid out as `[in_channels, out_channels / groups, kh, kw]`.
                    out_channels: weights.meta.shape()[1] * options.groups,
                    in_shape: &input.meta.shape()[2..],
                },
            )
        });

        set.with(Tunable::new(
            "conv_transpose2d_direct",
            |(input, weights, bias, options)| {
                conv_transpose2d_direct::<R>(input, weights, bias, options)
            },
        ))
        .with(Tunable::new(
            "conv_transpose2d_col2im",
            |(input, weights, bias, options)| {
                conv_transpose2d_col2im::<R>(input, weights, bias, options)
            },
        ))
    });

    TUNER.execute(
        &CubeTuneId::new(&input.client, &input.device),
        &client,
        tune_set,
        (input, weights, bias, options),
    )
}

pub fn create_transpose2d_input<R: CubeRuntime>(
    _key: &CubeAutotuneKey,
    (input, weights, bias, options): &(
        CubeTensor<R>,
        CubeTensor<R>,
        Option<CubeTensor<R>>,
        ConvTransposeOptions<2>,
    ),
) -> (
    CubeTensor<R>,
    CubeTensor<R>,
    Option<CubeTensor<R>>,
    ConvTransposeOptions<2>,
) {
    (
        input.clone(),
        weights.clone(),
        bias.clone(),
        options.clone(),
    )
}

fn create_key<R: CubeRuntime>(
    (input, weights, bias, options): &(
        CubeTensor<R>,
        CubeTensor<R>,
        Option<CubeTensor<R>>,
        ConvTransposeOptions<2>,
    ),
) -> CubeAutotuneKey {
    let [batch_size, in_channels, height, width] = input.meta.shape().dims();
    let [out_channels, _, kernel_h, kernel_w] = weights.meta.shape().dims();
    let ConvTransposeOptions {
        stride,
        padding,
        dilation,
        groups,
        padding_out,
    } = options.clone();
    CubeAutotuneKey::ConvTranspose(ConvTranspose2dAutotuneKey::new(
        [kernel_h, kernel_w],
        stride,
        padding,
        padding_out,
        dilation,
        groups,
        in_channels,
        out_channels,
        height,
        width,
        batch_size,
        bias.is_some(),
        input.dtype,
    ))
}
