use burn_backend::ops::ConvTransposeOptions;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};

use crate::{
    CubeAutotuneKey, CubeTuneId,
    kernel::conv::{ConvTranspose2dAutotuneKey, conv_transpose2d_col2im, conv_transpose2d_direct},
    tensor::CubeTensor,
};

/// Executes autotune on conv2d operations
pub fn conv_transpose2d_autotune(
    input: CubeTensor,
    weights: CubeTensor,
    bias: Option<CubeTensor>,
    options: ConvTransposeOptions<2>,
) -> CubeTensor {
    let client = input.client.clone();

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tune_id = CubeTuneId::new(&input.client, &input.device);
    let tune_set = TUNER.init(&tune_id, || {
        TunableSet::new(create_key, create_transpose2d_input)
            .with(Tunable::new(
                "conv_transpose2d_direct",
                |(input, weights, bias, options)| {
                    conv_transpose2d_direct(input, weights, bias, options)
                },
            ))
            .with(Tunable::new(
                "conv_transpose2d_col2im",
                |(input, weights, bias, options)| {
                    conv_transpose2d_col2im(input, weights, bias, options)
                },
            ))
    });

    TUNER.execute(&tune_id, &client, tune_set, (input, weights, bias, options))
}

pub fn create_transpose2d_input(
    _key: &CubeAutotuneKey,
    (input, weights, bias, options): &(
        CubeTensor,
        CubeTensor,
        Option<CubeTensor>,
        ConvTransposeOptions<2>,
    ),
) -> (
    CubeTensor,
    CubeTensor,
    Option<CubeTensor>,
    ConvTransposeOptions<2>,
) {
    (
        input.clone(),
        weights.clone(),
        bias.clone(),
        options.clone(),
    )
}

fn create_key(
    (input, weights, bias, options): &(
        CubeTensor,
        CubeTensor,
        Option<CubeTensor>,
        ConvTransposeOptions<2>,
    ),
) -> CubeAutotuneKey {
    let [batch_size, in_channels, height, width] = input.meta.shape().dims();
    let [_, out_channels_per_group, kernel_h, kernel_w] = weights.meta.shape().dims();
    let ConvTransposeOptions {
        stride,
        padding,
        dilation,
        groups,
        padding_out,
    } = options.clone();
    let out_channels = out_channels_per_group * groups;
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
