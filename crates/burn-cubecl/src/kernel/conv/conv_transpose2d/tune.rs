use burn_tensor::ops::ConvTransposeOptions;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, local_tuner};

use crate::{
    CubeAutotuneKey, CubeRuntime, CubeTuneId,
    kernel::conv::{ConvTranspose2dAutotuneKey, conv_transpose2d_col2im, conv_transpose2d_direct},
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
        TunableSet::new(create_key, create_transpose2d_input)
            .with(Tunable::new(conv_transpose2d_direct))
            .with(Tunable::new(conv_transpose2d_col2im))
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
    input: &CubeTensor<R>,
    weights: &CubeTensor<R>,
    bias: &Option<CubeTensor<R>>,
    options: &ConvTransposeOptions<2>,
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
    input: &CubeTensor<R>,
    weights: &CubeTensor<R>,
    bias: &Option<CubeTensor<R>>,
    options: &ConvTransposeOptions<2>,
) -> CubeAutotuneKey {
    let [batch_size, in_channels, height, width] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weights.shape.dims();
    let ConvTransposeOptions {
        stride,
        padding,
        dilation,
        groups,
        padding_out,
    } = options.clone();
    CubeAutotuneKey::ConvTranspose2d(ConvTranspose2dAutotuneKey::new(
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
