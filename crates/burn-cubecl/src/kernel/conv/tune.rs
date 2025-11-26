use burn_tensor::ops::ConvOptions;
use cubecl::tune::{LocalTuner, Tunable, TunableSet, anchor, local_tuner};

use crate::{
    CubeAutotuneKey, CubeRuntime, CubeTuneId,
    kernel::conv::{
        conv_direct, conv_gemm_cyclic, conv_gemm_tma, conv_gemm_tma_multi_stage, conv_im2col,
        conv_im2col_1x1,
    },
    tensor::CubeTensor,
};

use super::ConvAutotuneKey;

/// Executes autotune on conv2d operations
pub fn conv_autotune<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weights: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> CubeTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key::<R, N>, create_conv_input::<R, N>)
            .with(Tunable::new(conv_direct::<R, N>))
            .with(Tunable::new(conv_im2col_1x1::<R, N>))
            .with(Tunable::new(conv_im2col::<R, N>))
            .with(Tunable::new(conv_gemm_cyclic::<R, N>))
            .with(Tunable::new(conv_gemm_tma::<R, N>))
            .with(Tunable::new(conv_gemm_tma_multi_stage::<R, N>))
    });

    TUNER.execute(
        &CubeTuneId::new(&input.client, &input.device),
        &client,
        tunables,
        (input, weights, bias, options),
    )
}

pub fn create_conv_input<R: CubeRuntime, const N: usize>(
    _key: &CubeAutotuneKey,
    input: &CubeTensor<R>,
    weights: &CubeTensor<R>,
    bias: &Option<CubeTensor<R>>,
    options: &ConvOptions<N>,
) -> (
    CubeTensor<R>,
    CubeTensor<R>,
    Option<CubeTensor<R>>,
    ConvOptions<N>,
) {
    (
        input.clone(),
        weights.clone(),
        bias.clone(),
        options.clone(),
    )
}

fn create_key<R: CubeRuntime, const N: usize>(
    input: &CubeTensor<R>,
    weights: &CubeTensor<R>,
    bias: &Option<CubeTensor<R>>,
    options: &ConvOptions<N>,
) -> CubeAutotuneKey {
    let dtype = input.dtype;
    let rank = input.shape.num_dims();
    let dim_c = rank - 1;

    let batch_size = input.shape[0];
    let in_channels = input.shape[dim_c];
    let out_channels = weights.shape[0];

    let kernel_size = weights.shape[1..dim_c].to_vec();
    let in_shape = input.shape[1..dim_c]
        .iter()
        .map(|shape| anchor(*shape, None, None, None))
        .collect();

    let ConvOptions {
        stride,
        padding,
        dilation,
        groups,
    } = options.clone();
    CubeAutotuneKey::Conv2d(ConvAutotuneKey::new(
        kernel_size,
        stride.to_vec(),
        padding.to_vec(),
        dilation.to_vec(),
        groups,
        in_channels,
        out_channels,
        in_shape,
        batch_size,
        bias.is_some(),
        dtype,
    ))
}
