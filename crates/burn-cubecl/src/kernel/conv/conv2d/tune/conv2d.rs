use burn_tensor::{ops::ConvOptions, ElementConversion, Shape};
use cubecl::tune::{local_tuner, LocalTuner, TunableSet};

use super::Conv2dAutotuneKey;
use crate::{
    kernel::{
        conv::{
            conv2d_direct, conv2d_gemm_cmma_balanced, conv2d_gemm_cmma_large_m, conv2d_im2col,
            conv2d_implicit_gemm,
        },
        prng::random_uniform,
    },
    tensor::CubeTensor,
    CubeAutotuneKey, CubeRuntime, CubeTuneId, FloatElement,
};

/// Executes autotune on conv2d operations
pub fn conv2d_autotune<R: CubeRuntime, E: FloatElement>(
    input: CubeTensor<R>,
    weights: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<2>,
) -> CubeTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TunableSet::new(create_key::<R, E>, create_conv2d_input::<R, E>)
        .with_tunable(conv2d_direct::<R, E>)
        .with_tunable(conv2d_im2col::<R, E>)
        .with_tunable(conv2d_implicit_gemm::<R, E>)
        .with_tunable(conv2d_gemm_cmma_large_m::<R, E>)
        .with_tunable(conv2d_gemm_cmma_balanced::<R, E>);

    TUNER.execute(
        &CubeTuneId::new::<R>(&input.device),
        &client,
        &tunables,
        (input, weights, bias, options),
    )
}

pub fn create_conv2d_input<R: CubeRuntime, E: FloatElement>(
    key: &CubeAutotuneKey,
    input: &CubeTensor<R>,
    _weights: &CubeTensor<R>,
    _bias: &Option<CubeTensor<R>>,
    options: &ConvOptions<2>,
) -> (
    CubeTensor<R>,
    CubeTensor<R>,
    Option<CubeTensor<R>>,
    ConvOptions<2>,
) {
    let device = &input.device;
    let key = match key {
        CubeAutotuneKey::Conv2d(key) => key,
        _ => unreachable!(),
    };

    let random_bounds: (E, E) = ((-1.0).elem::<E>(), (1.0).elem::<E>());
    let input_shape = Shape::new([key.batch_size, key.in_channels, key.height, key.width]);
    let input = random_uniform(input_shape, device, random_bounds.0, random_bounds.1);
    let c_per_grp = key.in_channels / key.groups;
    let [kernel_h, kernel_w] = key.kernel_size;
    let weight_shape = Shape::new([key.out_channels, c_per_grp, kernel_h, kernel_w]);
    let weights = random_uniform(weight_shape, device, random_bounds.0, random_bounds.1);
    let bias_shape = Shape::new([key.out_channels]);
    let bias = key
        .has_bias
        .then(|| random_uniform(bias_shape, device, random_bounds.0, random_bounds.1));

    (input, weights, bias, options.clone())
}

fn create_key<R: CubeRuntime, E: FloatElement>(
    input: &CubeTensor<R>,
    weights: &CubeTensor<R>,
    bias: &Option<CubeTensor<R>>,
    options: &ConvOptions<2>,
) -> CubeAutotuneKey {
    let [batch_size, in_channels, height, width] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weights.shape.dims();
    let ConvOptions {
        stride,
        padding,
        dilation,
        groups,
    } = options.clone();
    CubeAutotuneKey::Conv2d(Conv2dAutotuneKey::new(
        [kernel_h, kernel_w],
        stride,
        padding,
        dilation,
        groups,
        in_channels,
        out_channels,
        height,
        width,
        batch_size,
        bias.is_some(),
        E::dtype(),
    ))
}
