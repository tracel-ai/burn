use burn_tensor::{ElementConversion, Shape, ops::ConvOptions};
use cubecl::tune::{LocalTuner, Tunable, TunableSet, anchor, local_tuner};

use crate::{
    CubeAutotuneKey, CubeRuntime, CubeTuneId, FloatElement,
    kernel::{
        conv::{
            conv_direct, conv_gemm_cyclic, conv_gemm_tma, conv_gemm_tma_multi_stage, conv_im2col,
            conv_im2col_1x1,
        },
        prng::random_uniform,
    },
    tensor::CubeTensor,
};

use super::ConvAutotuneKey;

/// Executes autotune on conv2d operations
pub fn conv_autotune<R: CubeRuntime, E: FloatElement, const N: usize>(
    input: CubeTensor<R>,
    weights: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> CubeTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key::<R, E, N>, create_conv_input::<R, E, N>)
            .with(Tunable::new(conv_direct::<R, E, N>))
            .with(Tunable::new(conv_im2col_1x1::<R, E, N>))
            .with(Tunable::new(conv_im2col::<R, E, N>))
            .with(Tunable::new(conv_gemm_cyclic::<R, E, N>))
            .with(Tunable::new(conv_gemm_tma::<R, E, N>))
            .with(Tunable::new(conv_gemm_tma_multi_stage::<R, E, N>))
    });

    TUNER.execute(
        &CubeTuneId::new::<R>(&input.client, &input.device),
        &client,
        tunables,
        (input, weights, bias, options),
    )
}

pub fn create_conv_input<R: CubeRuntime, E: FloatElement, const N: usize>(
    key: &CubeAutotuneKey,
    input: &CubeTensor<R>,
    _weights: &CubeTensor<R>,
    _bias: &Option<CubeTensor<R>>,
    options: &ConvOptions<N>,
) -> (
    CubeTensor<R>,
    CubeTensor<R>,
    Option<CubeTensor<R>>,
    ConvOptions<N>,
) {
    let device = &input.device;
    let key = match key {
        CubeAutotuneKey::Conv2d(key) => key,
        _ => unreachable!(),
    };

    let rand_bounds: (E, E) = ((-1.0).elem::<E>(), (1.0).elem::<E>());

    let mut input_shape = vec![key.batch_size];
    input_shape.extend(key.shape.iter().copied());
    input_shape.push(key.in_channels);
    let input = random_uniform(input_shape.into(), device, rand_bounds.0, rand_bounds.1);

    let c_per_grp = key.in_channels / key.groups;
    let mut weight_shape = vec![key.out_channels];
    weight_shape.extend(key.kernel_size.iter().copied());
    weight_shape.push(c_per_grp);

    let weights = random_uniform(weight_shape.into(), device, rand_bounds.0, rand_bounds.1);

    let bias_shape = Shape::new([key.out_channels]);
    let bias = key
        .has_bias
        .then(|| random_uniform(bias_shape, device, rand_bounds.0, rand_bounds.1));

    (input, weights, bias, options.clone())
}

fn create_key<R: CubeRuntime, E: FloatElement, const N: usize>(
    input: &CubeTensor<R>,
    weights: &CubeTensor<R>,
    bias: &Option<CubeTensor<R>>,
    options: &ConvOptions<N>,
) -> CubeAutotuneKey {
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
        E::dtype(),
    ))
}
