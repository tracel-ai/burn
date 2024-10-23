use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    ElementConversion, Shape,
};
use cubecl::{
    tune,
    tune::{local_tuner, tune_with, LocalTuner},
};

use crate::{
    kernel::{
        conv::{
            batches_per_run, can_do_implicit_gemm, conv2d_direct, conv2d_im2col,
            conv2d_implicit_gemm,
        },
        prng::random_uniform,
    },
    tensor::JitTensor,
    FloatElement, IntElement, JitAutotuneKey, JitRuntime, JitTuneId,
};

use super::Conv2dAutotuneKey;

/// Executes autotune on conv2d operations
pub fn conv2d_autotune<R: JitRuntime, E: FloatElement, I: IntElement>(
    input: JitTensor<R, E>,
    weights: JitTensor<R, E>,
    bias: Option<JitTensor<R, E>>,
    options: ConvOptions<2>,
) -> JitTensor<R, E> {
    let client = input.client.clone();

    static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!();

    TUNER.execute(
        &JitTuneId::new::<R>(&input.device),
        &client,
        Box::new(Conv2dOperations::<R, E, I>::new(
            input, weights, bias, options,
        )),
    )
}

#[tune(
    operations(conv2d_direct, conv2d_im2col, conv2d_implicit_gemm),
    create_key = create_key,
    should_run = should_run
)]
pub fn conv2d_operations<R: JitRuntime, E: FloatElement, I: IntElement>(
    key: JitAutotuneKey,
    input: JitTensor<R, E>,
    weights: JitTensor<R, E>,
    bias: Option<JitTensor<R, E>>,
    options: ConvOptions<2>,
) -> JitTensor<R, E> {
    let device = &input.device;
    let key = match key {
        JitAutotuneKey::Conv2d(key) => key,
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

    tune_with!(input, weights, bias, options)
}

fn should_run<R: JitRuntime, F: FloatElement, I: IntElement>(
    op: &Conv2dOperations<R, F, I>,
    key: &JitAutotuneKey,
    index: usize,
) -> bool {
    let key = match key {
        JitAutotuneKey::Conv2d(key) => key,
        _ => unreachable!(),
    };

    let out_h = calculate_conv_output_size(
        key.kernel_size[0],
        key.stride[0],
        key.padding[0],
        key.dilation[0],
        key.height,
    );
    let out_w = calculate_conv_output_size(
        key.kernel_size[1],
        key.stride[1],
        key.padding[1],
        key.dilation[1],
        key.width,
    );

    match index {
        // im2col
        1 => batches_per_run(key.batch_size, out_h, out_w).is_some(),
        // Implicit gemm.
        2 => can_do_implicit_gemm::<R, F>(
            key.batch_size,
            key.in_channels,
            key.out_channels,
            key.kernel_size,
            op.options.groups,
            out_h,
            out_w,
            &op.input.device,
        ),
        _ => true,
    }
}

fn create_key<R: JitRuntime, E: FloatElement>(
    input: &JitTensor<R, E>,
    weights: &JitTensor<R, E>,
    bias: &Option<JitTensor<R, E>>,
    options: &ConvOptions<2>,
) -> JitAutotuneKey {
    let [batch_size, in_channels, height, width] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weights.shape.dims();
    let ConvOptions {
        stride,
        padding,
        dilation,
        groups,
    } = options.clone();
    JitAutotuneKey::Conv2d(Conv2dAutotuneKey::new(
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
    ))
}
