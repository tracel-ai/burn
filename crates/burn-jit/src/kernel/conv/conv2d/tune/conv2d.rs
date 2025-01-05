use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    ElementConversion, Shape,
};
use cubecl::{
    ir::{Elem, FloatKind},
    tf32, tune,
    tune::{local_tuner, tune_with, LocalTuner},
};
use half::f16;

use super::Conv2dAutotuneKey;
use crate::{
    kernel::{
        conv::{
            algorithm::{Algorithm, ImplicitCmmaConv},
            batches_per_run, can_do_implicit_gemm,
            conv2d::gemm::base::ConvolutionProblem,
            conv2d_direct, conv2d_gemm_cmma_balanced, conv2d_gemm_cmma_large_m, conv2d_im2col,
            conv2d_implicit_gemm, has_tf32,
            precision::ConvPrecision,
            problem_from_key,
            selection::{Balanced, ConvSelector, Large},
        },
        prng::random_uniform,
    },
    tensor::JitTensor,
    FloatElement, JitAutotuneKey, JitRuntime, JitTuneId,
};

/// Executes autotune on conv2d operations
pub fn conv2d_autotune<R: JitRuntime, E: FloatElement>(
    input: JitTensor<R>,
    weights: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvOptions<2>,
) -> JitTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<JitAutotuneKey, JitTuneId> = local_tuner!();

    TUNER.execute(
        &JitTuneId::new::<R>(&input.device),
        &client,
        Box::new(Conv2dOperations::<R, E>::new(input, weights, bias, options)),
    )
}

#[tune(
    operations(
        conv2d_direct,
        conv2d_im2col,
        conv2d_implicit_gemm,
        conv2d_gemm_cmma_large_m,
        conv2d_gemm_cmma_balanced
    ),
    create_key = create_key::<R, E>,
    should_run = should_run
)]
pub fn conv2d_operations<R: JitRuntime, E: FloatElement>(
    key: JitAutotuneKey,
    input: JitTensor<R>,
    weights: JitTensor<R>,
    bias: Option<JitTensor<R>>,
    options: ConvOptions<2>,
) -> JitTensor<R> {
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

macro_rules! check_algo {
    ($algo:tt, $float:ty, $input:expr, $problem:expr) => {
        match (<$float>::as_elem_native_unchecked(), has_tf32(&$input)) {
            (Elem::Float(FloatKind::F32), true) => {
                can_launch::<$algo, R, ($float, tf32, f32)>($input, $problem)
            }
            (Elem::Float(FloatKind::Flex32), _) => {
                can_launch::<$algo, R, ($float, f16, f32)>($input, $problem)
            }
            _ => can_launch::<$algo, R, ($float, $float, f32)>($input, $problem),
        }
    };

    ($algo:tt, $input:expr, $problem:expr) => {
        let plane_dim = 32;
        let conv_problem = $problem;

        let (selection, config_input) = $algo::select_kernel::<R, CS>(plane_dim);
        let cube_dim = ImplicitCmmaConv::cube_dim(&selection);
        let cube_count = ImplicitCmmaConv::cube_count(&selection, &conv_problem);

        let advanced_config = Default::default();
        let config = ImplicitCmmaConv::make_config(
            config_input,
            &conv_problem,
            &cube_dim,
            &cube_count,
            &advanced_config,
        );

        match config {
            Ok(config) => ImplicitCmmaConv::can_launch::<R, CS>(
                &op.input.client,
                &conv_problem,
                &config,
                &selection,
            ),
            Err(_) => false,
        }
    };
}

fn should_run<R: JitRuntime, F: FloatElement>(
    op: &Conv2dOperations<R, F>,
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

    let conv_problem = problem_from_key::<R, F>(key, out_h, out_w);

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
            &op.input.client,
        ),
        // GEMM large m
        3 => check_algo!(Large, F, &op.input, &conv_problem),
        // GEMM balanced
        4 => check_algo!(Balanced, F, &op.input, &conv_problem),
        _ => true,
    }
}

fn can_launch<S: ConvSelector<ImplicitCmmaConv>, R: JitRuntime, CS: ConvPrecision>(
    input: &JitTensor<R>,
    conv_problem: &ConvolutionProblem,
) -> bool {
    let plane_dim = 32;

    let (selection, config_input) = S::select_kernel::<R, CS>(plane_dim);
    let cube_dim = ImplicitCmmaConv::cube_dim(&selection);
    let cube_count = ImplicitCmmaConv::cube_count(&selection, conv_problem);

    let advanced_config = Default::default();
    let config = ImplicitCmmaConv::make_config(
        config_input,
        conv_problem,
        &cube_dim,
        &cube_count,
        &advanced_config,
    );

    match config {
        Ok(config) => {
            ImplicitCmmaConv::can_launch::<R, CS>(&input.client, conv_problem, &config, &selection)
        }
        Err(_) => false,
    }
}

fn create_key<R: JitRuntime, E: FloatElement>(
    input: &JitTensor<R>,
    weights: &JitTensor<R>,
    bias: &Option<JitTensor<R>>,
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
        E::dtype(),
    ))
}
