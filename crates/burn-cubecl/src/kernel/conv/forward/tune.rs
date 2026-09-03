use burn_backend::cubecl::dtype_to_storage_type;
use burn_backend::ops::ConvOptions;
use cubecl::{
    ir::ElemType,
    tune::{LocalTuner, Tunable, TunableSet, anchor, local_tuner},
};
use cubek::convolution::{AcceleratedTileKind, DepthwiseStrategy, DepthwiseTiling};

use crate::{
    CubeAutotuneKey, CubeRuntime, CubeTuneId,
    kernel::conv::{
        ConvAutotuneKey, conv_direct, conv_im2col_1x1, forward::depthwise::conv_depthwise,
        forward::implicit_gemm::*,
    },
    tensor::CubeTensor,
};

/// The tilings the depthwise routine is offered under, beside the one it picks for itself.
///
/// Chosen greedily against a sweep of the whole grid over EfficientNet-B4's depthwise layers:
/// these four are what closes the gap between the routine's own rule (16.6 ms of depthwise
/// convolution at batch 4) and picking the best tiling per shape (15.7 ms). Adding more moves it
/// by under 1%, and every one of them costs a dense convolution a setup call that declines.
const DEPTHWISE_8X4_LINED: DepthwiseStrategy = DepthwiseStrategy::Fixed(DepthwiseTiling {
    rows: 8,
    cols: 4,
    chans: 1,
    lines: 2,
});
const DEPTHWISE_2X4_SCALAR: DepthwiseStrategy = DepthwiseStrategy::Fixed(DepthwiseTiling {
    rows: 2,
    cols: 4,
    chans: 1,
    lines: 1,
});
const DEPTHWISE_8X2_LINED: DepthwiseStrategy = DepthwiseStrategy::Fixed(DepthwiseTiling {
    rows: 8,
    cols: 2,
    chans: 1,
    lines: 2,
});
const DEPTHWISE_4X2_SCALAR: DepthwiseStrategy = DepthwiseStrategy::Fixed(DepthwiseTiling {
    rows: 4,
    cols: 2,
    chans: 1,
    lines: 1,
});

/// Executes autotune on convolution operations
pub fn conv_autotune<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> CubeTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tune_id = CubeTuneId::new(&input.client, &input.device);
    let tunables = TUNER.init(&tune_id, || {
        TunableSet::new(create_key::<R, N>, create_conv_input::<R, N>)
            .with(Tunable::new(
                "conv_direct",
                |(input, weight, bias, options)| conv_direct::<R, N>(input, weight, bias, options),
            ))
            // Declines with `NotDepthwise` on anything that is not one filter per channel, so
            // each of these costs a dense shape nothing but the setup call that rejects it.
            //
            // Several tilings rather than one because they are not close: over EfficientNet-B4's
            // depthwise layers, the best tile per shape beats the best single tile by 8%, and
            // which one wins swings with the window's depth and the block's width in a way the
            // shape does not predict. The routine's own default is the fallback when no tuning
            // has run; these are what let a run that does tune land on the right one.
            .with(Tunable::new(
                "conv_depthwise",
                |(input, weight, bias, options)| {
                    conv_depthwise::<R, N>(input, weight, bias, options, DepthwiseStrategy::Routine)
                },
            ))
            .with(Tunable::new(
                "conv_depthwise_8x4_lined",
                |(input, weight, bias, options)| {
                    conv_depthwise::<R, N>(input, weight, bias, options, DEPTHWISE_8X4_LINED)
                },
            ))
            .with(Tunable::new(
                "conv_depthwise_2x4_scalar",
                |(input, weight, bias, options)| {
                    conv_depthwise::<R, N>(input, weight, bias, options, DEPTHWISE_2X4_SCALAR)
                },
            ))
            .with(Tunable::new(
                "conv_depthwise_8x2_lined",
                |(input, weight, bias, options)| {
                    conv_depthwise::<R, N>(input, weight, bias, options, DEPTHWISE_8X2_LINED)
                },
            ))
            .with(Tunable::new(
                "conv_depthwise_4x2_scalar",
                |(input, weight, bias, options)| {
                    conv_depthwise::<R, N>(input, weight, bias, options, DEPTHWISE_4X2_SCALAR)
                },
            ))
            .with(Tunable::new(
                "conv_im2col_1x1",
                |(input, weight, bias, options)| {
                    conv_im2col_1x1::<R, N>(input, weight, bias, options)
                },
            ))
            .with(Tunable::new(
                "simple_sync_cmma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_sync(input, weight, bias, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_sync_mma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_sync(input, weight, bias, options, AcceleratedTileKind::Mma)
                },
            ))
            .with(Tunable::new(
                "simple_async_cmma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_async(input, weight, bias, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_async_mma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_async(input, weight, bias, options, AcceleratedTileKind::Mma)
                },
            ))
            .with(Tunable::new(
                "simple_tma_cmma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_tma(input, weight, bias, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_tma_mma",
                |(input, weight, bias, options)| {
                    conv_gemm_simple_tma(input, weight, bias, options, AcceleratedTileKind::Mma)
                },
            ))
    });

    TUNER.execute(&tune_id, &client, tunables, (input, weight, bias, options))
}

pub fn create_conv_input<R: CubeRuntime, const N: usize>(
    _key: &CubeAutotuneKey,
    (input, weights, bias, options): &(
        CubeTensor<R>,
        CubeTensor<R>,
        Option<CubeTensor<R>>,
        ConvOptions<N>,
    ),
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
    (input, weights, bias, options): &(
        CubeTensor<R>,
        CubeTensor<R>,
        Option<CubeTensor<R>>,
        ConvOptions<N>,
    ),
) -> CubeAutotuneKey {
    let dtype = input.dtype;
    let rank = input.meta.shape().num_dims();
    let dim_c = rank - 1;

    let batch_size = input.meta.shape()[0];
    let in_channels = input.meta.shape()[dim_c];
    let out_channels = weights.meta.shape()[0];

    let kernel_size = weights.meta.shape()[1..dim_c].to_vec();
    let in_shape = input.meta.shape()[1..dim_c]
        .iter()
        .map(|shape| anchor(*shape, None, None, None))
        .collect();

    let ConvOptions {
        stride,
        padding,
        dilation,
        groups,
    } = options.clone();

    let lhs_stride_align = if input.meta.strides()[dim_c] == 1 {
        stride_align(input.meta.strides(), dtype_to_storage_type(input.dtype))
    } else {
        0
    };
    let lhs_shape_align = pow2_factor(in_channels).min(lhs_stride_align);
    let rhs_stride_align = if weights.meta.strides()[dim_c] == 1 {
        stride_align(weights.meta.strides(), dtype_to_storage_type(weights.dtype))
    } else {
        0
    };
    let rhs_shape_align = pow2_factor(in_channels).min(rhs_stride_align);

    CubeAutotuneKey::Conv(ConvAutotuneKey::new(
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
        lhs_shape_align,
        lhs_stride_align,
        rhs_shape_align,
        rhs_stride_align,
    ))
}

/// Maximum factor relevant for strides. Currently set to 2^10 because that's 128-byte swizzle's
/// repeat number, so it's the largest align that can have performance impacts.
const MAX_STRIDE_FACTOR: u32 = 10;

/// Defines the non-contiguous stride alignment in terms of powers of two
fn stride_align(strides: &[usize], elem: ElemType) -> u8 {
    let max = MAX_STRIDE_FACTOR;
    let dim_c = strides.len() - 1;
    let factor = strides[..dim_c]
        .iter()
        .map(|it| (*it * elem.size_bits()) / 8)
        .map(|it| it.trailing_zeros())
        .min()
        .unwrap_or(max);
    factor.min(max) as u8
}

/// Defines the potential vectorization.
fn pow2_factor(axis: usize) -> u8 {
    axis.trailing_zeros().min(4) as u8
}
