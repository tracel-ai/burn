use burn_backend::ops::ConvOptions;
use cubecl::{
    ir::StorageType,
    tune::{LocalTuner, Tunable, TunableSet, anchor, local_tuner},
};
use cubek::convolution::AcceleratedTileKind;

use crate::{
    CubeAutotuneKey, CubeRuntime, CubeTuneId,
    kernel::conv::{ConvAutotuneKey, conv_direct, conv_im2col_1x1, forward::implicit_gemm::*},
    tensor::CubeTensor,
};

/// Executes autotune on convolution operations
pub fn conv_autotune<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> CubeTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key::<R, N>, create_conv_input::<R, N>)
            .with(Tunable::new("conv_direct", conv_direct::<R, N>))
            .with(Tunable::new("conv_im2col_1x1", conv_im2col_1x1::<R, N>))
            .with(Tunable::new(
                "simple_sync_cmma",
                |input, weight, bias, options| {
                    conv_gemm_simple_sync(input, weight, bias, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_sync_mma",
                |input, weight, bias, options| {
                    conv_gemm_simple_sync(input, weight, bias, options, AcceleratedTileKind::Mma)
                },
            ))
            .with(Tunable::new(
                "simple_async_cmma",
                |input, weight, bias, options| {
                    conv_gemm_simple_async(input, weight, bias, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_async_mma",
                |input, weight, bias, options| {
                    conv_gemm_simple_async(input, weight, bias, options, AcceleratedTileKind::Mma)
                },
            ))
            .with(Tunable::new(
                "simple_tma_cmma",
                |input, weight, bias, options| {
                    conv_gemm_simple_tma(input, weight, bias, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_tma_mma",
                |input, weight, bias, options| {
                    conv_gemm_simple_tma(input, weight, bias, options, AcceleratedTileKind::Mma)
                },
            ))
    });

    TUNER.execute(
        &CubeTuneId::new(&input.client, &input.device),
        &client,
        tunables,
        (input, weight, bias, options),
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
        ..
    } = options.clone();

    let lhs_stride_align = if input.strides[dim_c] == 1 {
        stride_align(&input.strides, input.dtype.into())
    } else {
        0
    };
    let lhs_shape_align = pow2_factor(in_channels).min(lhs_stride_align);
    let rhs_stride_align = if weights.strides[dim_c] == 1 {
        stride_align(&weights.strides, weights.dtype.into())
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
fn stride_align(strides: &[usize], elem: StorageType) -> u8 {
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
