use burn_backend::ops::ConvOptions;
use burn_std::Shape;
use cubecl::{
    ir::StorageType,
    tune::{LocalTuner, Tunable, TunableSet, anchor, local_tuner},
};
use cubek::convolution::AcceleratedTileKind;

use crate::{
    CubeAutotuneKey, CubeRuntime, CubeTuneId,
    kernel::conv::{
        ConvAutotuneKey,
        backward_weight::{fallback::conv_weight_backward_fallback, implicit_gemm::*},
    },
    tensor::CubeTensor,
};

/// Executes autotune on the weight gradients pass for convolution
pub fn wgrad_autotune<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    out_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N>,
) -> CubeTensor<R> {
    let client = input.client.clone();

    static TUNER: LocalTuner<CubeAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(|| {
        TunableSet::new(create_key::<R, N>, create_wgrad_input::<R, N>)
            .with(Tunable::new(
                "wgrad_fallback",
                conv_weight_backward_fallback::<R, N>,
            ))
            .with(Tunable::new(
                "simple_sync_cmma",
                |input, grad, shape, options| {
                    wgrad_gemm_simple_sync(input, grad, shape, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_sync_mma",
                |input, grad, shape, options| {
                    wgrad_gemm_simple_sync(input, grad, shape, options, AcceleratedTileKind::Mma)
                },
            ))
            .with(Tunable::new(
                "simple_async_cmma",
                |input, grad, shape, options| {
                    wgrad_gemm_simple_async(input, grad, shape, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_async_mma",
                |input, grad, shape, options| {
                    wgrad_gemm_simple_async(input, grad, shape, options, AcceleratedTileKind::Mma)
                },
            ))
            .with(Tunable::new(
                "simple_tma_cmma",
                |input, grad, shape, options| {
                    wgrad_gemm_simple_tma(input, grad, shape, options, AcceleratedTileKind::Cmma)
                },
            ))
            .with(Tunable::new(
                "simple_tma_mma",
                |input, grad, shape, options| {
                    wgrad_gemm_simple_tma(input, grad, shape, options, AcceleratedTileKind::Mma)
                },
            ))
    });

    TUNER.execute(
        &CubeTuneId::new(&input.client, &input.device),
        &client,
        tunables,
        (input, out_grad, weight_shape, options),
    )
}

pub fn create_wgrad_input<R: CubeRuntime, const N: usize>(
    _key: &CubeAutotuneKey,
    input: &CubeTensor<R>,
    out_grad: &CubeTensor<R>,
    weight_shape: &Shape,
    options: &ConvOptions<N>,
) -> (CubeTensor<R>, CubeTensor<R>, Shape, ConvOptions<N>) {
    (
        input.clone(),
        out_grad.clone(),
        weight_shape.clone(),
        options.clone(),
    )
}

fn create_key<R: CubeRuntime, const N: usize>(
    input: &CubeTensor<R>,
    out_grad: &CubeTensor<R>,
    weight_shape: &Shape,
    options: &ConvOptions<N>,
) -> CubeAutotuneKey {
    let dtype = input.dtype;
    let rank = input.meta.num_dims();
    let dim_c = rank - 1;

    let batch_size = input.meta.shape()[0];
    let in_channels = input.meta.shape()[dim_c];
    let out_channels = weight_shape[0];

    let kernel_size = weight_shape[1..dim_c].to_vec();
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

    let lhs_stride_align = if out_grad.meta.strides()[dim_c] == 1 {
        stride_align(out_grad.meta.strides(), out_grad.dtype.into())
    } else {
        0
    };
    let lhs_shape_align = pow2_factor(out_channels).min(lhs_stride_align);
    let rhs_stride_align = if input.meta.strides()[dim_c] == 1 {
        stride_align(input.meta.strides(), input.dtype.into())
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
        false,
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
