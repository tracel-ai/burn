use burn_backend::ops::ConvOptions;
use burn_std::Shape;
use cubek::{
    convolution::{ConvolutionArgs, Strategy, backward_weight, components::ConvSetupError},
    matmul::{
        AcceleratedTileKind, MatmulInputHandleRef, ReadingStrategy, components::MatmulElems,
        tune_key::MatmulElemType,
    },
};

use crate::{CubeRuntime, ops::numeric::empty_device_optimized_dtype, tensor::CubeTensor};

pub fn wgrad_gemm_simple_sync<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    out_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let read_strategy = match tile_kind {
        AcceleratedTileKind::Cmma => ReadingStrategy::Cyclic,
        AcceleratedTileKind::Mma => ReadingStrategy::Strided,
    };
    launch_backwards_weight::<R, N>(
        &Strategy::Simple {
            read_strategy,
            tile_kind,
        },
        input,
        out_grad,
        weight_shape,
        options,
    )
}

pub fn wgrad_gemm_simple_async<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    out_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let read_strategy = match tile_kind {
        AcceleratedTileKind::Cmma => ReadingStrategy::AsyncCyclic,
        AcceleratedTileKind::Mma => ReadingStrategy::AsyncStrided,
    };
    launch_backwards_weight::<R, N>(
        &Strategy::Simple {
            read_strategy,
            tile_kind,
        },
        input,
        out_grad,
        weight_shape,
        options,
    )
}

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components. Uses [`CmmaLargeMAlgorithm`] for the stage size
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn wgrad_gemm_simple_tma<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    out_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor<R>, ConvSetupError> {
    launch_backwards_weight::<R, N>(
        &Strategy::Simple {
            read_strategy: ReadingStrategy::Tma,
            tile_kind,
        },
        input,
        out_grad,
        weight_shape,
        options,
    )
}

/// Perform a convolution backwards weight pass using the implicit GEMM (im2col) algorithm, using
/// cubecl tiling matmul components.
///
/// * `input` - The input feature map
/// * `out_grad` - The output gradients
/// * `weight_shape` - The shape of the weights/weight gradients
/// * `options` - The options to use for the convolution
pub fn launch_backwards_weight<R: CubeRuntime, const N: usize>(
    strategy: &Strategy,
    input: CubeTensor<R>,
    out_grad: CubeTensor<R>,
    weight_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let out_dtype = input.dtype;

    let weight_grad = empty_device_optimized_dtype(
        input.client.clone(),
        input.device.clone(),
        weight_shape,
        out_dtype,
    );

    let client = input.client.clone();
    let dtypes = MatmulElems::from_globals(
        MatmulElemType {
            dtype: input.dtype.into(),
            quantized: false,
        },
        MatmulElemType {
            dtype: out_grad.dtype.into(),
            quantized: false,
        },
        MatmulElemType {
            dtype: out_dtype.into(),
            quantized: false,
        },
    );
    let input = MatmulInputHandleRef::new(input.as_handle_ref(), input.dtype.into());
    let out_grad = MatmulInputHandleRef::new(out_grad.as_handle_ref(), out_grad.dtype.into());

    backward_weight::launch_ref::<R, N>(
        strategy,
        &client,
        &input,
        &out_grad,
        &weight_grad.as_handle_ref(),
        ConvolutionArgs {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
        },
        dtypes,
    )?;

    Ok(weight_grad)
}
