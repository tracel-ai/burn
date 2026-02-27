use burn_backend::ops::ConvOptions;
use burn_std::Shape;
use cubek::{
    convolution::{
        AcceleratedTileKind, ConvolutionArgs, ReadingStrategy, Strategy, backward_data,
        components::ConvSetupError,
    },
    matmul::{
        definition::{MatmulElems, MatmulGlobalElems},
        launch::MatmulInputBinding,
    },
};

use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};

pub fn dgrad_gemm_simple_sync<R: CubeRuntime, const N: usize>(
    out_grad: CubeTensor<R>,
    weights: CubeTensor<R>,
    input_shape: Shape,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let read_strategy = match tile_kind {
        AcceleratedTileKind::Cmma => ReadingStrategy::Cyclic,
        AcceleratedTileKind::Mma => ReadingStrategy::Strided,
    };
    launch_backwards_data::<R, N>(
        &Strategy::Simple {
            read_strategy,
            tile_kind,
        },
        out_grad,
        weights,
        input_shape,
        options,
    )
}

pub fn dgrad_gemm_simple_async<R: CubeRuntime, const N: usize>(
    out_grad: CubeTensor<R>,
    weights: CubeTensor<R>,
    input_shape: Shape,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let read_strategy = match tile_kind {
        AcceleratedTileKind::Cmma => ReadingStrategy::AsyncCyclic,
        AcceleratedTileKind::Mma => ReadingStrategy::AsyncStrided,
    };
    launch_backwards_data::<R, N>(
        &Strategy::Simple {
            read_strategy,
            tile_kind,
        },
        out_grad,
        weights,
        input_shape,
        options,
    )
}

pub fn dgrad_gemm_simple_tma<R: CubeRuntime, const N: usize>(
    out_grad: CubeTensor<R>,
    weights: CubeTensor<R>,
    input_shape: Shape,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor<R>, ConvSetupError> {
    launch_backwards_data::<R, N>(
        &Strategy::Simple {
            read_strategy: ReadingStrategy::Tma,
            tile_kind,
        },
        out_grad,
        weights,
        input_shape,
        options,
    )
}

/// Perform a convolution backwards data pass using the implicit GEMM (im2col) algorithm, using
/// cubecl tiling matmul components.
///
/// * `input` - The input feature map
/// * `out_grad` - The output gradients
/// * `weight_shape` - The shape of the weights/weight gradients
/// * `options` - The options to use for the convolution
pub fn launch_backwards_data<R: CubeRuntime, const N: usize>(
    strategy: &Strategy,
    out_grad: CubeTensor<R>,
    weights: CubeTensor<R>,
    input_shape: Shape,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    if options.groups != 1 || options.stride.iter().any(|&s| s != 1) {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let out_dtype = out_grad.dtype;

    let in_grad = empty_device_dtype(
        out_grad.client.clone(),
        out_grad.device.clone(),
        input_shape,
        out_dtype,
    );

    let client = out_grad.client.clone();
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: out_grad.dtype.into(),
        rhs: weights.dtype.into(),
        out: out_dtype.into(),
    });
    let out_grad_dtype = out_grad.dtype;
    let weights_dtype = weights.dtype;
    let out_grad = MatmulInputBinding::new(out_grad.binding(), out_grad_dtype.into());
    let weights = MatmulInputBinding::new(weights.binding(), weights_dtype.into());

    backward_data::launch_ref::<R, N>(
        strategy,
        &client,
        out_grad,
        weights,
        in_grad.clone().binding(),
        ConvolutionArgs {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
        },
        dtypes,
    )?;

    Ok(in_grad)
}
