use crate::{CubeRuntime, ops::numeric::empty_device_optimized_dtype, tensor::CubeTensor};
use burn_backend::ops::{ConvOptions, conv::calculate_conv_output_sizes};
use cubek::{
    convolution::{ConvolutionArgs, Strategy, components::ConvSetupError, forward::launch_ref},
    matmul::{
        definition::{MatmulElemType, MatmulElems},
        launch::{AcceleratedTileKind, MatmulInputHandleRef, ReadingStrategy},
    },
};

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components. Uses [`CmmaLargeMAlgorithm`] for the stage size
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_gemm_simple_sync<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let read_strategy = match tile_kind {
        AcceleratedTileKind::Cmma => ReadingStrategy::Cyclic,
        AcceleratedTileKind::Mma => ReadingStrategy::Strided,
    };
    launch_convolution::<R, N>(
        &Strategy::Simple {
            read_strategy,
            tile_kind,
        },
        input,
        weight,
        bias,
        options,
    )
}

pub fn conv_gemm_simple_async<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let read_strategy = match tile_kind {
        AcceleratedTileKind::Cmma => ReadingStrategy::AsyncCyclic,
        AcceleratedTileKind::Mma => ReadingStrategy::AsyncStrided,
    };
    launch_convolution::<R, N>(
        &Strategy::Simple {
            read_strategy,
            tile_kind,
        },
        input,
        weight,
        bias,
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
#[allow(unused)]
pub fn conv_gemm_simple_tma<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor<R>, ConvSetupError> {
    launch_convolution::<R, N>(
        &Strategy::Simple {
            read_strategy: ReadingStrategy::Tma,
            tile_kind,
        },
        input,
        weight,
        bias,
        options,
    )
}

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components, using the specified algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn launch_convolution<R: CubeRuntime, const N: usize>(
    strategy: &Strategy,
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

    let out_dtype = input.dtype;
    let rank = input.shape.num_dims();
    let batch_size = input.shape[0];
    let dim_c = rank - 1;
    let shape = &input.shape[1..dim_c];

    let out_channels = weight.shape[0];
    let weight_shape = &weight.shape[1..dim_c];

    let mut out_shape = calculate_conv_output_sizes(
        weight_shape,
        &options.stride,
        &options.padding,
        &options.dilation,
        shape,
    );

    out_shape.insert(0, batch_size);
    out_shape.push(out_channels);

    let out = empty_device_optimized_dtype(
        input.client.clone(),
        input.device.clone(),
        out_shape.into(),
        out_dtype,
    );

    let bias = bias
        .as_ref()
        .map(|bias| MatmulInputHandleRef::Normal(bias.as_handle_ref(), bias.dtype.into()));

    let client = input.client.clone();
    let dtypes = MatmulElems::from_globals(
        MatmulElemType {
            dtype: input.dtype.into(),
            quantized: false,
        },
        MatmulElemType {
            dtype: weight.dtype.into(),
            quantized: false,
        },
        MatmulElemType {
            dtype: out_dtype.into(),
            quantized: false,
        },
    );
    let input = MatmulInputHandleRef::new(input.as_handle_ref(), input.dtype.into());
    let weight = MatmulInputHandleRef::new(weight.as_handle_ref(), weight.dtype.into());

    launch_ref::<R, N>(
        strategy,
        &client,
        &input,
        &weight,
        &bias,
        &out.as_handle_ref(),
        ConvolutionArgs {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
        },
        dtypes,
    )?;

    Ok(out)
}
