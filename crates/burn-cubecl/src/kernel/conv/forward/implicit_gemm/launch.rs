use crate::{CubeRuntime, ops::numeric::empty_device_dtype, tensor::CubeTensor};
use burn_backend::ops::{ConvOptions, conv::calculate_conv_output_sizes};
use cubek::{
    convolution::{
        AcceleratedTileKind, ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy,
        components::ConvSetupError, launch_ref,
    },
    matmul::definition::{MatmulElems, MatmulGlobalElems},
    std::InputBinding,
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
    let strategy = match tile_kind {
        AcceleratedTileKind::Cmma => Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleSyncCyclic,
            tile_kind,
        },
        AcceleratedTileKind::Mma => Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleSyncStrided,
            tile_kind,
        },
    };

    launch_convolution_forward::<R, N>(&strategy, input, weight, bias, options)
}

pub fn conv_gemm_simple_async<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let strategy = match tile_kind {
        AcceleratedTileKind::Cmma => Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleAsyncCyclic,
            tile_kind,
        },
        AcceleratedTileKind::Mma => Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleAsyncStrided,
            tile_kind,
        },
    };

    launch_convolution_forward::<R, N>(&strategy, input, weight, bias, options)
}

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components. Uses [`CmmaLargeMAlgorithm`] for the stage size
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_gemm_simple_tma<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
    tile_kind: AcceleratedTileKind,
) -> Result<CubeTensor<R>, ConvSetupError> {
    launch_convolution_forward::<R, N>(
        &Strategy::Inferred {
            algorithm: ConvAlgorithm::SimpleAsyncTma,
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
pub fn launch_convolution_forward<R: CubeRuntime, const N: usize>(
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
    let rank = input.meta.shape().num_dims();
    let batch_size = input.meta.shape()[0];
    let dim_c = rank - 1;
    let shape = &input.meta.shape()[1..dim_c];

    let out_channels = weight.meta.shape()[0];
    let weight_shape = &weight.meta.shape()[1..dim_c];

    let mut out_shape = calculate_conv_output_sizes(
        weight_shape,
        &options.stride,
        &options.padding,
        &options.dilation,
        shape,
    );

    out_shape.insert(0, batch_size);
    out_shape.push(out_channels);

    let out = empty_device_dtype(
        input.client.clone(),
        input.device.clone(),
        out_shape.into(),
        out_dtype,
    );

    let bias = bias.map(|bias| {
        let dtype = bias.dtype;
        InputBinding::Normal(bias.binding(), dtype.into())
    });

    let client = input.client.clone();
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: input.dtype.into(),
        rhs: weight.dtype.into(),
        out: out_dtype.into(),
    });
    let input_dtype = input.dtype;
    let weight_dtype = weight.dtype;
    let input = InputBinding::new(input.binding(), input_dtype.into());
    let weight = InputBinding::new(weight.binding(), weight_dtype.into());

    launch_ref::<R, N>(
        strategy,
        &client,
        ConvolutionInputs::Forward {
            input,
            weight,
            bias,
            out: out.clone().binding(),
        },
        ConvolutionArgs {
            stride: options.stride,
            padding: options.padding,
            dilation: options.dilation,
        },
        dtypes,
    )?;

    Ok(out)
}
