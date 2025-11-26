use burn_tensor::{
    DType,
    ops::{ConvOptions, conv::calculate_conv_output_sizes},
};
use cubecl::{
    convolution::{
        ConvolutionArgs,
        components::{
            AcceleratedConv, ConvSetupError,
            global::args::{ConcreteInputsFactory, ConcreteOutputFactory},
        },
        kernels::layered::algorithm::{
            Algorithm, multi_stage_tma::MultiStageTmaConvAlgorithm, simple::SimpleConvAlgorithm,
            simple_tma::SimpleTmaConvAlgorithm,
        },
        launch_conv,
    },
    matmul::{
        MatmulInputHandleRef,
        components::{InputArg, MatmulElems, OutputArg},
    },
};

use crate::{CubeRuntime, ops::numeric::empty_device_optimized_dtype, tensor::CubeTensor};

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components. Uses [`CmmaLargeMAlgorithm`] for the stage size
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_gemm_cyclic<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let out_dtype = input.dtype;
    conv_gemm_with_algo::<R, SimpleConvAlgorithm<AcceleratedConv>, N>(
        input, weight, bias, options, out_dtype,
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
pub fn conv_gemm_tma<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let out_dtype = input.dtype;
    conv_gemm_with_algo::<R, SimpleTmaConvAlgorithm<AcceleratedConv>, N>(
        input, weight, bias, options, out_dtype,
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
pub fn conv_gemm_tma_multi_stage<R: CubeRuntime, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    let out_dtype = input.dtype;
    conv_gemm_with_algo::<R, MultiStageTmaConvAlgorithm<AcceleratedConv>, N>(
        input, weight, bias, options, out_dtype,
    )
}

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components, using the specified algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_gemm_with_algo<R: CubeRuntime, Alg: Algorithm, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
    out_dtype: DType,
) -> Result<CubeTensor<R>, ConvSetupError>
where
    InputArg<Alg::Args>: ConcreteInputsFactory,
    OutputArg<Alg::Args>: ConcreteOutputFactory,
{
    if options.groups != 1 {
        return Err(ConvSetupError::Groups(options.groups));
    }

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

    let bias = bias.as_ref().map(|bias| bias.as_handle_ref());

    let client = input.client.clone();
    let dtypes =
        MatmulElems::from_globals(input.dtype.into(), weight.dtype.into(), out_dtype.into());
    let input = MatmulInputHandleRef::new(input.as_handle_ref(), input.dtype.into());
    let weight = MatmulInputHandleRef::new(weight.as_handle_ref(), weight.dtype.into());

    launch_conv::<R, Alg, N>(
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
