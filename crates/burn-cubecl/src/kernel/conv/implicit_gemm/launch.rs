use burn_tensor::ops::{ConvOptions, conv::calculate_conv_output_sizes};
use cubecl::{
    convolution::{
        ConvolutionArgs,
        components::{AcceleratedConv, ConvSetupError, global::args::ConcreteInputsFactory},
        kernels::layered::algorithm::{
            Algorithm, multi_stage_tma::MultiStageTmaConvAlgorithm, simple::SimpleConvAlgorithm,
            simple_tma::SimpleTmaConvAlgorithm,
        },
        launch_conv,
    },
    matmul::{
        MatmulInputHandleRef,
        components::{
            AccG, LhsG, MatmulPrecision, RhsG,
            global::args::{ConcreteOutputFactory, MatmulArgs},
        },
    },
};

use crate::{
    CubeElement, CubeRuntime, FloatElement, ops::numeric::empty_device_strided, tensor::CubeTensor,
};

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components. Uses [`CmmaLargeMAlgorithm`] for the stage size
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_gemm_cyclic<R: CubeRuntime, F: FloatElement, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    conv_gemm_with_algo::<R, F, SimpleConvAlgorithm<AcceleratedConv>, N>(
        input, weight, bias, options,
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
pub fn conv_gemm_tma<R: CubeRuntime, F: FloatElement, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    conv_gemm_with_algo::<R, F, SimpleTmaConvAlgorithm<AcceleratedConv>, N>(
        input, weight, bias, options,
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
pub fn conv_gemm_tma_multi_stage<R: CubeRuntime, F: FloatElement, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError> {
    conv_gemm_with_algo::<R, F, MultiStageTmaConvAlgorithm<AcceleratedConv>, N>(
        input, weight, bias, options,
    )
}

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components, using the specified algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv_gemm_with_algo<R: CubeRuntime, MP: MatmulPrecision, Alg: Algorithm, const N: usize>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<N>,
) -> Result<CubeTensor<R>, ConvSetupError>
where
    <Alg::Args as MatmulArgs>::Input<LhsG<MP>, RhsG<MP>, AccG<MP>>: ConcreteInputsFactory,
    <Alg::Args as MatmulArgs>::Output<AccG<MP>>: ConcreteOutputFactory,
    AccG<MP>: CubeElement,
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

    let out = empty_device_strided::<R, AccG<MP>>(
        input.client.clone(),
        input.device.clone(),
        out_shape.into(),
    );

    let bias = bias.as_ref().map(|bias| bias.as_handle_ref());

    let client = input.client.clone();
    let input = MatmulInputHandleRef::new(input.as_handle_ref());
    let weight = MatmulInputHandleRef::new(weight.as_handle_ref());

    launch_conv::<R, MP, Alg, N>(
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
    )?;

    Ok(out)
}
