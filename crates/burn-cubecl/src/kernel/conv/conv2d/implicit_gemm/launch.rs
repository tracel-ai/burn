use burn_tensor::{
    Shape,
    ops::{ConvOptions, conv::calculate_conv_output_size},
};
use cubecl::linalg::{
    convolution::{
        ConvLaunchError, ConvolutionArgs,
        algorithm::{Algorithm, simple::SimpleConvAlgorithm, simple_tma::SimpleTmaConvAlgorithm},
        args::ConvInputsLaunch,
        launch_conv2d_nhwc,
    },
    matmul::components::{
        MatmulPrecision,
        global::args::{ConcreteOutputFactory, MatmulArgs},
        tile::accelerated::Accelerated,
    },
};

use crate::{
    CubeElement, CubeRuntime, FloatElement,
    ops::{numeric::empty_device_strided, permute, permute_nchw_to_nhwc, permute_nhwc_to_nchw},
    tensor::CubeTensor,
};

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components. Uses [`CmmaLargeMAlgorithm`] for the stage size
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv2d_gemm_cyclic<R: CubeRuntime, F: FloatElement>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<2>,
) -> Result<CubeTensor<R>, ConvLaunchError> {
    conv2d_gemm_with_algo::<R, F, SimpleConvAlgorithm<Accelerated>>(input, weight, bias, options)
}

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components. Uses [`CmmaLargeMAlgorithm`] for the stage size
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv2d_gemm_tma<R: CubeRuntime, F: FloatElement>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<2>,
) -> Result<CubeTensor<R>, ConvLaunchError> {
    conv2d_gemm_with_algo::<R, F, SimpleTmaConvAlgorithm<Accelerated>>(input, weight, bias, options)
}

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components, using the specified algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv2d_gemm_with_algo<R: CubeRuntime, SP: MatmulPrecision, Alg: Algorithm>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<2>,
) -> Result<CubeTensor<R>, ConvLaunchError>
where
    SP::EI: CubeElement,
    SP::EO: CubeElement,
    <Alg::Args as MatmulArgs>::Input<SP::EI>: ConvInputsLaunch,
    <Alg::Args as MatmulArgs>::Output<SP::EO>: ConcreteOutputFactory,
{
    if options.groups != 1 {
        return Err(ConvLaunchError::Groups(options.groups));
    }

    let [batch_size, _, height, width] = input.shape.dims();
    let [out_channels, _, kernel_h, kernel_w] = weight.shape.dims();

    let out_h = calculate_conv_output_size(
        kernel_h,
        options.stride[0],
        options.padding[0],
        options.dilation[0],
        height,
    );
    let out_w = calculate_conv_output_size(
        kernel_w,
        options.stride[1],
        options.padding[1],
        options.dilation[1],
        width,
    );

    let input = permute_nchw_to_nhwc(input);
    let weight = permute(weight, &[0, 2, 3, 1]);

    let out_shape = Shape::new([batch_size, out_h, out_w, out_channels]);
    let out =
        empty_device_strided::<R, SP::EO>(input.client.clone(), input.device.clone(), out_shape);

    let bias = bias.as_ref().map(|bias| bias.as_handle_ref());

    launch_conv2d_nhwc::<R, SP, Alg>(
        &input.client,
        &input.as_handle_ref(),
        &weight.as_handle_ref(),
        &bias,
        &out.as_handle_ref(),
        ConvolutionArgs {
            stride: (options.stride[0], options.stride[1]),
            padding: (options.padding[0], options.padding[1]),
            dilation: (options.dilation[0], options.dilation[1]),
        },
    )?;

    Ok(permute_nhwc_to_nchw(out))
}
