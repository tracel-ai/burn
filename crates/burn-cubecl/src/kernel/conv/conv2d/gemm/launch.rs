use std::any::TypeId;

use burn_tensor::{
    Shape,
    ops::{ConvOptions, conv::calculate_conv_output_size},
};
use cubecl::{
    Feature,
    ir::{Elem, FloatKind},
    linalg::{
        convolution::{
            ConvLaunchError,
            algorithm::{Algorithm, ImplicitCmmaConv},
            base::ConvolutionProblem,
            launch_conv2d_nhwc,
            selection::{Balanced, ConvSelector, Large},
        },
        matmul::{self, components::MatmulPrecision},
    },
    tensor_line_size, tf32,
};

use crate::{
    CubeElement, CubeRuntime, FloatElement,
    kernel::{conv::permute_nchw_to_nhwc, into_contiguous},
    ops::{numeric::empty_device, permute, reshape},
    tensor::CubeTensor,
};

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components. Uses [`CmmaLargeMAlgorithm`] for the stage size
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv2d_gemm_cmma_large_m<R: CubeRuntime, F: FloatElement>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<2>,
) -> Result<CubeTensor<R>, ConvLaunchError> {
    conv2d_gemm_cmma_strategy::<R, F, ImplicitCmmaConv, Large>(input, weight, bias, options)
}

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components. Uses [`CmmaBalancedAlgorithm`] for the stage size
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv2d_gemm_cmma_balanced<R: CubeRuntime, F: FloatElement>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<2>,
) -> Result<CubeTensor<R>, ConvLaunchError> {
    conv2d_gemm_cmma_strategy::<R, F, ImplicitCmmaConv, Balanced>(input, weight, bias, options)
}

fn conv2d_gemm_cmma_strategy<
    R: CubeRuntime,
    F: FloatElement,
    Alg: Algorithm,
    S: ConvSelector<Alg>,
>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<2>,
) -> Result<CubeTensor<R>, ConvLaunchError> {
    if TypeId::of::<F>() == TypeId::of::<f32>() && has_tf32(&input) {
        conv2d_gemm_with_algo::<R, (f32, tf32, f32, f32), Alg, S>(input, weight, bias, options)
    } else {
        conv2d_gemm_with_algo::<R, F, Alg, S>(input, weight, bias, options)
    }
}

/// Perform a 2D convolution using the implicit GEMM (im2col) algorithm, using cubecl tiling matmul
/// components, using the specified algorithm.
///
/// * `input` - The input feature map
/// * `weight` - The weights (filter) applied to each kernel
/// * `bias` - The bias added to each channel
/// * `options` - The options to use for the convolution
pub fn conv2d_gemm_with_algo<
    R: CubeRuntime,
    SP: MatmulPrecision,
    Alg: Algorithm,
    S: ConvSelector<Alg>,
>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: Option<CubeTensor<R>>,
    options: ConvOptions<2>,
) -> Result<CubeTensor<R>, ConvLaunchError>
where
    SP::EI: CubeElement,
    SP::EO: CubeElement,
{
    if options.groups != 1 {
        return Err(ConvLaunchError::Groups(options.groups));
    }

    let [batch_size, in_channels, height, width] = input.shape.dims();
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

    let input = permute_nchw_to_nhwc::<R, SP::EI>(input);
    let weight = into_contiguous(permute(weight, &[2, 3, 1, 0]));

    // Implicit GEMM matrix size
    let gemm_m = batch_size * out_h * out_w;
    let gemm_n = out_channels;
    let gemm_k = kernel_h * kernel_w * in_channels;

    let weight = reshape(weight, Shape::new([gemm_k, gemm_n]));

    let out_shape = Shape::new([gemm_m, gemm_n]);
    let out = empty_device::<R, SP::EO>(input.client.clone(), input.device.clone(), out_shape);

    // Target 128 bit accesses
    let available_vectorizations = R::supported_line_sizes()
        .iter()
        .copied()
        .filter(|it| *it as usize * size_of::<SP::EI>() <= 16)
        .collect::<Vec<_>>();
    let lhs_line_size = tensor_line_size(
        &available_vectorizations,
        &input.shape.dims,
        &input.strides,
        3,
    );
    let rhs_line_size = tensor_line_size(
        &available_vectorizations,
        &weight.shape.dims,
        &weight.strides,
        1,
    );
    let out_line_size =
        tensor_line_size(&available_vectorizations, &out.shape.dims, &out.strides, 1);

    let problem = ConvolutionProblem {
        m: gemm_m,
        n: gemm_n,
        k: gemm_k,
        lhs_layout: matmul::components::MatrixLayout::RowMajor,
        rhs_layout: matmul::components::MatrixLayout::RowMajor,
        lhs_line_size,
        rhs_line_size,
        out_line_size,
        out_shape_y: out_h,
        out_shape_x: out_w,
        has_bias: bias.is_some(),

        kernel_size: (kernel_h as u32, kernel_w as u32),
        stride: (options.stride[0] as u32, options.stride[1] as u32),
        padding: (options.padding[0] as i32, options.padding[1] as i32),
        dilation: (options.dilation[0] as u32, options.dilation[1] as u32),
    };

    let bias = bias.unwrap_or_else(|| {
        empty_device::<R, SP::EI>(input.client.clone(), input.device.clone(), Shape::new([1]))
    });

    launch_conv2d_nhwc::<R, SP, Alg, S>(
        &input.client,
        input.as_tensor_arg::<SP::EI>(lhs_line_size),
        weight.as_tensor_arg::<SP::EI>(rhs_line_size),
        bias.as_tensor_arg::<SP::EI>(out_line_size),
        out.as_tensor_arg::<SP::EO>(out_line_size),
        problem,
    )?;

    // Reset to NCHW
    let out = reshape(out, Shape::new([batch_size, out_h, out_w, out_channels]));
    Ok(permute(out, &[0, 3, 1, 2]))
}

pub(crate) fn has_tf32<R: CubeRuntime>(c: &CubeTensor<R>) -> bool {
    c.client
        .properties()
        .feature_enabled(Feature::Type(Elem::Float(FloatKind::TF32)))
}
