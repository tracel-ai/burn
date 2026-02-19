use super::{
    adaptive_avgpool::{adaptive_avg_pool2d, adaptive_avg_pool2d_backward},
    avgpool::{avg_pool2d, avg_pool2d_backward},
    conv::{conv_transpose2d, conv_transpose3d, conv2d, conv3d},
    deform_conv::{backward::deform_conv2d_backward, deform_conv2d},
    interpolate::{bicubic_interpolate, bilinear_interpolate, nearest_interpolate},
    maxpool::{max_pool2d, max_pool2d_backward, max_pool2d_with_indices},
};
#[cfg(feature = "simd")]
use crate::ops::simd::{
    avgpool::try_avg_pool2d_simd, conv::try_conv2d_simd, maxpool::try_max_pool2d_simd,
};
use crate::{
    NdArray, SharedArray, element::FloatNdArrayElement, execute_with_int_dtype,
    tensor::NdArrayTensor,
};
use crate::{
    element::{IntNdArrayElement, QuantElement},
    ops::interpolate::nearest_interpolate_backward,
};
use burn_backend::{
    ElementConversion, TensorMetadata,
    ops::{attention::attention_fallback, *},
    tensor::FloatTensor,
};

macro_rules! module_op {
    // Module op with inputs (inp), optional (opt) and arguments (args).
    // Converts NdArrayStorage to SharedArray for compatibility with existing operations.
    (inp($($x:tt),+), opt($($opt:tt),*), $element:ident, $op:expr) => {{
        #[allow(unused_parens, unreachable_patterns)]
        match ($($x),+) {
            ($(NdArrayTensor::F32($x)),+) => {
                type $element = f32;
                $op(
                    $($x.into_shared()),+
                    $(, $opt.map(|o| match o { NdArrayTensor::F32(val) => val.into_shared(), _ => panic!("Optional argument type mismatch") }))*
                )
            }
            ($(NdArrayTensor::F64($x)),+) => {
                type $element = f64;
                $op(
                    $($x.into_shared()),+
                    $(, $opt.map(|o| match o { NdArrayTensor::F64(val) => val.into_shared(), _ => panic!("Optional argument type mismatch") }))*
                )
            }
            _ => panic!("Data type mismatch"),
        }
    }};
}

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> ModuleOps<Self>
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    fn conv2d(
        x: NdArrayTensor,
        weight: NdArrayTensor,
        bias: Option<NdArrayTensor>,
        options: ConvOptions<2>,
    ) -> NdArrayTensor {
        module_op!(inp(x, weight), opt(bias), E, |x, weight, bias| {
            #[cfg(feature = "simd")]
            let (x, weight, bias) = match try_conv2d_simd(x, weight, bias, options.clone()) {
                Ok(out) => return out.into(),
                Err(args) => args,
            };
            conv2d::<E>(x, weight, bias, options).into()
        })
    }

    fn deform_conv2d(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        options: DeformConvOptions<2>,
    ) -> FloatTensor<Self> {
        module_op!(
            inp(x, offset, weight),
            opt(mask, bias),
            E,
            |x, offset, weight, mask, bias| deform_conv2d::<E>(
                x, offset, weight, mask, bias, options
            )
            .into()
        )
    }

    fn deform_conv2d_backward(
        x: FloatTensor<Self>,
        offset: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        mask: Option<FloatTensor<Self>>,
        bias: Option<FloatTensor<Self>>,
        output_grad: FloatTensor<Self>,
        options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Self> {
        module_op!(
            inp(x, offset, weight, output_grad),
            opt(mask, bias),
            E,
            |x, offset, weight, output_grad, mask, bias| {
                let (x, offset, weight, mask, bias) = deform_conv2d_backward::<E>(
                    x,
                    offset,
                    weight,
                    mask,
                    bias,
                    output_grad,
                    options,
                );
                DeformConv2dBackward::new(
                    x.into(),
                    offset.into(),
                    weight.into(),
                    mask.map(|m| m.into()),
                    bias.map(|b| b.into()),
                )
            }
        )
    }

    fn conv_transpose2d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<2>,
    ) -> FloatTensor<Self> {
        module_op!(inp(x, weight), opt(bias), E, |x, weight, bias| {
            conv_transpose2d::<E>(x, weight, bias, options).into()
        })
    }

    fn avg_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        module_op!(inp(x), opt(), E, |x| {
            #[cfg(feature = "simd")]
            let x = match if ceil_mode {
                // SIMD path doesn't support ceil_mode yet, skip it
                Err(x)
            } else {
                try_avg_pool2d_simd(x, kernel_size, stride, padding, count_include_pad)
            } {
                Ok(out) => return out.into(),
                Err(x) => x,
            };
            avg_pool2d::<E>(
                x,
                kernel_size,
                stride,
                padding,
                count_include_pad,
                ceil_mode,
            )
            .into()
        })
    }

    fn avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        module_op!(inp(x, grad), opt(), E, |x, grad| avg_pool2d_backward::<E>(
            x,
            grad,
            kernel_size,
            stride,
            padding,
            count_include_pad,
            ceil_mode
        )
        .into())
    }

    fn max_pool2d(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> FloatTensor<Self> {
        module_op!(inp(x), opt(), E, |x| {
            #[cfg(feature = "simd")]
            let x = match if ceil_mode {
                // SIMD path doesn't support ceil_mode yet, skip it
                Err(x)
            } else {
                try_max_pool2d_simd(x, kernel_size, stride, padding, dilation)
            } {
                Ok(out) => return out.into(),
                Err(x) => x,
            };
            max_pool2d::<E>(x, kernel_size, stride, padding, dilation, ceil_mode).into()
        })
    }

    fn max_pool2d_with_indices(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
    ) -> MaxPool2dWithIndices<NdArray<E, I, Q>> {
        module_op!(inp(x), opt(), E, |x| {
            let (output, indices) = max_pool2d_with_indices::<E, I>(
                x,
                kernel_size,
                stride,
                padding,
                dilation,
                ceil_mode,
            );
            MaxPool2dWithIndices::new(output.into(), indices.into())
        })
    }

    fn max_pool2d_with_indices_backward(
        x: FloatTensor<Self>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ceil_mode: bool,
        output_grad: FloatTensor<Self>,
        indices: NdArrayTensor,
    ) -> MaxPool2dBackward<NdArray<E, I, Q>> {
        execute_with_int_dtype!(indices, IntElem, |idx_s: SharedArray<IntElem>| {
            // Convert indices from runtime dtype to the expected I type
            // (pool indices are bounded by tensor dimensions, so conversion is safe)
            let indices: SharedArray<I> = idx_s.mapv(|x| x.elem()).into_shared();
            module_op!(inp(x, output_grad), opt(), E, |x, output_grad| {
                let output = max_pool2d_backward::<E, I>(
                    x,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    ceil_mode,
                    output_grad,
                    indices,
                );
                MaxPool2dBackward::new(output.into())
            })
        })
    }

    fn adaptive_avg_pool2d(x: FloatTensor<Self>, output_size: [usize; 2]) -> FloatTensor<Self> {
        module_op!(inp(x), opt(), E, |x| adaptive_avg_pool2d::<E>(
            x,
            output_size
        )
        .into())
    }

    fn adaptive_avg_pool2d_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        module_op!(inp(x, grad), opt(), E, |x, grad| {
            adaptive_avg_pool2d_backward::<E>(x, grad).into()
        })
    }

    fn interpolate(
        x: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        match options.mode {
            InterpolateMode::Nearest => {
                module_op!(inp(x), opt(), E, |x| nearest_interpolate::<E>(
                    x,
                    output_size
                )
                .into())
            }
            InterpolateMode::Bilinear => {
                let align_corners = options.align_corners;
                module_op!(inp(x), opt(), E, |x| bilinear_interpolate::<E>(
                    x,
                    output_size,
                    align_corners
                )
                .into())
            }
            InterpolateMode::Bicubic => {
                let align_corners = options.align_corners;
                module_op!(inp(x), opt(), E, |x| bicubic_interpolate::<E>(
                    x,
                    output_size,
                    align_corners
                )
                .into())
            }
        }
    }

    fn interpolate_backward(
        x: FloatTensor<Self>,
        grad: FloatTensor<Self>,
        output_size: [usize; 2],
        options: InterpolateOptions,
    ) -> FloatTensor<Self> {
        match options.mode {
            InterpolateMode::Nearest => module_op!(inp(x, grad), opt(), E, |x, grad| {
                nearest_interpolate_backward::<E>(x, grad, output_size).into()
            }),
            InterpolateMode::Bilinear => {
                panic!("bilinear interpolation backward is not supported for ndarray backend")
            }
            InterpolateMode::Bicubic => {
                panic!("bicubic interpolation backward is not supported for ndarray backend")
            }
        }
    }

    fn conv3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvOptions<3>,
    ) -> FloatTensor<Self> {
        module_op!(inp(x, weight), opt(bias), E, |x, weight, bias| conv3d::<E>(
            x, weight, bias, options
        )
        .into())
    }

    fn conv_transpose3d(
        x: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        bias: Option<FloatTensor<Self>>,
        options: ConvTransposeOptions<3>,
    ) -> FloatTensor<Self> {
        module_op!(inp(x, weight), opt(bias), E, |x, weight, bias| {
            conv_transpose3d::<E>(x, weight, bias, options).into()
        })
    }

    fn attention(
        query: FloatTensor<Self>,
        key: FloatTensor<Self>,
        value: FloatTensor<Self>,
        mask: Option<burn_backend::tensor::BoolTensor<Self>>,
        attn_bias: Option<FloatTensor<Self>>,
        options: AttentionOptions,
    ) -> FloatTensor<Self> {
        attention_fallback::<Self>(query, key, value, mask, attn_bias, options)
    }
}
