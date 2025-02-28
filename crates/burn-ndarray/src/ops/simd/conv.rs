use core::{marker::PhantomData, mem::transmute};

use burn_common::{iter_range_par, run_par};
use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    DType, Element, TensorMetadata,
};
use bytemuck::Zeroable;
use macerator::{SimdExt, VMulAdd};
use ndarray::{
    s, ArcArray1, Array4, ArrayView3, ArrayView4, ArrayViewMut2, ArrayViewMut3, Dim, Ix1, Ix4,
};
use pulp::{Arch, Simd};
use seq_macro::seq;

use crate::{ops::simd::lanes, FloatNdArrayElement, NdArrayTensor, UnsafeSharedRef};

type Args<E> = (NdArrayTensor<E>, NdArrayTensor<E>, Option<NdArrayTensor<E>>);

#[allow(clippy::result_large_err)]
pub fn try_conv2d_simd<E: FloatNdArrayElement>(
    x: NdArrayTensor<E>,
    weight: NdArrayTensor<E>,
    bias: Option<NdArrayTensor<E>>,
    options: ConvOptions<2>,
) -> Result<NdArrayTensor<E>, Args<E>> {
    match E::dtype() {
        DType::F64 => conv2d::<f64, _>(x, weight, bias, options, PhantomData),
        DType::F32 => conv2d::<f32, _>(x, weight, bias, options, PhantomData),
        DType::I64 => conv2d::<i64, _>(x, weight, bias, options, PhantomData),
        DType::I32 => conv2d::<i32, _>(x, weight, bias, options, PhantomData),
        DType::I16 => conv2d::<i16, _>(x, weight, bias, options, PhantomData),
        DType::U64 => conv2d::<u64, _>(x, weight, bias, options, PhantomData),
        DType::U32 => conv2d::<u32, _>(x, weight, bias, options, PhantomData),
        DType::U16 => conv2d::<u16, _>(x, weight, bias, options, PhantomData),
        _ => Err((x, weight, bias)),
    }
}

fn cast<T, E>(tensor: NdArrayTensor<T>) -> NdArrayTensor<E> {
    unsafe { transmute::<NdArrayTensor<T>, NdArrayTensor<E>>(tensor) }
}

#[allow(clippy::result_large_err)]
fn conv2d<E: VMulAdd + Element, T: Element>(
    x: NdArrayTensor<T>,
    weight: NdArrayTensor<T>,
    bias: Option<NdArrayTensor<T>>,
    options: ConvOptions<2>,
    _ty: PhantomData<E>,
) -> Result<NdArrayTensor<T>, Args<T>> {
    let [out_channels, _, k_height, k_width] = weight.shape().dims();
    let channels_per_group = out_channels / options.groups;
    let lanes = lanes::<E>();

    if channels_per_group % lanes != 0 {
        return Err((x, weight, bias));
    }

    let x = cast::<_, E>(x);
    let weight = cast::<_, E>(weight);
    let bias = bias.map(|bias| cast::<_, E>(bias));

    let [batch_size, _in_channels, in_height, in_width] = x.shape().dims();
    let [dilate_h, dilate_w] = options.dilation;
    let [stride_h, stride_w] = options.stride;
    let [pad_h, pad_w] = options.padding;

    let out_height = calculate_conv_output_size(k_height, stride_h, pad_h, dilate_h, in_height);
    let out_width = calculate_conv_output_size(k_width, stride_w, pad_w, dilate_w, in_width);

    let x = x.array.into_dimensionality::<Ix4>().unwrap();
    let weights = weight.array.into_dimensionality::<Ix4>().unwrap();
    let weights = weights.permuted_axes([1, 2, 3, 0]);
    let weights = weights.as_standard_layout();
    let bias = bias.map(|bias| bias.array.into_dimensionality::<Ix1>().unwrap());
    let oc_blocks = out_channels / lanes;

    let mut out = unsafe {
        Array4::<E>::uninit(Dim([batch_size, out_height, out_width, out_channels])).assume_init()
    };
    let unsafe_shared_out = UnsafeSharedRef::new(&mut out);

    run_par!(|| {
        iter_range_par!(0, batch_size * oc_blocks).for_each(|k| unsafe {
            let b = k / oc_blocks;
            let ob = k % oc_blocks;
            let x = x.slice(s![b, .., .., ..]);
            let out = unsafe_shared_out.get();
            let mut out = out.slice_mut(s![b, .., .., ..]);
            conv2d_launch(x, weights.view(), &bias, &mut out, &options, ob);
        });
    });

    let output = out.permuted_axes([0, 3, 1, 2]);
    Ok(cast(NdArrayTensor::new(output.into_dyn().into_shared())))
}

/// Size of register blocks, we need to hardcode this because Rust and the `seq` macro don't support
/// using associated constants as constant parameters. 8 works for all semi-modern CPUs but might
/// not be perfectly optimized for AVX-512 capable CPUs (which probably should use 16).
/// This should always be conservative, since oversizing it will cause register spills and that's
/// **much** worse than the performance lost with lower values.
const REGISTER_BLOCK: usize = 8;
inner_with_register_blocking_size!(8);

#[allow(clippy::identity_op, clippy::erasing_op)]
#[inline(always)]
#[pulp::with_simd(conv2d_launch = Arch::new())]
pub fn conv2d_simd<S: Simd, E: VMulAdd + Element>(
    simd: S,
    x: ArrayView3<'_, E>,
    weights: ArrayView4<'_, E>,
    bias: &Option<ArcArray1<E>>,
    out: &mut ArrayViewMut3<'_, E>,
    options: &ConvOptions<2>,
    ob: usize,
) {
    let padded = options.padding != [0, 0];
    let strided = options.stride != [1, 1] || options.dilation != [1, 1];
    match (padded, strided) {
        (true, true) => unsafe {
            run_conv2d::<S, E, REGISTER_BLOCK, true, true>(simd, x, weights, bias, out, options, ob)
        },
        (true, false) => unsafe {
            run_conv2d::<S, E, REGISTER_BLOCK, true, false>(
                simd, x, weights, bias, out, options, ob,
            )
        },
        (false, true) => unsafe {
            run_conv2d::<S, E, REGISTER_BLOCK, false, true>(
                simd, x, weights, bias, out, options, ob,
            )
        },
        (false, false) => unsafe {
            run_conv2d::<S, E, REGISTER_BLOCK, false, false>(
                simd, x, weights, bias, out, options, ob,
            )
        },
    }
}

#[inline(always)]
unsafe fn run_conv2d<S: Simd, E: VMulAdd, const RB: usize, const PAD: bool, const STRIDE: bool>(
    simd: S,
    x: ArrayView3<E>,
    weights: ArrayView4<E>,
    bias: &Option<ArcArray1<E>>,
    out: &mut ArrayViewMut3<E>,
    options: &ConvOptions<2>,
    ob: usize,
) {
    let (_, k_height, k_width, out_channels) = weights.dim();
    let (out_height, out_width, _) = out.dim();
    let channels_per_group = out_channels / options.groups;
    let lanes = E::lanes::<S>();

    let [mut pad_h, mut pad_w] = options.padding;
    let [stride_h, stride_w] = options.stride;
    let [dilate_h, dilate_w] = options.dilation;

    // Trick compiler into inlining 0 to padding
    if !PAD {
        pad_h = 0;
        pad_w = 0;
    }

    let oc_b = channels_per_group.min(lanes);
    let ow_b = RB;

    let ow_start = pad_w;
    let ow_width = out_width - 2 * pad_w;
    let oh_start = pad_h;
    let oh_end = out_height - pad_h;

    let ow_blocks = ow_width / ow_b;

    unsafe {
        let oc = ob * oc_b;

        let bias = if let Some(bias) = &bias {
            simd.vload_unaligned(&bias[oc])
        } else {
            Zeroable::zeroed()
        };

        for oh in oh_start..oh_end {
            let mut out = out.slice_mut(s![oh, .., ..]);
            for ow_block in 0..ow_blocks {
                let ow = ow_block * ow_b + ow_start;

                #[allow(clippy::if_same_then_else)]
                if STRIDE {
                    conv2d_inner_nopad(
                        simd, &x, &weights, &mut out, bias, oh, ow, oc, stride_h, stride_w,
                        dilate_h, dilate_w, k_height, k_width, pad_h, pad_w,
                    );
                } else {
                    conv2d_inner_nopad_nostride(
                        simd, &x, &weights, &mut out, bias, oh, ow, oc, k_height, k_width, pad_h,
                        pad_w,
                    );
                }
            }
        }
        conv2d_remainder(
            simd,
            x,
            weights,
            out,
            bias,
            oc,
            ow_blocks * ow_b,
            stride_h,
            stride_w,
            dilate_h,
            dilate_w,
            pad_h,
            pad_w,
            k_height,
            k_width,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn conv2d_remainder<S: Simd, E: VMulAdd>(
    simd: S,
    x: ArrayView3<E>,
    weights: ArrayView4<E>,
    out: &mut ArrayViewMut3<E>,
    bias: E::Vector<S>,
    oc: usize,
    owb_end: usize,
    stride_h: usize,
    stride_w: usize,
    dilate_h: usize,
    dilate_w: usize,
    pad_h: usize,
    pad_w: usize,
    k_height: usize,
    k_width: usize,
) {
    let (in_channels, in_height, in_width) = x.dim();
    let (out_height, out_width, _) = out.dim();
    let oh_start = pad_h;
    let oh_end = out_height - pad_h;
    let ow_start = pad_w;

    let height1 = in_height + pad_h;
    let width1 = in_width + pad_w;

    for oh in (0..oh_start).chain(oh_end..out_height) {
        for ow in 0..out_width {
            let mut acc = bias;

            for ic in 0..in_channels {
                for kh in 0..k_height {
                    let ih = oh * stride_h + kh * dilate_h;
                    if (ih < pad_h) | (ih >= height1) {
                        continue;
                    }
                    let ih = ih - pad_h;

                    for kw in 0..k_width {
                        let iw = ow * stride_w + kw * dilate_w;
                        if (iw < pad_w) | (iw >= width1) {
                            continue;
                        }
                        let iw = iw - pad_w;

                        let f0 = simd.vload_unaligned(&weights[[ic, kh, kw, oc]]);

                        let i0 = simd.splat(*x.uget([ic, ih, iw]));
                        acc = E::vmuladd(simd, i0, f0, acc);
                    }
                }
            }

            simd.vstore_unaligned(&mut out[[oh, ow, oc]], acc);
        }
    }
    for ow in (0..ow_start).chain(owb_end..out_width) {
        for oh in 0..out_height {
            let mut acc = bias;

            for ic in 0..in_channels {
                for kh in 0..k_height {
                    let ih = oh * stride_h + kh * dilate_h;
                    if (ih < pad_h) | (ih >= height1) {
                        continue;
                    }
                    let ih = ih - pad_h;

                    for kw in 0..k_width {
                        let iw = ow * stride_w + kw * dilate_w;
                        if (iw < pad_w) | (iw >= width1) {
                            continue;
                        }
                        let iw = iw - pad_w;

                        let f0 = simd.vload_unaligned(&weights[[ic, kh, kw, oc]]);

                        let i0 = simd.splat(*x.uget([ic, ih, iw]));
                        acc = E::vmuladd(simd, i0, f0, acc);
                    }
                }
            }

            simd.vstore_unaligned(&mut out[[oh, ow, oc]], acc);
        }
    }
}

macro_rules! inner_with_register_blocking_size {
    ($rb: literal) => {
        #[allow(clippy::erasing_op, clippy::identity_op, clippy::too_many_arguments)]
        #[inline(always)]
        unsafe fn conv2d_inner_nopad<S: Simd, E: VMulAdd>(
            simd: S,
            x: &ArrayView3<E>,
            weights: &ArrayView4<E>,
            out: &mut ArrayViewMut2<E>,
            bias: E::Vector<S>,
            oh: usize,
            ow: usize,
            oc: usize,
            stride_h: usize,
            stride_w: usize,
            dilate_h: usize,
            dilate_w: usize,
            k_height: usize,
            k_width: usize,
            pad_h: usize,
            pad_w: usize,
        ) {
            let in_channels = x.shape()[0];

            seq!(N in 0..$rb {
                let mut acc~N = bias;
            });

            for ic in 0..in_channels {
                for kh in 0..k_height {
                    let ih = oh * stride_h + kh * dilate_h - pad_h;

                    for kw in 0..k_width {
                        let f0 = simd.vload_unaligned(&weights[[ic, kh, kw, oc]]);
                        let iw = ow * stride_w + kw * dilate_w - pad_w;

                        seq!(N in 0..$rb {
                            let i~N = simd.splat(*x.uget([ic, ih, iw + N * stride_w]));
                        });
                        seq!(N in 0..$rb {
                            acc~N = E::vmuladd(simd, i~N, f0, acc~N);
                        });
                    }
                }
            }

            seq!(N in 0..$rb {
                simd.vstore_unaligned(&mut out[[ow + N, oc]], acc~N);
            });
        }

        #[allow(clippy::erasing_op, clippy::identity_op, clippy::too_many_arguments)]
        #[inline(always)]
        unsafe fn conv2d_inner_nopad_nostride<S: Simd, E: VMulAdd>(
            simd: S,
            x: &ArrayView3<E>,
            weights: &ArrayView4<E>,
            out: &mut ArrayViewMut2<E>,
            bias: E::Vector<S>,
            oh: usize,
            ow: usize,
            oc: usize,
            k_height: usize,
            k_width: usize,
            pad_h: usize,
            pad_w: usize,
        ) {
            let in_channels = x.shape()[0];

            seq!(N in 0..$rb {
                let mut acc~N = bias;
            });

            for ic in 0..in_channels {
                for kh in 0..k_height {
                    let ih = oh + kh - pad_h;

                    for kw in 0..k_width {
                        let f0 = simd.vload_unaligned(&weights[[ic, kh, kw, oc]]);
                        let iw = ow + kw - pad_w;

                        seq!(N in 0..$rb {
                            let i~N = simd.splat(*x.uget([ic, ih, iw + N]));
                        });
                        seq!(N in 0..$rb {
                            acc~N = E::vmuladd(simd, i~N, f0, acc~N);
                        });
                    }
                }
            }

            seq!(N in 0..$rb {
                simd.vstore_unaligned(&mut out[[ow + N, oc]], acc~N);
            });
        }
    };
}
pub(crate) use inner_with_register_blocking_size;
