use core::{marker::PhantomData, mem::transmute};

use burn_backend::{
    DType,
    Element,
    ops::{ConvOptions, conv::calculate_conv_output_size},
};
use bytemuck::Zeroable;
use macerator::{Simd, VMulAdd, Vector, vload_unaligned, vstore_unaligned};
use ndarray::{
    ArcArray1, Array4, ArrayView3, ArrayView4, ArrayViewMut2, ArrayViewMut3, Dim, Ix1, Ix4, s,
};
use seq_macro::seq;

use crate::{FloatNdArrayElement, SharedArray, UnsafeSharedRef, iter_range_par, run_par};

type Args<E> = (SharedArray<E>, SharedArray<E>, Option<SharedArray<E>>);

#[allow(clippy::result_large_err)]
pub fn try_conv2d_simd<E: FloatNdArrayElement>(
    x: SharedArray<E>,
    weight: SharedArray<E>,
    bias: Option<SharedArray<E>>,
    options: ConvOptions<2>,
) -> Result<SharedArray<E>, Args<E>> {
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

fn cast<T, E>(tensor: SharedArray<T>) -> SharedArray<E> {
    unsafe { transmute::<SharedArray<T>, SharedArray<E>>(tensor) }
}

/// Out-channel last SIMD accelerated direct convolution. Loop order and register blocking based on
/// E. Georganas, S. Avancha, K. Banerjee, D. Kalamkar, G. Henry, H. Pabst, A. Heinecke (2018).
/// Anatomy Of High-Performance Deep Learning Convolutions On SIMD Architectures.
/// SC '18, Article 6, pp. 1-12. arXiv:1808.05567. <https://arxiv.org/abs/1808.05567>.
#[allow(clippy::result_large_err)]
fn conv2d<E: VMulAdd + Element, T: Element>(
    x: SharedArray<T>,
    weight: SharedArray<T>,
    bias: Option<SharedArray<T>>,
    options: ConvOptions<2>,
    _ty: PhantomData<E>,
) -> Result<SharedArray<T>, Args<T>> {
    let [out_channels, _, k_height, k_width] = weight.shape().try_into().unwrap();
    let channels_per_group = out_channels / options.groups;

    #[macerator::with_simd]
    fn precheck<S: Simd, E: VMulAdd>(_ty: PhantomData<E>) -> (usize, bool) {
        (E::lanes::<S>(), E::is_accelerated::<S>())
    }

    let (lanes, accelerated) = precheck::<E>(PhantomData);

    if !accelerated || !channels_per_group.is_multiple_of(lanes) {
        return Err((x, weight, bias));
    }

    let x = cast::<_, E>(x);
    let weight = cast::<_, E>(weight);
    let bias = bias.map(|bias| cast::<_, E>(bias));

    let [batch_size, _in_channels, in_height, in_width] = x.shape().try_into().unwrap();
    let [dilate_h, dilate_w] = options.dilation;
    let [stride_h, stride_w] = options.stride;
    let [pad_h, pad_w] = options.padding;
    let padded = options.padding != [0, 0];
    let strided = options.stride != [1, 1] || options.dilation != [1, 1];
    let grouped = options.groups != 1;

    let out_height = calculate_conv_output_size(k_height, stride_h, pad_h, dilate_h, in_height);
    let out_width = calculate_conv_output_size(k_width, stride_w, pad_w, dilate_w, in_width);

    let x = x.into_dimensionality::<Ix4>().unwrap();
    let weights = weight.into_dimensionality::<Ix4>().unwrap();
    let weights = weights.permuted_axes([1, 2, 3, 0]);
    let weights = weights.as_standard_layout();
    let bias = bias.map(|bias| bias.into_dimensionality::<Ix1>().unwrap());
    // floor division means `(oc_blocks - 1) * lanes` can never be greater than `out_channels - lanes`.
    let oc_blocks = out_channels / lanes;

    let mut out = unsafe {
        Array4::<E>::uninit(Dim([batch_size, out_height, out_width, out_channels])).assume_init()
    };
    let unsafe_shared_out = UnsafeSharedRef::new(&mut out);

    run_par!(|| {
        // SAFETY: Slices are guaranteed to be non-overlapping, so having an unsafe shared reference
        // is safe. `oc_blocks * lanes` must be `<= out_channels` to satisfy safety of inner function.
        iter_range_par!(0, batch_size * oc_blocks).for_each(|k| unsafe {
            let b = k / oc_blocks;
            let ob = k % oc_blocks;
            let x = x.slice(s![b, .., .., ..]);
            let out = unsafe_shared_out.get();
            let mut out = out.slice_mut(s![b, .., .., ..]);
            let w = weights.view();

            match (padded, strided, grouped) {
                (true, true, true) => {
                    conv2d_launch::<E, true, true, true>(x, w, &bias, &mut out, &options, ob)
                }
                (true, false, true) => {
                    conv2d_launch::<E, true, false, true>(x, w, &bias, &mut out, &options, ob)
                }
                (false, true, true) => {
                    conv2d_launch::<E, false, true, true>(x, w, &bias, &mut out, &options, ob)
                }
                (false, false, true) => {
                    conv2d_launch::<E, false, false, true>(x, w, &bias, &mut out, &options, ob)
                }
                (true, true, false) => {
                    conv2d_launch::<E, true, true, false>(x, w, &bias, &mut out, &options, ob)
                }
                (true, false, false) => {
                    conv2d_launch::<E, true, false, false>(x, w, &bias, &mut out, &options, ob)
                }
                (false, true, false) => {
                    conv2d_launch::<E, false, true, false>(x, w, &bias, &mut out, &options, ob)
                }
                (false, false, false) => {
                    conv2d_launch::<E, false, false, false>(x, w, &bias, &mut out, &options, ob)
                }
            }
        });
    });

    let output = out.permuted_axes([0, 3, 1, 2]);
    Ok(cast(output.into_dyn().into_shared()))
}

/// Size of register blocks, we need to hardcode this because Rust and the `seq` macro don't support
/// using associated constants as constant parameters. 8 works for all semi-modern CPUs but might
/// not be perfectly optimized for AVX-512 capable CPUs (which probably should use 16).
/// This should always be conservative, since oversizing it will cause register spills and that's
/// **much** worse than the performance lost with lower values.
const REGISTER_BLOCK: usize = 8;
inner_with_register_blocking_size!(8);

/// Run a loop of conv2d.
/// # SAFETY
/// See `conv2d_inner_nopad`, `conv2d_inner_nopad_nostride`, `conv2d_remainder`.
/// Required preconditions: `ob * simd_lanes` must be `<= out_channels - simd_lanes`, `weights` and
/// `out` must have unit stride for the out channels.
#[inline(always)]
#[macerator::with_simd]
unsafe fn conv2d_launch<
    'a,
    S: Simd,
    E: VMulAdd,
    const PAD: bool,
    const STRIDE: bool,
    const GROUPS: bool,
>(
    x: ArrayView3<'a, E>,
    weights: ArrayView4<'a, E>,
    bias: &'a Option<ArcArray1<E>>,
    out: &'a mut ArrayViewMut3<'a, E>,
    options: &'a ConvOptions<2>,
    ob: usize,
) where
    'a: 'a,
{
    let (in_channels, k_height, k_width, out_channels) = weights.dim();
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
    let ow_b = REGISTER_BLOCK;

    let ow_start = pad_w;
    let ow_width = out_width.saturating_sub(2 * pad_w);
    let oh_start = pad_h;
    let oh_end = out_height.saturating_sub(pad_h);

    let ow_blocks = ow_width / ow_b;
    let oc = ob * oc_b;
    let group = oc / channels_per_group;
    let mut ic_off = group * in_channels;
    if !GROUPS {
        ic_off = 0;
    }

    unsafe {
        let bias = if let Some(bias) = &bias {
            vload_unaligned::<S, _>(&bias[oc])
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
                        &x, &weights, &mut out, bias, oh, ow, oc, ic_off, stride_h, stride_w,
                        dilate_h, dilate_w, k_height, k_width, pad_h, pad_w,
                    );
                } else {
                    conv2d_inner_nopad_nostride(
                        &x, &weights, &mut out, bias, oh, ow, oc, ic_off, k_height, k_width, pad_h,
                        pad_w,
                    );
                }
            }
        }
        conv2d_remainder(
            x,
            weights,
            out,
            bias,
            oc,
            ic_off,
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

/// Execute the non-unrolled and/or padded portion of the convolution. This has more checks and is
/// much slower, so we want to minimize the amount of pixels that need to be processed by this
///
/// SAFETY: `oc` must be an index that's at most `out_channels - simd_lanes`, so the full vector
/// is in bounds. Weights and `out` must be channels last (with `stride == 1`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn conv2d_remainder<S: Simd, E: VMulAdd>(
    x: ArrayView3<E>,
    weights: ArrayView4<E>,
    out: &mut ArrayViewMut3<E>,
    bias: Vector<S, E>,
    oc: usize,
    ic_off: usize,
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
    let in_channels = weights.shape()[0];
    let (_, in_height, in_width) = x.dim();
    let (out_height, out_width, _) = out.dim();
    let oh_start = pad_h;
    let oh_end = out_height.saturating_sub(pad_h);
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

                        // Load a full vector from the weights. This is guaranteed to be in bounds
                        // as long as `oc <= out_channels - simd_lanes` and out channels are last.
                        // We need to ensure the weights are reshaped appropriately.
                        let f0 = unsafe { vload_unaligned(&weights[[ic, kh, kw, oc]]) };

                        // The loop bounds ensure `ic`, `ih` and `iw` are always in bounds, but the
                        // compiler can't prove this. We can't use `as_slice` with fixed bounds
                        // because we want to support arbitrary input layouts. So an unchecked load
                        // is used.
                        let i0 = unsafe { x.uget([ic, ih, iw]) }.splat::<S>();
                        acc = i0.mul_add(f0, acc);
                    }
                }
            }

            // Store a full vector from the output. This is guaranteed to be in bounds
            // as long as `oc <= out_channels - simd_lanes` and oc stride is 1. We create `out` with
            // channels last, so this always holds.
            unsafe { vstore_unaligned(&mut out[[oh, ow, oc]], acc) };
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

                        // Load a full vector from the weights. This is guaranteed to be in bounds
                        // as long as `oc <= out_channels - simd_lanes` and out channels are last.
                        // We need to ensure the weights are reshaped appropriately.
                        let f0 = unsafe { vload_unaligned(&weights[[ic, kh, kw, oc]]) };

                        // The loop bounds ensure `ic`, `ih` and `iw` are always in bounds, but the
                        // compiler can't prove this. We can't use `as_slice` with fixed bounds
                        // because we want to support arbitrary input layouts. So an unchecked load
                        // is used.
                        let i0 = unsafe { x.uget([ic_off + ic, ih, iw]) }.splat::<S>();
                        acc = i0.mul_add(f0, acc);
                    }
                }
            }

            // Store a full vector from the output. This is guaranteed to be in bounds
            // as long as `oc <= out_channels - simd_lanes` and oc stride is 1. We create `out` with
            // channels last, so this always holds.
            unsafe { vstore_unaligned(&mut out[[oh, ow, oc]], acc) };
        }
    }
}

macro_rules! inner_with_register_blocking_size {
    ($rb: literal) => {
        /// Execute the unrolled and unpadded portion of the convolution. Any pixel that is more than
        /// `pad_h` away from the horizontal border, and `pad_w` away from the vertical border is
        /// guaranteed to always be in bounds (because of the way out size is calculated).
        ///
        /// SAFETY: `oc` must be an index that's at most `out_channels - simd_lanes`, so the full vector
        /// is in bounds. Weights and `out` must be channels last (with `stride == 1`).
        #[allow(clippy::erasing_op, clippy::identity_op, clippy::too_many_arguments)]
        #[inline(always)]
        unsafe fn conv2d_inner_nopad<S: Simd, E: VMulAdd>(
            x: &ArrayView3<E>,
            weights: &ArrayView4<E>,
            out: &mut ArrayViewMut2<E>,
            bias: Vector<S, E>,
            oh: usize,
            ow: usize,
            oc: usize,
            ic_off: usize,
            stride_h: usize,
            stride_w: usize,
            dilate_h: usize,
            dilate_w: usize,
            k_height: usize,
            k_width: usize,
            pad_h: usize,
            pad_w: usize,
        ) {
            let in_channels = weights.shape()[0];

            seq!(N in 0..$rb {
                let mut acc~N = bias;
            });

            for ic in 0..in_channels {
                for kh in 0..k_height {
                    let ih = oh * stride_h + kh * dilate_h - pad_h;

                    for kw in 0..k_width {
                        // Load a full vector from the weights. This is guaranteed to be in bounds
                        // as long as `oc <= out_channels - simd_lanes` and out channels are last.
                        // We need to ensure the weights are reshaped appropriately.
                        let f0 = unsafe { vload_unaligned(&weights[[ic, kh, kw, oc]]) };
                        let iw = ow * stride_w + kw * dilate_w - pad_w;

                        seq!(N in 0..$rb {
                            // The loop bounds ensure `ic`, `ih` and `iw` are always in bounds, but the
                            // compiler can't prove this. We can't use `as_slice` with fixed bounds
                            // because we want to support arbitrary input layouts. So an unchecked load
                            // is used.
                            let i~N = unsafe { x.uget([ic + ic_off, ih, iw + N * stride_w]) }.splat::<S>();
                        });
                        seq!(N in 0..$rb {
                            acc~N = i~N.mul_add(f0, acc~N);
                        });
                    }
                }
            }

            seq!(N in 0..$rb {
                // Store a full vector from the output. This is guaranteed to be in bounds
                // as long as `oc <= out_channels - simd_lanes` and oc stride is 1. We create `out` with
                // channels last, so this always holds.
                unsafe { vstore_unaligned(&mut out[[ow + N, oc]], acc~N) };
            });
        }

        /// Execute the unrolled and unpadded portion of the convolution. Any pixel that is more than
        /// `pad_h` away from the horizontal border, and `pad_w` away from the vertical border is
        /// guaranteed to always be in bounds (because of the way out size is calculated).
        ///
        /// SAFETY: `oc` must be an index that's at most `out_channels - simd_lanes`, so the full vector
        /// is in bounds. Weights and `out` must be channels last (with `stride == 1`).
        #[allow(clippy::erasing_op, clippy::identity_op, clippy::too_many_arguments)]
        #[inline(always)]
        unsafe fn conv2d_inner_nopad_nostride<S: Simd, E: VMulAdd>(
            x: &ArrayView3<E>,
            weights: &ArrayView4<E>,
            out: &mut ArrayViewMut2<E>,
            bias: Vector<S, E>,
            oh: usize,
            ow: usize,
            oc: usize,
            ic_off: usize,
            k_height: usize,
            k_width: usize,
            pad_h: usize,
            pad_w: usize,
        ) {
            let in_channels = weights.shape()[0];

            seq!(N in 0..$rb {
                let mut acc~N = bias;
            });

            for ic in 0..in_channels {
                for kh in 0..k_height {
                    let ih = oh + kh - pad_h;

                    for kw in 0..k_width {
                        // Load a full vector from the weights. This is guaranteed to be in bounds
                        // as long as `oc <= out_channels - simd_lanes` and out channels are last.
                        // We need to ensure the weights are reshaped appropriately.
                        let f0 = unsafe { vload_unaligned(&weights[[ic, kh, kw, oc]]) };
                        let iw = ow + kw - pad_w;

                        seq!(N in 0..$rb {
                            // The loop bounds ensure `ic`, `ih` and `iw` are always in bounds, but the
                            // compiler can't prove this. We can't use `as_slice` with fixed bounds
                            // because we want to support arbitrary input layouts. So an unchecked load
                            // is used.
                            let i~N = unsafe { x.uget([ic + ic_off, ih, iw + N]) }.splat::<S>();
                        });
                        seq!(N in 0..$rb {
                            acc~N = i~N.mul_add(f0, acc~N);
                        });
                    }
                }
            }

            seq!(N in 0..$rb {
                // Store a full vector from the output. This is guaranteed to be in bounds
                // as long as `oc <= out_channels - simd_lanes` and oc stride is 1. We create `out` with
                // channels last, so this always holds.
                unsafe { vstore_unaligned(&mut out[[ow + N, oc]], acc~N) };
            });
        }
    };
}
pub(crate) use inner_with_register_blocking_size;
