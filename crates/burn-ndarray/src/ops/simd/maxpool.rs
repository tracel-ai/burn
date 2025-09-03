use core::{marker::PhantomData, mem::transmute};

use crate::{SharedArray, sharing::UnsafeSharedRef};

use burn_common::{iter_range_par, run_par};
use burn_tensor::{DType, Element, quantization::QuantValue};
use macerator::{Simd, VOrd};
use ndarray::{Array4, s};
use nhwc::max_pool2d_nhwc;

use super::{MinMax, should_use_simd};

#[macerator::with_simd]
fn is_accelerated_impl<S: Simd, T: VOrd>(_x: PhantomData<T>) -> bool {
    <T as VOrd>::is_min_max_accelerated::<S>()
}

fn is_accelerated<T: VOrd>() -> bool {
    is_accelerated_impl::<T>(PhantomData)
}

macro_rules! launch_kernel {
    ($ty: ty, $func: ident, $x: expr, $($arg: expr),*) => {
        match <$ty as Element>::dtype() {
            DType::F64 if is_accelerated::<f64>() => Ok(cast($func::<f64>(cast($x), $($arg),*))),
            DType::F32 if is_accelerated::<f32>() => Ok(cast($func::<f32>(cast($x), $($arg),*))),
            DType::I64 if is_accelerated::<i64>() => Ok(cast($func::<i64>(cast($x), $($arg),*))),
            DType::I32 if is_accelerated::<i32>() => Ok(cast($func::<i32>(cast($x), $($arg),*))),
            DType::I16 if is_accelerated::<i16>() => Ok(cast($func::<i16>(cast($x), $($arg),*))),
            DType::I8 if is_accelerated::<i8>() => Ok(cast($func::<i8>(cast($x), $($arg),*))),
            DType::U64 if is_accelerated::<u64>() => Ok(cast($func::<u64>(cast($x), $($arg),*))),
            DType::U32 if is_accelerated::<u32>() => Ok(cast($func::<u32>(cast($x), $($arg),*))),
            DType::U16 if is_accelerated::<u16>() => Ok(cast($func::<u16>(cast($x), $($arg),*))),
            DType::U8 if is_accelerated::<u8>() => Ok(cast($func::<u8>(cast($x), $($arg),*))),
            DType::Bool if is_accelerated::<u8>() => Ok(cast($func::<u8>(cast($x), $($arg),*))),
            DType::QFloat(scheme) => match scheme.value {
                QuantValue::Q8F | QuantValue::Q8S if is_accelerated::<i8>() => Ok(cast($func::<i8>(cast($x), $($arg),*))),
                _ => Err($x)
            },
            _ => Err($x),
        }
    };
}

pub(crate) fn try_max_pool2d_simd<E: Element>(
    x: SharedArray<E>,
    ksize: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Result<SharedArray<E>, SharedArray<E>> {
    let [_, c, _, _] = x.shape().try_into().unwrap();
    if !should_use_simd(c) || x.strides()[1] != 1 {
        return Err(x);
    }

    launch_kernel!(E, max_pool2d_nhwc, x, ksize, stride, padding, dilation)
}

fn cast<T, E>(tensor: SharedArray<T>) -> SharedArray<E> {
    unsafe { transmute::<SharedArray<T>, SharedArray<E>>(tensor) }
}

mod nhwc {
    use itertools::Itertools;
    use macerator::{Simd, vload_unaligned, vstore_unaligned};
    use ndarray::{ArrayView3, ArrayViewMut3, Ix4};
    use seq_macro::seq;

    use crate::ops::simd::lanes;

    use super::*;

    // Until you can use associated constants as array size, we need to hardcode this.
    // The most common config (x86-v3) has 16 registers, so use half of them for accumulators.
    const BLOCK_REGISTERS: usize = 8;

    pub(crate) fn max_pool2d_nhwc<E: Element + VOrd + MinMax>(
        x: SharedArray<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> SharedArray<E> {
        let [kernel_height, kernel_width] = kernel_size;
        let [pad_h, pad_w] = padding;
        let [stride_height, stride_width] = stride;
        let [dilation_height, dilation_width] = dilation;
        let [batch_size, channels, x_height, x_width] = x.shape().try_into().unwrap();
        let lanes = lanes::<E>();

        let ch_block = lanes * BLOCK_REGISTERS;

        let out_height = ((x_height + 2 * pad_h - dilation_height * (kernel_height - 1) - 1)
            / stride_height)
            + 1;
        let out_width =
            ((x_width + 2 * pad_w - dilation_width * (kernel_width - 1) - 1) / stride_width) + 1;

        let mut output = unsafe {
            Array4::<E>::uninit((batch_size, out_height, out_width, channels)).assume_init()
        };
        let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

        let x = x.into_dimensionality::<Ix4>().unwrap();
        let x = x.view();
        let x = x.permuted_axes([0, 2, 3, 1]);

        // Floor division ensures `blocks * lanes * blocking factor` is always `<= out_channels`.
        // An exclusive loop will always have `lanes * blocking factor` elements in bounds.
        let blocks = channels / ch_block;
        let blocks_end = blocks * ch_block;
        // Floor division means simd_end is always divisible by `lanes` and `<= out_channels`. An
        // exclusive loop will always have `lanes` elements in bounds.
        let simd_end = channels / lanes * lanes;
        let simd_unblocked = (simd_end - blocks_end) / lanes;
        let remainder = channels - simd_end;

        run_par!(|| {
            // SAFETY: Loop ranges are non-overlapping, so the unsafe shared reference is safe.
            iter_range_par!(0, batch_size * blocks).for_each(|k| unsafe {
                let block = k % blocks;
                let b = k / blocks;

                let output = unsafe_shared_out.get();
                let x = x.slice(s![b, .., .., ..]);
                let out = output.slice_mut(s![b, .., .., ..]);
                loop_blocked(x, out, kernel_size, stride, padding, dilation, block);
            });
            // SAFETY: See `loop_unblocked`
            iter_range_par!(0, batch_size * simd_unblocked).for_each(|k| unsafe {
                let ch = (k % simd_unblocked) * lanes + blocks_end;
                let b = k / simd_unblocked;

                let output = unsafe_shared_out.get();
                let x = x.slice(s![b, .., .., ..]);
                let out = output.slice_mut(s![b, .., .., ..]);
                loop_unblocked(x, out, kernel_size, stride, padding, dilation, ch);
            });
            // SAFETY: Loop ranges are non-overlapping, so the unsafe shared reference is safe.
            iter_range_par!(0, batch_size * remainder).for_each(|k| unsafe {
                let ch = (k % remainder) + simd_end;
                let b = k / remainder;

                let output = unsafe_shared_out.get();
                let x = x.slice(s![b, .., .., ..]);
                let out = output.slice_mut(s![b, .., .., ..]);
                loop_scalar(x, out, kernel_size, stride, padding, dilation, ch);
            });
        });

        output = output.permuted_axes([0, 3, 1, 2]);

        output.into_dyn().into_shared()
    }

    /// Execute the blocked (unrolled) portion of the pool.
    #[allow(
        clippy::too_many_arguments,
        clippy::erasing_op,
        clippy::identity_op,
        unused_mut
    )]
    #[inline(always)]
    #[macerator::with_simd]
    fn loop_blocked<'a, S: Simd, E: Element + VOrd + MinMax>(
        x: ArrayView3<'a, E>,
        mut out: ArrayViewMut3<'a, E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        block: usize,
    ) where
        'a: 'a,
    {
        let [kernel_height, kernel_width] = kernel_size;
        let [pad_h, pad_w] = padding;
        let [stride_height, stride_width] = stride;
        let [dilation_height, dilation_width] = dilation;

        let (x_height, x_width, _) = x.dim();
        let (out_height, out_width, _) = out.dim();
        let lanes = E::lanes::<S>();
        let ch_block = lanes * BLOCK_REGISTERS;

        let min = E::MIN.splat::<S>();
        // If outside padding area, kernels are guaranteed to be in bounds
        for oh in pad_h..out_height.saturating_sub(pad_h) {
            for ow in pad_w..out_width.saturating_sub(pad_w) {
                seq!(N in 0..8 {
                    let mut acc~N = min;
                });
                let ch = block * ch_block;
                let ch_end = ch + ch_block;
                let mut out = out.slice_mut(s![oh, ow, ch..ch_end]);

                for kh in 0..kernel_height {
                    let ih = oh * stride_height + kh * dilation_height - pad_h;

                    for kw in 0..kernel_width {
                        let iw = ow * stride_width + kw * dilation_width - pad_w;
                        let x = x.slice(s![ih, iw, ch..ch_end]);

                        seq!(N in 0..8 {
                            // SAFETY:
                            // Load a full vector from x[N * lanes]. This is bounds checked by the
                            // slice above.
                            acc~N = acc~N.max(unsafe { vload_unaligned(&x[N * lanes]) });
                        });
                    }
                }

                seq!(N in 0..8 {
                    // SAFETY:
                    // Store a full vector to out[N * lanes]. This is bounds checked by the
                    // slice above.
                    unsafe { vstore_unaligned(&mut out[N * lanes], acc~N) };
                });
            }
        }

        // Border pixels need bounds checks
        if (pad_h, pad_w) != (0, 0) {
            let v_borders = (0..pad_h)
                .chain(out_height.saturating_sub(pad_h)..out_height)
                .cartesian_product(0..out_width);
            let h_borders = (0..out_height)
                .cartesian_product((0..pad_w).chain(out_width.saturating_sub(pad_w)..out_width));

            for (oh, ow) in v_borders.chain(h_borders) {
                seq!(N in 0..8 {
                    let mut acc~N = min;
                });
                let ch = block * ch_block;
                let ch_end = ch + ch_block;
                let mut out = out.slice_mut(s![oh, ow, ch..ch_end]);

                for kh in 0..kernel_height {
                    let ih = oh * stride_height + kh * dilation_height;
                    if ih < pad_h || ih >= x_height + pad_h {
                        continue;
                    }
                    let ih = ih - pad_h;

                    for kw in 0..kernel_width {
                        let iw = ow * stride_width + kw * dilation_width;
                        if iw < pad_w || iw >= x_width + pad_w {
                            continue;
                        }
                        let iw = iw - pad_w;

                        let x = x.slice(s![ih, iw, ch..ch_end]);

                        seq!(N in 0..8 {
                            // SAFETY:
                            // Load a full vector from x[N * lanes]. This is bounds checked by the
                            // slice above.
                            acc~N = acc~N.max(unsafe { vload_unaligned(&x[N * lanes]) });
                        });
                    }
                }

                seq!(N in 0..8 {
                    // SAFETY:
                    // Store a full vector to out[N * lanes]. This is bounds checked by the
                    // slice above.
                    unsafe { vstore_unaligned(&mut out[N * lanes], acc~N) };
                });
            }
        }
    }

    /// Execute the unblocked (not unrolled) portion of the pool.
    ///
    /// SAFETY: Safe as long as `ch + simd_lanes <= out_channels`.
    #[allow(clippy::too_many_arguments, unused_mut)]
    #[inline(always)]
    #[macerator::with_simd]
    unsafe fn loop_unblocked<'a, S: Simd, E: Element + VOrd + MinMax>(
        x: ArrayView3<'a, E>,
        mut out: ArrayViewMut3<'a, E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ch: usize,
    ) where
        'a: 'a,
    {
        let [kernel_height, kernel_width] = kernel_size;
        let [pad_h, pad_w] = padding;
        let [stride_height, stride_width] = stride;
        let [dilation_height, dilation_width] = dilation;

        let (x_height, x_width, _) = x.dim();
        let (out_height, out_width, _) = out.dim();

        for oh in pad_h..out_height.saturating_sub(pad_h) {
            for ow in pad_w..out_width.saturating_sub(pad_w) {
                let mut acc = E::MIN.splat::<S>();
                let out = &mut out[[oh, ow, ch]];

                for kh in 0..kernel_height {
                    let ih = oh * stride_height + kh * dilation_height - pad_h;

                    for kw in 0..kernel_width {
                        let iw = ow * stride_width + kw * dilation_width - pad_w;
                        // Load a full vector from `x`. In bounds as long as `out_channels >= ch + lanes`
                        acc = acc.max(unsafe { vload_unaligned(&x[[ih, iw, ch]]) });
                    }
                }
                // Store a full vector to `out`. In bounds as long as `out_channels >= ch + lanes`.
                unsafe { vstore_unaligned(out, acc) };
            }
        }

        // Border pixels need bounds checks
        if (pad_h, pad_w) != (0, 0) {
            let v_borders = (0..pad_h)
                .chain(out_height.saturating_sub(pad_h)..out_height)
                .cartesian_product(0..out_width);
            let h_borders = (0..out_height)
                .cartesian_product((0..pad_w).chain(out_width.saturating_sub(pad_w)..out_width));

            for (oh, ow) in v_borders.chain(h_borders) {
                let mut acc = E::MIN.splat::<S>();
                let out = &mut out[[oh, ow, ch]];

                for kh in 0..kernel_height {
                    let ih = oh * stride_height + kh * dilation_height;
                    if ih < pad_h || ih >= x_height + pad_h {
                        continue;
                    }
                    let ih = ih - pad_h;

                    for kw in 0..kernel_width {
                        let iw = ow * stride_width + kw * dilation_width;
                        if iw < pad_w || iw >= x_width + pad_w {
                            continue;
                        }
                        let iw = iw - pad_w;
                        // Load a full vector from `x`. In bounds as long as `out_channels >= ch + lanes`
                        acc = acc.max(unsafe { vload_unaligned(&x[[ih, iw, ch]]) });
                    }
                }
                // Store a full vector to `out`. In bounds as long as `out_channels >= ch + lanes`.
                unsafe { vstore_unaligned(out, acc) };
            }
        }
    }

    fn loop_scalar<E: Element + MinMax>(
        x: ArrayView3<'_, E>,
        mut out: ArrayViewMut3<'_, E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        ch: usize,
    ) {
        let [kernel_height, kernel_width] = kernel_size;
        let [pad_h, pad_w] = padding;
        let [stride_height, stride_width] = stride;
        let [dilation_height, dilation_width] = dilation;

        let (x_height, x_width, _) = x.dim();
        let (out_height, out_width, _) = out.dim();

        for oh in 0..out_height {
            for ow in 0..out_width {
                let mut acc = E::MIN;

                for kh in 0..kernel_height {
                    let ih = oh * stride_height + kh * dilation_height;
                    if ih < pad_h || ih >= x_height + pad_h {
                        continue;
                    }
                    let ih = ih - pad_h;

                    for kw in 0..kernel_width {
                        let iw = ow * stride_width + kw * dilation_width;
                        if iw < pad_w || iw >= x_width + pad_w {
                            continue;
                        }
                        let iw = iw - pad_w;
                        acc = acc.max(x[[ih, iw, ch]]);
                    }
                }

                out[[oh, ow, ch]] = acc;
            }
        }
    }
}
