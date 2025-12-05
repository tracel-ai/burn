use core::{marker::PhantomData, mem::transmute};

use crate::{SharedArray, iter_range_par, run_par, sharing::UnsafeSharedRef};

use burn_tensor::{DType, Element, ElementConversion};
use bytemuck::Zeroable;
use macerator::{Simd, VAdd, VDiv};
use ndarray::{Array4, s};
use nhwc::avg_pool_nhwc;

use super::should_use_simd;

#[macerator::with_simd]
fn is_accelerated<S: Simd, T: VAdd + VDiv>(_x: PhantomData<T>) -> bool {
    <T as VAdd>::is_accelerated::<S>() && <T as VDiv>::is_accelerated::<S>()
}

pub(crate) fn try_avg_pool2d_simd<E: Element>(
    x: SharedArray<E>,
    ksize: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    with_pad: bool,
) -> Result<SharedArray<E>, SharedArray<E>> {
    // Strides must be unit, dilation isn't supported, rows must be contiguous
    if x.strides()[1] != 1 || !should_use_simd(x.shape()[1]) {
        return Err(x);
    }

    match E::dtype() {
        DType::F64 if is_accelerated::<f64>(PhantomData) => Ok(cast(avg_pool_nhwc::<f64>(
            cast(x),
            ksize,
            stride,
            padding,
            with_pad,
        ))),
        DType::F32 if is_accelerated::<f32>(PhantomData) => Ok(cast(avg_pool_nhwc::<f32>(
            cast(x),
            ksize,
            stride,
            padding,
            with_pad,
        ))),
        DType::F16 if is_accelerated::<half::f16>(PhantomData) => {
            Ok(cast(avg_pool_nhwc::<half::f16>(
                cast(x),
                ksize,
                stride,
                padding,
                with_pad,
            )))
        }
        _ => Err(x),
    }
}

fn cast<T, E>(tensor: SharedArray<T>) -> SharedArray<E> {
    unsafe { transmute::<SharedArray<T>, SharedArray<E>>(tensor) }
}

mod nhwc {
    use itertools::Itertools;
    use macerator::{Simd, Vector, vload_unaligned, vstore_unaligned};
    use ndarray::{ArrayView3, ArrayViewMut3};
    use seq_macro::seq;

    use crate::ops::simd::lanes;

    use super::*;

    // Until you can use associated constants as array size, we need to hardcode this.
    // The most common config (x86-v3) has 16 registers, so use half of them for accumulators.
    const BLOCK_REGISTERS: usize = 8;

    pub(crate) fn avg_pool_nhwc<E: Element + VAdd + VDiv>(
        x: SharedArray<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        with_pad: bool,
    ) -> SharedArray<E> {
        let [kernel_height, kernel_width] = kernel_size;
        let [pad_h, pad_w] = padding;
        let [stride_height, stride_width] = stride;
        let [batch_size, channels, x_height, x_width] = x.shape().try_into().unwrap();
        let lanes = lanes::<E>();

        let ch_block = lanes * BLOCK_REGISTERS;

        let out_height = ((x_height + 2 * pad_h - (kernel_height - 1) - 1) / stride_height) + 1;
        let out_width = ((x_width + 2 * pad_w - (kernel_width - 1) - 1) / stride_width) + 1;

        let mut output = unsafe {
            Array4::<E>::uninit((batch_size, out_height, out_width, channels)).assume_init()
        };
        let unsafe_shared_out = UnsafeSharedRef::new(&mut output);
        let x = x.view();
        let x = x.permuted_axes(vec![0, 2, 3, 1]);

        // Floor division ensures `blocks * lanes * blocking factor` is always `<= out_channels`.
        // An exclusive loop will always have `lanes * blocking factor` elements in bounds.
        let blocks = channels / ch_block;
        let blocks_end = blocks * ch_block;
        // Floor division means simd_end is always divisible by `lanes` and `<= out_channels`. An
        // exclusive loop will always have `lanes` elements in bounds.
        let simd_end = channels / lanes * lanes;
        let num_simd_unblocked = (simd_end - blocks_end) / lanes;
        let remainder = channels - simd_end;

        run_par!(|| {
            // SAFETY: Loop ranges are non-overlapping, so the unsafe shared reference is safe.
            iter_range_par!(0, batch_size * blocks).for_each(|k| unsafe {
                let block = k % blocks;
                let b = k / blocks;

                let output = unsafe_shared_out.get();

                let x = x.slice(s![b, .., .., ..]);
                let out = output.slice_mut(s![b, .., .., ..]);

                loop_blocked(x, out, kernel_size, stride, padding, with_pad, block);
            });
            // SAFETY: See `loop_unblocked`
            iter_range_par!(0, batch_size * num_simd_unblocked).for_each(|k| unsafe {
                let ch = (k % num_simd_unblocked) * lanes + blocks_end;
                let b = k / num_simd_unblocked;

                let output = unsafe_shared_out.get();

                let x = x.slice(s![b, .., .., ..]);
                let out = output.slice_mut(s![b, .., .., ..]);

                loop_unblocked(x, out, kernel_size, stride, padding, with_pad, ch);
            });
            // SAFETY: Loop ranges are non-overlapping, so the unsafe shared reference is safe.
            iter_range_par!(0, batch_size * remainder).for_each(|k| unsafe {
                let ch = (k % remainder) + simd_end;
                let b = k / remainder;

                let output = unsafe_shared_out.get();

                let x = x.slice(s![b, .., .., ..]);
                let out = output.slice_mut(s![b, .., .., ..]);

                loop_scalar(x, out, kernel_size, stride, padding, with_pad, ch);
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
    #[macerator::with_simd]
    fn loop_blocked<'a, S: Simd, E: Element + VAdd + VDiv>(
        x: ArrayView3<'a, E>,
        mut out: ArrayViewMut3<'a, E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        with_pad: bool,
        block: usize,
    ) where
        'a: 'a,
    {
        let [kernel_height, kernel_width] = kernel_size;
        let [pad_h, pad_w] = padding;
        let [stride_height, stride_width] = stride;

        let (x_height, x_width, _) = x.dim();
        let (out_height, out_width, _) = out.dim();
        let lanes = E::lanes::<S>();

        let ch_block = lanes * BLOCK_REGISTERS;

        // If pixels are more than `padding` from the edges, the in pixel cannot be out of bounds
        for oh in pad_h..out_height.saturating_sub(pad_h) {
            for ow in pad_w..out_width.saturating_sub(pad_w) {
                seq!(N in 0..8 {
                    let mut sum~N: Vector<S, E> = Zeroable::zeroed();
                });
                let ch = block * ch_block;
                let ch_end = ch + ch_block;
                let mut out = out.slice_mut(s![oh, ow, ch..ch_end]);

                for kh in 0..kernel_height {
                    let ih = oh * stride_height + kh - pad_h;

                    for kw in 0..kernel_width {
                        let iw = ow * stride_width + kw - pad_w;
                        let x = x.slice(s![ih, iw, ch..ch_end]);

                        seq!(N in 0..8 {
                            // SAFETY:
                            // Load a full vector from x[N * lanes]. This is bounds checked by the
                            // slice above.
                            sum~N += unsafe { vload_unaligned(&x[N * lanes]) };
                        });
                    }
                }

                let count = kernel_height * kernel_width;
                let count = (count as u64).elem::<E>();
                let count_v = count.splat();
                seq!(N in 0..8 {
                    let s~N = sum~N / count_v;
                    // SAFETY:
                    // Store a full vector to out[N * lanes]. This is bounds checked by the
                    // slice above.
                    unsafe { vstore_unaligned(&mut out[N * lanes], s~N) };
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
                    let mut sum~N: Vector<S, E> = Zeroable::zeroed();
                });
                let mut count: usize = 0;
                let ch = block * ch_block;
                let ch_end = ch + ch_block;
                let mut out = out.slice_mut(s![oh, ow, ch..ch_end]);

                for kh in 0..kernel_height {
                    let ih = oh * stride_height + kh;
                    if ih < pad_h || ih >= x_height + pad_h {
                        continue;
                    }
                    let ih = ih - pad_h;

                    for kw in 0..kernel_width {
                        let iw = ow * stride_width + kw;
                        if iw < pad_w || iw >= x_width + pad_w {
                            continue;
                        }
                        let iw = iw - pad_w;
                        count += 1;

                        let x = x.slice(s![ih, iw, ch..ch_end]);

                        seq!(N in 0..8 {
                            // SAFETY:
                            // Load a full vector from x[N * lanes]. This is bounds checked by the
                            // slice above.
                            sum~N += unsafe { vload_unaligned(&x[N * lanes]) };
                        });
                    }
                }

                if with_pad {
                    count = kernel_height * kernel_width;
                }

                let count = (count as u64).elem::<E>();
                let count_v = count.splat();
                seq!(N in 0..8 {
                    let s~N = sum~N / count_v;
                    // SAFETY:
                    // Store a full vector to out[N * lanes]. This is bounds checked by the
                    // slice above.
                    unsafe { vstore_unaligned(&mut out[N * lanes], s~N) };
                });
            }
        }
    }

    /// Execute the unblocked (not unrolled) portion of the pool.
    ///
    /// SAFETY: Safe as long as `ch + simd_lanes <= out_channels`.
    #[allow(clippy::too_many_arguments, unused_mut)]
    #[macerator::with_simd]
    unsafe fn loop_unblocked<'a, S: Simd, E: Element + VAdd + VDiv>(
        x: ArrayView3<'a, E>,
        mut out: ArrayViewMut3<'a, E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        with_pad: bool,
        ch: usize,
    ) where
        'a: 'a,
    {
        let [kernel_height, kernel_width] = kernel_size;
        let [pad_h, pad_w] = padding;
        let [stride_height, stride_width] = stride;

        let (x_height, x_width, _) = x.dim();
        let (out_height, out_width, _) = out.dim();

        // If pixels are not within padding range, bounds checks are always true
        for oh in pad_h..out_height - pad_h {
            for ow in pad_w..out_width - pad_w {
                let mut sum: Vector<S, E> = Zeroable::zeroed();

                for kh in 0..kernel_height {
                    let ih = oh * stride_height + kh - pad_h;

                    for kw in 0..kernel_width {
                        let iw = ow * stride_width + kw - pad_w;
                        // Load a full vector from `x`. In bounds as long as `out_channels >= ch + lanes`
                        let s0 = unsafe { vload_unaligned(&x[[ih, iw, ch]]) };
                        sum += s0;
                    }
                }

                let count = kernel_height * kernel_width;
                let count: E = (count as u64).elem();
                let count_v = count.splat();
                let s0 = sum / count_v;
                // Store a full vector to `out`. In bounds as long as `out_channels >= ch + lanes`.
                unsafe { vstore_unaligned(&mut out[[oh, ow, ch]], s0) };
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
                let mut sum: Vector<S, E> = Zeroable::zeroed();
                let mut count: usize = 0;

                for kh in 0..kernel_height {
                    let ih = oh * stride_height + kh;
                    if ih < pad_h || ih >= x_height + pad_h {
                        continue;
                    }
                    let ih = ih - pad_h;

                    for kw in 0..kernel_width {
                        let iw = ow * stride_width + kw;
                        if iw < pad_w || iw >= x_width + pad_w {
                            continue;
                        }
                        let iw = iw - pad_w;
                        count += 1;

                        // Load a full vector from `x`. In bounds as long as `out_channels >= ch + lanes`
                        sum += unsafe { vload_unaligned(&x[[ih, iw, ch]]) };
                    }
                }

                if with_pad {
                    count = kernel_height * kernel_width;
                }

                let count = (count as u64).elem::<E>();
                let count_v = count.splat();
                let s0 = sum / count_v;
                // Store a full vector to `out`. In bounds as long as `out_channels >= ch + lanes`.
                unsafe { vstore_unaligned(&mut out[[oh, ow, ch]], s0) };
            }
        }
    }

    /// Execute scalar portion of the pooling
    #[allow(clippy::too_many_arguments)]
    fn loop_scalar<E: Element + VAdd + VDiv>(
        x: ArrayView3<'_, E>,
        mut out: ArrayViewMut3<'_, E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        with_pad: bool,
        ch: usize,
    ) {
        let [kernel_height, kernel_width] = kernel_size;
        let [pad_h, pad_w] = padding;
        let [stride_height, stride_width] = stride;

        let (x_height, x_width, _) = x.dim();
        let (out_height, out_width, _) = out.dim();

        // If pixels are not within padding range, bounds checks are always true
        for oh in pad_h..out_height.saturating_sub(pad_h) {
            for ow in pad_w..out_width.saturating_sub(pad_w) {
                let mut sum: E = Zeroable::zeroed();

                for kh in 0..kernel_height {
                    let ih = oh * stride_height + kh - pad_h;

                    for kw in 0..kernel_width {
                        let iw = ow * stride_width + kw - pad_w;
                        sum = sum + x[[ih, iw, ch]];
                    }
                }

                let count = (kernel_height * kernel_width) as u64;
                out[[oh, ow, ch]] = sum / count.elem();
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
                let mut sum: E = Zeroable::zeroed();
                let mut count: usize = 0;

                for kh in 0..kernel_height {
                    let ih = oh * stride_height + kh;
                    if ih < pad_h || ih >= x_height + pad_h {
                        continue;
                    }
                    let ih = ih - pad_h;

                    for kw in 0..kernel_width {
                        let iw = ow * stride_width + kw;
                        if iw < pad_w || iw >= x_width + pad_w {
                            continue;
                        }
                        let iw = iw - pad_w;
                        count += 1;
                        sum = sum + x[[ih, iw, ch]];
                    }
                }

                if with_pad {
                    count = kernel_height * kernel_width;
                }

                out[[oh, ow, ch]] = sum / (count as u64).elem();
            }
        }
    }
}
