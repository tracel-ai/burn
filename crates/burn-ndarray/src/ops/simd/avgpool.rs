use core::mem::transmute;

use crate::{sharing::UnsafeSharedRef, tensor::NdArrayTensor};

use burn_common::{iter_range_par, run_par};
use burn_tensor::{DType, Element, ElementConversion, TensorMetadata};
use bytemuck::Zeroable;
use macerator::{SimdExt, VAdd, VDiv};
use ndarray::{s, Array4};
use nhwc::avg_pool_nhwc;
use pulp::{Arch, Simd};

use super::should_use_simd;

pub(crate) fn try_avg_pool2d_simd<E: Element>(
    x: NdArrayTensor<E>,
    ksize: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    with_pad: bool,
) -> Result<NdArrayTensor<E>, NdArrayTensor<E>> {
    // Strides must be unit, dilation isn't supported, rows must be contiguous
    if x.array.strides()[1] != 1 || !should_use_simd(x.array.shape()[1]) {
        return Err(x);
    }

    match E::dtype() {
        DType::F64 => Ok(cast(avg_pool_nhwc::<f64>(
            cast(x),
            ksize,
            stride,
            padding,
            with_pad,
        ))),
        DType::F32 => Ok(cast(avg_pool_nhwc::<f32>(
            cast(x),
            ksize,
            stride,
            padding,
            with_pad,
        ))),
        _ => Err(x),
    }
}

fn cast<T, E>(tensor: NdArrayTensor<T>) -> NdArrayTensor<E> {
    unsafe { transmute::<NdArrayTensor<T>, NdArrayTensor<E>>(tensor) }
}

mod nhwc {
    use itertools::Itertools;
    use ndarray::{ArrayView3, ArrayViewMut3};
    use seq_macro::seq;

    use crate::ops::simd::lanes;

    use super::*;

    // Until you can use associated constants as array size, we need to hardcode this.
    // The most common config (x86-v3) has 16 registers, so use half of them for accumulators.
    const BLOCK_REGISTERS: usize = 8;

    pub(crate) fn avg_pool_nhwc<E: Element + VAdd + VDiv>(
        x: NdArrayTensor<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        with_pad: bool,
    ) -> NdArrayTensor<E> {
        let [kernel_height, kernel_width] = kernel_size;
        let [pad_h, pad_w] = padding;
        let [stride_height, stride_width] = stride;
        let [batch_size, channels, x_height, x_width] = x.shape().dims();
        let lanes = lanes::<E>();

        let ch_block = lanes * BLOCK_REGISTERS;

        let out_height = ((x_height + 2 * pad_h - (kernel_height - 1) - 1) / stride_height) + 1;
        let out_width = ((x_width + 2 * pad_w - (kernel_width - 1) - 1) / stride_width) + 1;

        let mut output = unsafe {
            Array4::<E>::uninit((batch_size, out_height, out_width, channels)).assume_init()
        };
        let unsafe_shared_out = UnsafeSharedRef::new(&mut output);
        let x = x.array.view();
        let x = x.permuted_axes(vec![0, 2, 3, 1]);

        let blocks = channels / ch_block;
        let blocks_end = blocks * ch_block;
        let simd_end = channels / lanes * lanes;
        let num_simd_unblocked = (simd_end - blocks_end) / lanes;
        let remainder = channels - simd_end;

        run_par!(|| {
            iter_range_par!(0, batch_size * blocks).for_each(|k| unsafe {
                let block = k % blocks;
                let b = k / blocks;

                let output = unsafe_shared_out.get();

                let x = x.slice(s![b, .., .., ..]);
                let out = output.slice_mut(s![b, .., .., ..]);

                loop_blocked(x, out, kernel_size, stride, padding, with_pad, block);
            });
            iter_range_par!(0, batch_size * num_simd_unblocked).for_each(|k| unsafe {
                let ch = (k % num_simd_unblocked) * lanes + blocks_end;
                let b = k / num_simd_unblocked;

                let output = unsafe_shared_out.get();

                let x = x.slice(s![b, .., .., ..]);
                let out = output.slice_mut(s![b, .., .., ..]);

                loop_unblocked(x, out, kernel_size, stride, padding, with_pad, ch);
            });
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

        NdArrayTensor::new(output.into_dyn().into_shared())
    }

    #[allow(clippy::too_many_arguments, clippy::erasing_op, clippy::identity_op)]
    #[pulp::with_simd(loop_blocked = Arch::new())]
    unsafe fn loop_blocked_simd<S: Simd, E: Element + VAdd + VDiv>(
        simd: S,
        x: ArrayView3<'_, E>,
        mut out: ArrayViewMut3<'_, E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        with_pad: bool,
        block: usize,
    ) {
        let [kernel_height, kernel_width] = kernel_size;
        let [pad_h, pad_w] = padding;
        let [stride_height, stride_width] = stride;

        let (x_height, x_width, _) = x.dim();
        let (out_height, out_width, _) = out.dim();
        let lanes = E::lanes::<S>();

        let ch_block = lanes * BLOCK_REGISTERS;

        // If pixels are not within padding range, bounds checks are always true
        for oh in pad_h..out_height - pad_h {
            for ow in pad_w..out_width - pad_w {
                seq!(N in 0..8 {
                    let mut sum~N = Zeroable::zeroed();
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
                            let s~N = simd.vload_unaligned(x.as_ptr().add(N * lanes));
                            sum~N = E::vadd(simd, sum~N, s~N);
                        });
                    }
                }

                let count = kernel_height * kernel_width;
                let count_v = simd.splat((count as u64).elem::<E>());
                seq!(N in 0..8 {
                    let s~N = E::vdiv(simd, sum~N, count_v);
                    simd.vstore_unaligned(out.as_mut_ptr().add(N * lanes), s~N);
                });
            }
        }

        // Border pixels need bounds checks
        if (pad_h, pad_w) != (0, 0) {
            let v_borders = (0..pad_h)
                .chain(out_height - pad_h..out_height)
                .cartesian_product(0..out_width);
            let h_borders =
                (0..out_height).cartesian_product((0..pad_w).chain(out_width - pad_w..out_width));

            for (oh, ow) in v_borders.chain(h_borders) {
                seq!(N in 0..8 {
                    let mut sum~N = Zeroable::zeroed();
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
                            let s~N = simd.vload_unaligned(x.as_ptr().add(N * lanes));
                            sum~N = E::vadd(simd, sum~N, s~N);
                        });
                    }
                }

                if with_pad {
                    count = kernel_height * kernel_width;
                }

                let count_v = simd.splat((count as u64).elem::<E>());
                seq!(N in 0..8 {
                    let s~N = E::vdiv(simd, sum~N, count_v);
                    simd.vstore_unaligned(out.as_mut_ptr().add(N * lanes), s~N);
                });
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[pulp::with_simd(loop_unblocked = Arch::new())]
    unsafe fn loop_unblocked_simd<S: Simd, E: Element + VAdd + VDiv>(
        simd: S,
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
        for oh in pad_h..out_height - pad_h {
            for ow in pad_w..out_width - pad_w {
                let mut sum = Zeroable::zeroed();

                for kh in 0..kernel_height {
                    let ih = oh * stride_height + kh - pad_h;

                    for kw in 0..kernel_width {
                        let iw = ow * stride_width + kw - pad_w;
                        let s0 = simd.vload_unaligned(&x[[ih, iw, ch]]);
                        sum = E::vadd(simd, sum, s0);
                    }
                }

                let count = kernel_height * kernel_width;
                let count_v = simd.splat((count as u64).elem::<E>());
                let s0 = E::vdiv(simd, sum, count_v);
                simd.vstore_unaligned(&mut out[[oh, ow, ch]], s0);
            }
        }

        // Border pixels need bounds checks
        if (pad_h, pad_w) != (0, 0) {
            let v_borders = (0..pad_h)
                .chain(out_height - pad_h..out_height)
                .cartesian_product(0..out_width);
            let h_borders =
                (0..out_height).cartesian_product((0..pad_w).chain(out_width - pad_w..out_width));

            for (oh, ow) in v_borders.chain(h_borders) {
                let mut sum = Zeroable::zeroed();
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

                        let s0 = simd.vload_unaligned(&x[[ih, iw, ch]]);
                        sum = E::vadd(simd, sum, s0);
                    }
                }

                if with_pad {
                    count = kernel_height * kernel_width;
                }

                let count_v = simd.splat((count as u64).elem::<E>());
                let s0 = E::vdiv(simd, sum, count_v);
                simd.vstore_unaligned(&mut out[[oh, ow, ch]], s0);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn loop_scalar<E: Element + VAdd + VDiv>(
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
        for oh in pad_h..out_height - pad_h {
            for ow in pad_w..out_width - pad_w {
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
                .chain(out_height - pad_h..out_height)
                .cartesian_product(0..out_width);
            let h_borders =
                (0..out_height).cartesian_product((0..pad_w).chain(out_width - pad_w..out_width));

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
