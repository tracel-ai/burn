use core::mem::transmute;

use crate::{sharing::UnsafeSharedRef, tensor::NdArrayTensor};

use burn_common::{iter_range_par, run_par};
use burn_tensor::{DType, Element, ElementConversion, TensorMetadata};
use bytemuck::{cast_slice_mut, Zeroable};
use macerator::{SimdExt, VAdd, VDiv};
use ndarray::{s, Array4};
use nhwc::avg_pool_nhwc;
use pulp::{Arch, Simd};

use super::{should_use_simd, store4_unaligned};

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
    use crate::ops::simd::load4_unaligned;

    use super::*;

    #[pulp::with_simd(avg_pool_nhwc = Arch::new())]
    pub(crate) fn avg_pool2d_nhwc_simd<S: Simd, E: Element + VAdd + VDiv>(
        simd: S,
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
        let lanes = E::lanes::<S>();
        // Until you can use associated constants as array size, we need to hardcode this.
        // The most common config (x86-v3) has 16 registers, so use half of them for accumulators.
        const BLOCK_REGISTERS: usize = 8;
        let ch_block = lanes * BLOCK_REGISTERS;

        let out_height = ((x_height + 2 * pad_h - (kernel_height - 1) - 1) / stride_height) + 1;
        let out_width = ((x_width + 2 * pad_w - (kernel_width - 1) - 1) / stride_width) + 1;

        let mut output = unsafe {
            Array4::<E>::uninit((batch_size, out_height, out_width, channels)).assume_init()
        };
        let unsafe_shared_out = UnsafeSharedRef::new(&mut output);
        let x = x.array.view();
        let x = x.permuted_axes(vec![0, 2, 3, 1]);

        let blocks = channels.div_ceil(ch_block);

        run_par!(|| {
            iter_range_par!(0, batch_size * blocks).for_each(|k| unsafe {
                let block = k % blocks;
                let b = k / blocks;

                let output = unsafe_shared_out.get();

                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = [Zeroable::zeroed(); BLOCK_REGISTERS];
                        let mut count: E = 0.elem();
                        let ch = block * ch_block;
                        let ch_end = (ch + ch_block).min(channels);
                        let mut out = output.slice_mut(s![b, oh, ow, ch..ch_end]);

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
                                count = count + 1.elem();

                                let x = x.slice(s![b, ih, iw, ch..ch_end]);

                                let mut c = 0;
                                while c as isize <= x.len() as isize - 4 * lanes as isize {
                                    let (s0, s1, s2, s3) = load4_unaligned(simd, x.as_ptr().add(c));
                                    let c_vec = c / lanes;
                                    sum[c_vec] = E::vadd(simd, sum[c_vec], s0);
                                    sum[c_vec + 1] = E::vadd(simd, sum[c_vec + 1], s1);
                                    sum[c_vec + 2] = E::vadd(simd, sum[c_vec + 2], s2);
                                    sum[c_vec + 3] = E::vadd(simd, sum[c_vec + 3], s3);

                                    c += 4 * lanes;
                                }
                                while c as isize <= x.len() as isize - lanes as isize {
                                    let s0 = simd.vload_unaligned(x.as_ptr().add(c));

                                    let c_vec = c / lanes;
                                    sum[c_vec] = E::vadd(simd, sum[c_vec], s0);

                                    c += lanes;
                                }
                                for c in c..x.len() {
                                    let sum = cast_slice_mut::<_, E>(&mut sum);
                                    sum[c] = sum[c].add(x[c]);
                                }
                            }
                        }

                        if with_pad {
                            count = ((kernel_height * kernel_width) as u64).elem();
                        }

                        let count_v = simd.splat(count);
                        let mut c = 0;
                        while c as isize <= out.len() as isize - 4 * lanes as isize {
                            let c_vec = c / lanes;
                            let s0 = E::vdiv(simd, sum[c_vec], count_v);
                            let s1 = E::vdiv(simd, sum[c_vec + 1], count_v);
                            let s2 = E::vdiv(simd, sum[c_vec + 2], count_v);
                            let s3 = E::vdiv(simd, sum[c_vec + 3], count_v);
                            store4_unaligned(simd, out.as_mut_ptr().add(c), s0, s1, s2, s3);
                            c += 4 * lanes;
                        }
                        while c as isize <= out.len() as isize - lanes as isize {
                            let s0 = E::vdiv(simd, sum[c / lanes], count_v);
                            simd.vstore_unaligned(out.as_mut_ptr().add(c), s0);
                            c += lanes;
                        }
                        for c in c..out.len() {
                            let sum = cast_slice_mut::<_, E>(&mut sum);
                            out[c] = sum[c].div(count);
                        }
                    }
                }
            })
        });

        output = output.permuted_axes([0, 3, 1, 2]);

        NdArrayTensor::new(output.into_dyn().into_shared())
    }
}
