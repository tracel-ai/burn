use core::mem::transmute;

use crate::{sharing::UnsafeSharedRef, tensor::NdArrayTensor};

use burn_common::{iter_range_par, run_par};
use burn_tensor::{quantization::QuantizationType, DType, Element, TensorMetadata};
use bytemuck::cast_slice_mut;
use macerator::{SimdExt, VOrd};
use ndarray::{s, Array4};
use nhwc::max_pool2d_nhwc;
use pulp::{Arch, Simd};

use super::{should_use_simd, store4_unaligned, MinMax};

macro_rules! launch_kernel {
    ($ty: ty, $func: ident, $x: expr, $($arg: expr),*) => {
        match <$ty as Element>::dtype() {
            DType::F64 => Ok(cast($func::<f64>(cast($x), $($arg),*))),
            DType::F32 => Ok(cast($func::<f32>(cast($x), $($arg),*))),
            DType::F16 | DType::BF16 => Err($x), // Once AVX-512 stabilizes we can use f16
            DType::I64 => Ok(cast($func::<i64>(cast($x), $($arg),*))),
            DType::I32 => Ok(cast($func::<i32>(cast($x), $($arg),*))),
            DType::I16 => Ok(cast($func::<i16>(cast($x), $($arg),*))),
            DType::I8 => Ok(cast($func::<i8>(cast($x), $($arg),*))),
            DType::U64 => Ok(cast($func::<u64>(cast($x), $($arg),*))),
            DType::U32 => Ok(cast($func::<u32>(cast($x), $($arg),*))),
            DType::U16 => Ok(cast($func::<u16>(cast($x), $($arg),*))),
            DType::U8 => Ok(cast($func::<u8>(cast($x), $($arg),*))),
            DType::Bool => Ok(cast($func::<u8>(cast($x), $($arg),*))),
            DType::QFloat(scheme) => match scheme.q_type() {
                QuantizationType::QInt8 => Ok(cast($func::<i8>(cast($x), $($arg),*))),
            },
        }
    };
}

pub(crate) fn try_max_pool2d_simd<E: Element>(
    x: NdArrayTensor<E>,
    ksize: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Result<NdArrayTensor<E>, NdArrayTensor<E>> {
    let [_, c, _, _] = x.shape().dims();
    if !should_use_simd(c) || x.array.strides()[1] != 1 {
        return Err(x);
    }

    launch_kernel!(E, max_pool2d_nhwc, x, ksize, stride, padding, dilation)
}

fn cast<T, E>(tensor: NdArrayTensor<T>) -> NdArrayTensor<E> {
    unsafe { transmute::<NdArrayTensor<T>, NdArrayTensor<E>>(tensor) }
}

mod nhwc {
    use ndarray::Ix4;

    use crate::ops::simd::load4_unaligned;

    use super::*;

    #[pulp::with_simd(max_pool2d_nhwc = Arch::new())]
    pub(crate) fn max_pool2d_nhwc_simd<S: Simd, E: Element + VOrd + MinMax>(
        simd: S,
        x: NdArrayTensor<E>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> NdArrayTensor<E> {
        let [kernel_height, kernel_width] = kernel_size;
        let [pad_h, pad_w] = padding;
        let [stride_height, stride_width] = stride;
        let [dilation_height, dilation_width] = dilation;
        let [batch_size, channels, x_height, x_width] = x.shape().dims();
        let min = E::MIN;
        let lanes = E::lanes::<S>();
        // Until you can use associated constants as array size, we need to hardcode this.
        // The most common config (x86-v3) has 16 registers, so use half of them for accumulators.
        const BLOCK_REGISTERS: usize = 8;
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

        let x = x.array.into_dimensionality::<Ix4>().unwrap();
        let x = x.view();
        let x = x.permuted_axes([0, 2, 3, 1]);

        let blocks = channels.div_ceil(ch_block);

        run_par!(|| {
            iter_range_par!(0, batch_size * blocks).for_each(|k| unsafe {
                let block = k % blocks;
                let b = k / blocks;

                let output = unsafe_shared_out.get();

                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut max_val = [simd.splat(min); BLOCK_REGISTERS];
                        let ch = block * ch_block;
                        let ch_end = (ch + ch_block).min(channels);
                        let mut out = output.slice_mut(s![b, oh, ow, ch..ch_end]);

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

                                let x = x.slice(s![b, ih, iw, ch..ch_end]);

                                let mut c = 0;
                                while c as isize <= x.len() as isize - 4 * lanes as isize {
                                    let (s0, s1, s2, s3) = load4_unaligned(simd, x.as_ptr().add(c));
                                    let c_vec = c / lanes;
                                    max_val[c_vec] = E::vmax(simd, max_val[c_vec], s0);
                                    max_val[c_vec + 1] = E::vmax(simd, max_val[c_vec + 1], s1);
                                    max_val[c_vec + 2] = E::vmax(simd, max_val[c_vec + 2], s2);
                                    max_val[c_vec + 3] = E::vmax(simd, max_val[c_vec + 3], s3);

                                    c += 4 * lanes;
                                }
                                while c as isize <= x.len() as isize - lanes as isize {
                                    let s0 = simd.vload_unaligned(x.as_ptr().add(c));

                                    let c_vec = c / lanes;
                                    max_val[c_vec] = E::vmax(simd, max_val[c_vec], s0);

                                    c += lanes;
                                }
                                for c in c..x.len() {
                                    let max_val = cast_slice_mut(&mut max_val);
                                    max_val[c] = MinMax::max(max_val[c], x[c]);
                                }
                            }
                        }

                        let mut c = 0;
                        while c as isize <= out.len() as isize - 4 * lanes as isize {
                            let c_vec = c / lanes;
                            let s0 = max_val[c_vec];
                            let s1 = max_val[c_vec + 1];
                            let s2 = max_val[c_vec + 2];
                            let s3 = max_val[c_vec + 3];
                            store4_unaligned(simd, out.as_mut_ptr().add(c), s0, s1, s2, s3);
                            c += 4 * lanes;
                        }
                        while c as isize <= out.len() as isize - lanes as isize {
                            simd.vstore_unaligned(out.as_mut_ptr().add(c), max_val[c / lanes]);
                            c += lanes;
                        }
                        for c in c..out.len() {
                            let max_val = cast_slice_mut(&mut max_val);
                            out[c] = max_val[c];
                        }
                    }
                }
            })
        });

        output = output.permuted_axes([0, 3, 1, 2]);

        NdArrayTensor::new(output.into_dyn().into_shared())
    }
}
