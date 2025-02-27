use core::{marker::PhantomData, mem::transmute};
use std::time::Instant;

use burn_common::{iter_range_par, run_par};
use burn_tensor::{
    ops::{conv::calculate_conv_output_size, ConvOptions},
    DType, Element, TensorMetadata,
};
use bytemuck::Zeroable;
use macerator::{SimdExt, VMulAdd};
use ndarray::{s, Array4, Dim, Ix1, Ix4, NdIndex};
use num_traits::Zero;
use pulp::{Arch, Simd};
use seq_macro::seq;

use crate::{FloatNdArrayElement, NdArrayElement, NdArrayTensor, UnsafeSharedRef};

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

#[allow(clippy::result_large_err, clippy::identity_op, clippy::erasing_op)]
#[pulp::with_simd(conv2d = Arch::new())]
pub fn conv2d_simd<S: Simd, E: VMulAdd + Element + Zero, T: Element>(
    simd: S,
    x: NdArrayTensor<T>,
    weight: NdArrayTensor<T>,
    bias: Option<NdArrayTensor<T>>,
    options: ConvOptions<2>,
    _ty: PhantomData<E>,
) -> Result<NdArrayTensor<T>, Args<T>> {
    let [out_channels, in_channels, k_height, k_width] = weight.shape().dims();
    let channels_per_group = out_channels / options.groups;
    let lanes = E::lanes::<S>();

    if channels_per_group % lanes != 0 {
        return Err((x, weight, bias));
    }

    let x = cast::<_, E>(x);
    let weight = cast::<_, E>(weight);
    let bias = bias.map(|bias| cast::<_, E>(bias));

    let [dilate_h, dilate_w] = options.dilation;
    let [stride_h, stride_w] = options.stride;
    let [pad_h, pad_w] = options.padding;
    let [batch_size, _in_channels, in_height, in_width] = x.shape().dims();

    const N_REG: usize = 8;

    let oc_b = channels_per_group.min(lanes);
    let ow_b = N_REG;

    let out_height = calculate_conv_output_size(k_height, 1, pad_h, dilate_h, in_height);
    let out_width = calculate_conv_output_size(k_width, 1, pad_w, dilate_w, in_width);

    let x = x.array.into_dimensionality::<Ix4>().unwrap();
    let weights = weight.array.into_dimensionality::<Ix4>().unwrap();
    let weights = weights.permuted_axes([1, 2, 3, 0]);
    let weights = weights.as_standard_layout();
    let bias = bias.map(|bias| bias.array.into_dimensionality::<Ix1>().unwrap());

    let mut out = Array4::<E>::zeros(Dim([batch_size, out_height, out_width, out_channels]));
    let unsafe_shared_out = UnsafeSharedRef::new(&mut out);

    let oc_blocks = out_channels.div_ceil(oc_b);
    let ow_blocks = out_width.div_ceil(ow_b);

    run_par!(|| {
        iter_range_par!(0, batch_size * oc_blocks).for_each(|k| unsafe {
            let ob = k % oc_blocks;
            let b = k / oc_blocks;
            let oc = ob * oc_b;
            let out = unsafe_shared_out.get();
            let bias = if let Some(bias) = &bias {
                simd.vload_unaligned(&bias[oc])
            } else {
                Zeroable::zeroed()
            };

            let x = x.slice(s![b, .., .., ..]);
            for oh in 0..out_height {
                let mut out = out.slice_mut(s![b, oh, .., ..]);
                let ih = oh * stride_h;
                for ow_block in 0..ow_blocks {
                    seq!(N in 0..8 {
                        let mut acc~N = bias;
                    });
                    let ow = ow_block * ow_b;
                    let ow_end = ow_b.min(out_width - ow);
                    let aligned = ow_end == ow_b;

                    for ic in 0..in_channels {
                        for kh in 0..k_height {
                            let ih = ih + kh * dilate_h;
                            if ih < pad_h || ih >= in_height + pad_h {
                                continue;
                            }
                            let ih = ih - pad_h;

                            for kw in 0..k_width {
                                let f0 = simd.vload_unaligned(&weights[[ic, kh, kw, oc]]);
                                let iw = ow * stride_w + kw * dilate_w;

                                if aligned
                                    && iw >= pad_w
                                    && iw + ow_b * stride_w < in_width + pad_h
                                {
                                    let iw = iw - pad_w;
                                    seq!(N in 0..8 {
                                        let i~N = simd.splat(*x.uget([ic, ih, iw + N * stride_w]));
                                    });
                                    seq!(N in 0..8 {
                                        acc~N = E::vmuladd(simd, i~N, f0, acc~N);
                                    });
                                } else {
                                    seq!(N in 0..8 {
                                        {
                                            let ow = ow + N;
                                            if iw >= pad_w && iw < in_width + pad_w && ow < out_width {
                                                let iw = iw - pad_w;
                                                let i~N = simd.splat(*x.uget([ic, ih, iw + N * stride_w]));
                                                acc~N = E::vmuladd(simd, i~N, f0, acc~N);
                                            }
                                        }
                                    });
                                }
                            }
                        }
                    }

                    if aligned {
                        seq!(N in 0..8 {
                            simd.vstore_unaligned(&mut out[[ow + N, oc]], acc~N);
                        });
                    } else {
                        seq!(N in 0..8 {
                            #[allow(clippy::identity_op)]
                            if ow + N >= out_width {
                                continue;
                            }
                            #[allow(clippy::identity_op)]
                            simd.vstore_unaligned(&mut out[[ow + N, oc]], acc~N);
                        });
                    }
                }
            }
        });
    });

    let output = out.permuted_axes([0, 3, 1, 2]);
    Ok(cast(NdArrayTensor::new(output.into_dyn().into_shared())))
}
