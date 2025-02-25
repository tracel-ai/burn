use core::{mem::transmute, ops::AddAssign, ptr::null};

use crate::{sharing::UnsafeSharedRef, tensor::NdArrayTensor};

use burn_common::{iter_range_par, run_par};
use burn_tensor::{DType, Element, ElementConversion, TensorMetadata};
use bytemuck::{cast_slice, cast_slice_mut, Zeroable};
use macerator::{SimdExt, VAdd, VDiv};
use ndarray::{s, Array4, ArrayView2, ArrayViewMut2};
use pulp::{Arch, Simd};

use super::{load2, load4, store2, store2_unaligned, store4, store4_unaligned};

pub(crate) fn try_avg_pool2d_simd<E: Element>(
    x: NdArrayTensor<E>,
    ksize: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    with_pad: bool,
) -> Result<NdArrayTensor<E>, NdArrayTensor<E>> {
    // Strides must be unit, dilation isn't supported, rows must be contiguous
    if stride != [1, 1] || x.array.strides()[3] != 1 {
        return Err(x);
    }

    match E::dtype() {
        DType::F64 => Ok(cast(avg_pool2d::<f64>(cast(x), ksize, padding, with_pad))),
        DType::F32 => Ok(cast(avg_pool2d::<f32>(cast(x), ksize, padding, with_pad))),
        _ => Err(x),
    }
}

fn cast<T, E>(tensor: NdArrayTensor<T>) -> NdArrayTensor<E> {
    unsafe { transmute::<NdArrayTensor<T>, NdArrayTensor<E>>(tensor) }
}

/// SIMD version of maxpool - requires unit stride and no dilation. Fall back to non-SIMD if that
/// requirement isn't met.
#[pulp::with_simd(avg_pool2d = Arch::new())]
pub(crate) fn avg_pool2d_simd<S: Simd, E: Element + VAdd + VDiv + AddAssign>(
    simd: S,
    x: NdArrayTensor<E>,
    kernel_size: [usize; 2],
    padding: [usize; 2],
    with_pad: bool,
) -> NdArrayTensor<E> {
    let [k_height, k_width] = kernel_size;
    let [pad_h, pad_w] = padding;
    let [batch_size, channels, x_height, x_width] = x.shape().dims();
    let lanes = E::lanes::<S>();
    let x = x.array;
    let pad_value = 0.elem();

    let buf_rows = (k_height + 3).max((k_height.div_ceil(2) - 1) * 2 + 1);
    let width1 = x_width + k_width - 1;

    let out_height = x_height + 2 * pad_h - (k_height - 1) - 1 + 1;
    let out_width = x_width + 2 * pad_w - (k_width - 1) - 1 + 1;

    let mut const_border_row: Vec<E::Vector<S>> = vec![Zeroable::zeroed(); width1.div_ceil(lanes)];
    let const_border_row = cast_slice_mut::<_, E>(&mut const_border_row);
    const_border_row.fill(pad_value);

    let mut src_row: Vec<E::Vector<S>> = vec![Zeroable::zeroed(); width1.div_ceil(lanes)];
    let const_count_row: Vec<E::Vector<S>> = if with_pad {
        let mut row = vec![Zeroable::zeroed(); width1.div_ceil(lanes)];
        cast_slice_mut::<_, E>(&mut row)[pad_w..pad_w + x_width].fill(1.elem());
        row
    } else {
        vec![]
    };
    let const_count_row = cast_slice::<_, E>(&const_count_row);

    // Keep buffer in step with alignment so we can always have aligned loads/stores
    let buf_step = out_width.next_multiple_of(align_of::<E::Vector<S>>());

    if padding != [0, 0] {
        let src_row = cast_slice_mut(&mut src_row);
        src_row[..pad_w].fill(pad_value);
        let right = x_width + pad_w;
        src_row[right..right + pad_w].fill(pad_value);
    }

    let ay = pad_h as isize;

    // We write to every spot
    let mut output =
        unsafe { Array4::uninit((batch_size, channels, out_height, out_width)).assume_init() };
    let unsafe_shared_out = UnsafeSharedRef::new(&mut output);

    run_par!(|| {
        iter_range_par!(0, batch_size * channels).for_each(|k| unsafe {
            let mut rows = vec![null(); buf_rows];
            let mut cnt_rows = vec![null(); buf_rows * !with_pad as usize];
            let mut src_row = src_row.clone();
            let src_row = cast_slice_mut(&mut src_row);
            let mut ring_buf: Vec<E::Vector<S>> = vec![Zeroable::zeroed(); buf_step * buf_rows];
            let ring_buf = cast_slice_mut::<_, E>(&mut ring_buf);
            // Total count for each output pixel
            let mut cnt_buf: Vec<E::Vector<S>> = if !with_pad {
                vec![Zeroable::zeroed(); buf_step * buf_rows]
            } else {
                vec![]
            };
            let cnt_buf = cast_slice_mut::<_, E>(&mut cnt_buf);

            let b = k / channels;
            let c = k % channels;

            let mut row_count = 0;
            let mut count = x_height;
            let mut start_y = 0;
            let mut src_y = 0;
            let mut dy = 0;
            let mut i;

            let src: ArrayView2<E> = x.slice(s![b, c, .., ..]);
            let output = unsafe_shared_out.get();
            let mut dst: ArrayViewMut2<E> = output.slice_mut(s![b, c, .., ..]);

            loop {
                let dcount = buf_rows as isize - ay - start_y as isize - row_count as isize;
                let mut dcount = if dcount > 0 {
                    dcount as usize
                } else {
                    buf_rows + 1 - k_height
                };
                dcount = dcount.min(count);
                count -= dcount;

                while dcount > 0 {
                    let buf_y = (start_y + row_count) % buf_rows;
                    let buf_off = buf_y * buf_step;
                    let buf_row = &mut ring_buf[buf_off..buf_off + out_width];
                    let cnt_row = &mut cnt_buf
                        [buf_off * !with_pad as usize..(buf_off + out_width) * !with_pad as usize];

                    row_count += 1;
                    if row_count > buf_rows {
                        row_count -= 1;
                        start_y += 1;
                    }

                    // Load row
                    src_row[pad_w..pad_w + x_width]
                        .copy_from_slice(src.row(src_y).as_slice().unwrap());

                    avg_pool_row(
                        simd,
                        src_row,
                        const_count_row,
                        buf_row,
                        cnt_row,
                        k_width,
                        with_pad,
                    );

                    dcount -= 1;
                    src_y += 1;
                }

                let max_i = buf_rows.min(out_height - dy + k_height - 1);
                i = 0;

                while i < max_i {
                    let src_y = (dy + i) as isize - ay;
                    if src_y < 0 || src_y as usize >= start_y + row_count {
                        rows[i] = const_border_row.as_ptr();
                        if !with_pad {
                            cnt_rows[i] = const_border_row.as_ptr();
                        }
                    } else {
                        let buf_y = src_y as usize % buf_rows;
                        rows[i] = ring_buf.as_ptr().add(buf_y * buf_step);
                        if !with_pad {
                            cnt_rows[i] = cnt_buf.as_ptr().add(buf_y * buf_step);
                        }
                    }

                    i += 1;
                }

                if i < k_height {
                    break;
                }
                i -= k_height - 1;

                avg_pool_col(
                    simd,
                    &rows,
                    &cnt_rows,
                    dst.slice_mut(s![dy.., ..]),
                    k_height,
                    ((k_height * k_width) as u64).elem(),
                    i,
                    with_pad,
                );

                dy += i;
            }
        })
    });

    NdArrayTensor::new(output.into_dyn().into_shared())
}

fn avg_pool_row<S: Simd, T: VAdd + Element>(
    simd: S,
    row: &[T],
    row_cnt: &[T],
    dst: &mut [T],
    dst_cnt: &mut [T],
    ksize: usize,
    with_pad: bool,
) {
    let width = dst.len() as isize;
    let lanes = T::lanes::<S>();
    let mut ow = 0;

    let mut count0 = T::splat(simd, 1.elem());
    let mut count1 = count0;
    let mut count2 = count0;
    let mut count3 = count0;

    unsafe {
        let row = row.as_ptr();
        let cnt = row_cnt.as_ptr();
        let dst = dst.as_mut_ptr();
        let dst_cnt = dst_cnt.as_mut_ptr();

        while ow as isize <= width - 4 * lanes as isize {
            let sptr = row.add(ow);
            let (mut s0, mut s1, mut s2, mut s3) = load4(simd, row);

            if !with_pad {
                let cptr = cnt.add(ow);
                count0 = simd.vload(cptr);
                count1 = simd.vload(cptr.add(lanes));
                count2 = simd.vload(cptr.add(2 * lanes));
                count3 = simd.vload(cptr.add(3 * lanes));
            }

            for kw in 1..ksize {
                let sptr = sptr.add(kw);
                s0 = T::vadd(simd, s0, simd.vload_unaligned(sptr));
                s1 = T::vadd(simd, s1, simd.vload_unaligned(sptr.add(lanes)));
                s2 = T::vadd(simd, s2, simd.vload_unaligned(sptr.add(2 * lanes)));
                s3 = T::vadd(simd, s3, simd.vload_unaligned(sptr.add(3 * lanes)));

                if !with_pad {
                    let sptr = cnt.add(ow + kw);
                    count0 = T::vadd(simd, count0, simd.vload_unaligned(sptr));
                    count1 = T::vadd(simd, count1, simd.vload_unaligned(sptr.add(lanes)));
                    count2 = T::vadd(simd, count2, simd.vload_unaligned(sptr.add(2 * lanes)));
                    count3 = T::vadd(simd, count3, simd.vload_unaligned(sptr.add(3 * lanes)));
                }
            }

            store4(simd, dst.add(ow), s0, s1, s2, s3);

            if !with_pad {
                store4(simd, dst_cnt.add(ow), count0, count1, count2, count3);
            }

            ow += 4 * lanes;
        }
        if ow as isize <= width - 2 * lanes as isize {
            let sptr = row.add(ow);
            let (mut s0, mut s1) = load2(simd, sptr);

            if !with_pad {
                let cptr = cnt.add(ow);
                count0 = simd.vload(cptr);
                count1 = simd.vload(cptr.add(lanes));
            }

            for kw in 1..ksize {
                let sptr = sptr.add(kw);
                s0 = T::vadd(simd, s0, simd.vload_unaligned(sptr));
                s1 = T::vadd(simd, s1, simd.vload_unaligned(sptr.add(lanes)));

                if !with_pad {
                    let sptr = cnt.add(ow + kw);
                    count0 = T::vadd(simd, count0, simd.vload_unaligned(sptr));
                    count1 = T::vadd(simd, count1, simd.vload_unaligned(sptr.add(lanes)));
                }
            }

            store2(simd, dst.add(ow), s0, s1);
            if !with_pad {
                store2(simd, dst_cnt.add(ow), count0, count1);
            }

            ow += 2 * lanes;
        }
        if ow as isize <= width - lanes as isize {
            let mut s0 = simd.vload(row.add(ow));
            if !with_pad {
                count0 = simd.vload(cnt.add(ow));
            }

            for kw in 1..ksize {
                s0 = T::vadd(simd, s0, simd.vload_unaligned(row.add(ow + kw)));

                if !with_pad {
                    count0 = T::vadd(simd, count0, simd.vload_unaligned(cnt.add(ow + kw)));
                }
            }

            simd.vstore(dst.add(ow), s0);
            if !with_pad {
                simd.vstore(dst_cnt.add(ow), count0);
            }

            ow += lanes;
        }
        if ow as isize <= width - lanes as isize / 2 {
            let mut s0 = simd.vload_low(row.add(ow));
            if !with_pad {
                count0 = simd.vload_low(cnt.add(ow));
            }

            for kw in 1..ksize {
                s0 = T::vadd(simd, s0, simd.vload_low(row.add(ow + kw)));
                if !with_pad {
                    count0 = T::vadd(simd, count0, simd.vload_low(cnt.add(ow + kw)));
                }
            }

            simd.vstore_low(dst.add(ow), s0);
            if !with_pad {
                simd.vstore_low(dst_cnt.add(ow), count0);
            }

            ow += lanes / 2;
        }
    }

    #[allow(clippy::needless_range_loop)]
    for ow in ow..width as usize {
        let mut s0 = row[ow];
        let mut count0 = if !with_pad { row_cnt[ow] } else { 1.elem() };
        for k in 1..ksize {
            s0 = s0.add(row[ow + k]);
            if !with_pad {
                count0 = count0.add(row_cnt[ow + k]);
            }
        }
        dst[ow] = s0;
        if !with_pad {
            dst_cnt[ow] = count0;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn avg_pool_col<S: Simd, T: VAdd + VDiv + Element>(
    simd: S,
    src: &[*const T],
    src_cnt: &[*const T],
    mut dst: ArrayViewMut2<T>,
    ksize: usize,
    total_ksize: T,
    mut count: usize,
    with_pad: bool,
) {
    let width = dst.shape()[1] as isize;
    let lanes = T::lanes::<S>();

    let mut x;
    let mut y = 0;

    let mut count0 = T::splat(simd, total_ksize);
    let mut count1 = count0;
    let mut count2 = count0;
    let mut count3 = count0;

    unsafe {
        while count > 1 && ksize > 1 {
            let dst0 = dst.row_mut(y).as_mut_ptr();
            let dst1 = dst.row_mut(y + 1).as_mut_ptr();
            x = 0;
            while x as isize <= width - 4 * lanes as isize {
                let (mut s0, mut s1, mut s2, mut s3) = load4(simd, src[y + 1].add(x));

                if !with_pad {
                    let sptr = src_cnt[y + 1].add(x);
                    count0 = simd.vload(sptr);
                    count1 = simd.vload(sptr.add(lanes));
                    count2 = simd.vload(sptr.add(2 * lanes));
                    count3 = simd.vload(sptr.add(3 * lanes));
                }

                for k in 2..ksize {
                    let sptr = src[y + k].add(x);
                    s0 = T::vadd(simd, s0, simd.vload(sptr));
                    s1 = T::vadd(simd, s1, simd.vload(sptr.add(lanes)));
                    s2 = T::vadd(simd, s2, simd.vload(sptr.add(2 * lanes)));
                    s3 = T::vadd(simd, s3, simd.vload(sptr.add(3 * lanes)));
                    if !with_pad {
                        let sptr = src_cnt[y + k].add(x);
                        count0 = T::vadd(simd, count0, simd.vload(sptr));
                        count1 = T::vadd(simd, count1, simd.vload(sptr.add(lanes)));
                        count2 = T::vadd(simd, count2, simd.vload(sptr.add(2 * lanes)));
                        count3 = T::vadd(simd, count3, simd.vload(sptr.add(3 * lanes)));
                    }
                }

                // Row 1
                {
                    let sptr = src[y].add(x);
                    let mut s0 = T::vadd(simd, s0, simd.vload(sptr));
                    let mut s1 = T::vadd(simd, s1, simd.vload(sptr.add(lanes)));
                    let mut s2 = T::vadd(simd, s2, simd.vload(sptr.add(2 * lanes)));
                    let mut s3 = T::vadd(simd, s3, simd.vload(sptr.add(3 * lanes)));
                    if !with_pad {
                        let sptr = src_cnt[y].add(x);
                        let count0 = T::vadd(simd, count0, simd.vload(sptr));
                        let count1 = T::vadd(simd, count1, simd.vload(sptr.add(lanes)));
                        let count2 = T::vadd(simd, count2, simd.vload(sptr.add(2 * lanes)));
                        let count3 = T::vadd(simd, count3, simd.vload(sptr.add(3 * lanes)));
                        s0 = T::vdiv(simd, s0, count0);
                        s1 = T::vdiv(simd, s1, count1);
                        s2 = T::vdiv(simd, s2, count2);
                        s3 = T::vdiv(simd, s3, count3);
                    } else {
                        s0 = T::vdiv(simd, s0, count0);
                        s1 = T::vdiv(simd, s1, count1);
                        s2 = T::vdiv(simd, s2, count2);
                        s3 = T::vdiv(simd, s3, count3);
                    }
                    store4_unaligned(simd, dst0.add(x), s0, s1, s2, s3);
                }

                // Row 2
                {
                    let sptr = src[y + ksize].add(x);
                    let mut s0 = T::vadd(simd, s0, simd.vload(sptr));
                    let mut s1 = T::vadd(simd, s1, simd.vload(sptr.add(lanes)));
                    let mut s2 = T::vadd(simd, s2, simd.vload(sptr.add(2 * lanes)));
                    let mut s3 = T::vadd(simd, s3, simd.vload(sptr.add(3 * lanes)));
                    if !with_pad {
                        let sptr = src_cnt[y + ksize].add(x);
                        count0 = T::vadd(simd, count0, simd.vload(sptr));
                        count1 = T::vadd(simd, count1, simd.vload(sptr.add(lanes)));
                        count2 = T::vadd(simd, count2, simd.vload(sptr.add(2 * lanes)));
                        count3 = T::vadd(simd, count3, simd.vload(sptr.add(3 * lanes)));
                    }
                    s0 = T::vdiv(simd, s0, count0);
                    s1 = T::vdiv(simd, s1, count1);
                    s2 = T::vdiv(simd, s2, count2);
                    s3 = T::vdiv(simd, s3, count3);
                    store4_unaligned(simd, dst1.add(x), s0, s1, s2, s3);
                }

                x += 4 * lanes;
            }
            if x as isize <= width - 2 * lanes as isize {
                let (mut s0, mut s1) = load2(simd, src[y + 1].add(x));

                if !with_pad {
                    let sptr = src_cnt[y + 1].add(x);
                    count0 = simd.vload(sptr);
                    count1 = simd.vload(sptr.add(lanes));
                }

                for k in 2..ksize {
                    let sptr = src[y + k].add(x);
                    s0 = T::vadd(simd, s0, simd.vload(sptr));
                    s1 = T::vadd(simd, s1, simd.vload(sptr.add(lanes)));
                    if !with_pad {
                        let sptr = src_cnt[y + k].add(x);
                        count0 = T::vadd(simd, count0, simd.vload(sptr));
                        count1 = T::vadd(simd, count1, simd.vload(sptr.add(lanes)));
                    }
                }

                // Row 1
                {
                    let sptr = src[y].add(x);
                    let mut s0 = T::vadd(simd, s0, simd.vload(sptr));
                    let mut s1 = T::vadd(simd, s1, simd.vload(sptr.add(lanes)));
                    if !with_pad {
                        let sptr = src_cnt[y].add(x);
                        let count0 = T::vadd(simd, count0, simd.vload(sptr));
                        let count1 = T::vadd(simd, count1, simd.vload(sptr.add(lanes)));
                        s0 = T::vdiv(simd, s0, count0);
                        s1 = T::vdiv(simd, s1, count1);
                    } else {
                        s0 = T::vdiv(simd, s0, count0);
                        s1 = T::vdiv(simd, s1, count1);
                    }
                    store2_unaligned(simd, dst0.add(x), s0, s1);
                }

                // Row 2
                {
                    let sptr = src[y + ksize].add(x);
                    let mut s0 = T::vadd(simd, s0, simd.vload(sptr));
                    let mut s1 = T::vadd(simd, s1, simd.vload(sptr.add(lanes)));
                    if !with_pad {
                        let sptr = src_cnt[y + ksize].add(x);
                        count0 = T::vadd(simd, count0, simd.vload(sptr));
                        count1 = T::vadd(simd, count1, simd.vload(sptr.add(lanes)));
                    }
                    s0 = T::vdiv(simd, s0, count0);
                    s1 = T::vdiv(simd, s1, count1);
                    store2_unaligned(simd, dst1.add(x), s0, s1);
                }

                x += 2 * lanes;
            }
            if x as isize <= width - lanes as isize {
                let mut s0 = simd.vload(src[y + 1].add(x));
                if !with_pad {
                    count0 = simd.vload(src_cnt[y + 1].add(x));
                }

                for k in 2..ksize {
                    s0 = T::vadd(simd, s0, simd.vload(src[y + k].add(x)));
                    if !with_pad {
                        count0 = T::vadd(simd, count0, simd.vload(src_cnt[y + k].add(x)));
                    }
                }

                // Row 1
                {
                    let mut s0 = T::vadd(simd, s0, simd.vload(src[y].add(x)));
                    if !with_pad {
                        let count0 = T::vadd(simd, count0, simd.vload(src_cnt[y].add(x)));
                        s0 = T::vdiv(simd, s0, count0);
                    } else {
                        s0 = T::vdiv(simd, s0, count0);
                    }
                    simd.vstore_unaligned(dst0.add(x), s0);
                }

                // Row 2
                {
                    let mut s0 = T::vadd(simd, s0, simd.vload(src[y + ksize].add(x)));
                    if !with_pad {
                        count0 = T::vadd(simd, count0, simd.vload(src_cnt[y + ksize].add(x)));
                    }
                    s0 = T::vdiv(simd, s0, count0);
                    simd.vstore_unaligned(dst1.add(x), s0);
                }

                x += lanes;
            }
            if x as isize <= width - lanes as isize / 2 {
                let mut s0 = simd.vload_low(src[y + 1].add(x));
                if !with_pad {
                    count0 = simd.vload_low(src_cnt[y + 1].add(x));
                }

                for k in 2..ksize {
                    s0 = T::vadd(simd, s0, simd.vload_low(src[y + k].add(x)));
                    if !with_pad {
                        count0 = T::vadd(simd, count0, simd.vload_low(src_cnt[y + k].add(x)));
                    }
                }

                // Row 1
                {
                    let mut s0 = T::vadd(simd, s0, simd.vload_low(src[y].add(x)));
                    if !with_pad {
                        let count0 = T::vadd(simd, count0, simd.vload_low(src_cnt[y].add(x)));
                        s0 = T::vdiv(simd, s0, count0);
                    } else {
                        s0 = T::vdiv(simd, s0, count0);
                    }
                    simd.vstore_low(dst0.add(x), s0);
                }

                // Row 2
                {
                    let mut s0 = T::vadd(simd, s0, simd.vload_low(src[y + ksize].add(x)));
                    if !with_pad {
                        count0 = T::vadd(simd, count0, simd.vload_low(src_cnt[y + ksize].add(x)));
                    }
                    s0 = T::vdiv(simd, s0, count0);
                    simd.vstore_low(dst1.add(x), s0);
                }

                x += lanes / 2;
            }
            for x in x..width as usize {
                let mut s0 = *src[y + 1].add(x);
                let mut count0 = if !with_pad {
                    *src_cnt[y + 1]
                } else {
                    total_ksize
                };

                for k in 2..ksize {
                    s0 = s0.add(*src[y + k].add(x));
                    if !with_pad {
                        count0 = count0.add(*src_cnt[y + k].add(x));
                    }
                }

                if !with_pad {
                    let count0 = count0 + *src_cnt[y];
                    dst[[y, x]] = s0.add(*src[y].add(x)).div(count0);
                } else {
                    dst[[y, x]] = s0.add(*src[y].add(x)).div(count0);
                }
                if !with_pad {
                    count0 = count0 + *src_cnt[y];
                }
                dst[[y + 1, x]] = s0.add(*src[y + ksize].add(x)).div(count0);
            }
            count -= 2;
            y += 2;
        }
        while count > 0 {
            let dst0 = dst.row_mut(y).as_mut_ptr();

            x = 0;
            while x as isize <= width - 4 * lanes as isize {
                let (mut s0, mut s1, mut s2, mut s3) = load4(simd, src[y].add(x));
                if !with_pad {
                    let sptr = src_cnt[y].add(x);
                    count0 = simd.vload(sptr);
                    count1 = simd.vload(sptr.add(lanes));
                    count2 = simd.vload(sptr.add(2 * lanes));
                    count3 = simd.vload(sptr.add(3 * lanes));
                }

                for k in 1..ksize {
                    let sptr = src[y + k].add(x);
                    s0 = T::vadd(simd, s0, simd.vload(sptr));
                    s1 = T::vadd(simd, s1, simd.vload(sptr.add(lanes)));
                    s2 = T::vadd(simd, s2, simd.vload(sptr.add(2 * lanes)));
                    s3 = T::vadd(simd, s3, simd.vload(sptr.add(3 * lanes)));
                    if !with_pad {
                        let sptr = src_cnt[y + k].add(x);
                        count0 = T::vadd(simd, count0, simd.vload(sptr));
                        count1 = T::vadd(simd, count1, simd.vload(sptr.add(lanes)));
                        count2 = T::vadd(simd, count2, simd.vload(sptr.add(2 * lanes)));
                        count3 = T::vadd(simd, count3, simd.vload(sptr.add(3 * lanes)));
                    }
                }

                s0 = T::vdiv(simd, s0, count0);
                s1 = T::vdiv(simd, s1, count1);
                s2 = T::vdiv(simd, s2, count2);
                s3 = T::vdiv(simd, s3, count3);
                store4_unaligned(simd, dst0.add(x), s0, s1, s2, s3);

                x += 4 * lanes;
            }
            if x as isize <= width - 2 * lanes as isize {
                let (mut s0, mut s1) = load2(simd, src[y].add(x));
                if !with_pad {
                    let sptr = src_cnt[y].add(x);
                    count0 = simd.vload(sptr);
                    count1 = simd.vload(sptr.add(lanes));
                }

                for k in 1..ksize {
                    let sptr = src[y + k].add(x);
                    s0 = T::vadd(simd, s0, simd.vload(sptr));
                    s1 = T::vadd(simd, s1, simd.vload(sptr.add(lanes)));
                    if !with_pad {
                        let sptr = src_cnt[y + k].add(x);
                        count0 = T::vadd(simd, count0, simd.vload(sptr));
                        count1 = T::vadd(simd, count1, simd.vload(sptr.add(lanes)));
                    }
                }

                s0 = T::vdiv(simd, s0, count0);
                s1 = T::vdiv(simd, s1, count1);
                store2_unaligned(simd, dst0.add(x), s0, s1);

                x += 2 * lanes;
            }
            if x as isize <= width - lanes as isize {
                let mut s0 = simd.vload(src[y].add(x));
                if !with_pad {
                    count0 = simd.vload(src_cnt[y].add(x));
                }

                for k in 1..ksize {
                    s0 = T::vadd(simd, s0, simd.vload(src[y + k].add(x)));
                    if !with_pad {
                        count0 = T::vadd(simd, count0, simd.vload(src_cnt[y + k].add(x)));
                    }
                }

                s0 = T::vdiv(simd, s0, count0);
                simd.vstore_unaligned(dst0.add(x), s0);
                x += lanes;
            }
            if x as isize <= width - lanes as isize / 2 {
                let mut s0 = simd.vload_low(src[y].add(x));
                if !with_pad {
                    count0 = simd.vload_low(src_cnt[y].add(x));
                }

                for k in 1..ksize {
                    s0 = T::vadd(simd, s0, simd.vload_low(src[y + k].add(x)));
                    if !with_pad {
                        count0 = T::vadd(simd, count0, simd.vload_low(src_cnt[y + k].add(x)));
                    }
                }

                s0 = T::vdiv(simd, s0, count0);
                simd.vstore_low(dst0.add(x), s0);
                x += lanes / 2;
            }
            for x in x..width as usize {
                let mut s0 = *src[y].add(x);
                let mut count0 = if !with_pad {
                    *src_cnt[y + 1]
                } else {
                    total_ksize
                };

                for k in 1..ksize {
                    s0 = s0.add(*src[y + k].add(x));
                    if !with_pad {
                        count0 = count0.add(*src_cnt[y + k].add(x));
                    }
                }

                s0 = s0.div(count0);
                dst[[y, x]] = s0;
            }

            count -= 1;
            y += 1;
        }
    }
}
