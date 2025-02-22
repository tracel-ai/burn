use std::{fmt::Debug, ptr::null_mut};

use burn_tensor::Shape;
use bytemuck::{cast_slice, cast_slice_mut, Zeroable};
use macerator::VOrd;
use pulp::Simd;

use super::filter::{
    MaxOp, MinOp, MorphColumnFilter, MorphColumnVec, MorphOperator, MorphRowFilter, MorphRowVec,
    VecMorphOperator,
};

pub type RowFilter<T, Op> = MorphRowFilter<T, Op, MorphRowVec<T, Op>>;
pub type ErodeRow<T> = RowFilter<T, MinOp>;
pub type DilateRow<T> = RowFilter<T, MaxOp>;
pub type ColFilter<T, Op> = MorphColumnFilter<T, Op, MorphColumnVec<T, Op>>;
pub type ErodeCol<T> = ColFilter<T, MinOp>;
pub type DilateCol<T> = ColFilter<T, MaxOp>;

pub struct FilterEngine<S: Simd, T: VOrd, Op: MorphOperator<T> + VecMorphOperator<T>> {
    /// Vector aligned ring buffer to serve as intermediate, since image isn't always aligned
    ring_buf: Vec<T::Vector<S>>,
    /// Vector aligned row buffer to serve as intermediate, since image isn't always aligned
    src_row: Vec<T::Vector<S>>,
    const_border_value: Vec<T>,
    const_border_row: Vec<T::Vector<S>>,
    /// Pointers to each row offset in the ring buffer
    rows: Vec<*const T>,

    row_filter: RowFilter<T, Op>,
    col_filter: ColFilter<T, Op>,

    ksize: (usize, usize),
    anchor: (usize, usize),
    dx1: usize,
    dx2: usize,
    row_count: usize,
    dst_y: usize,
    start_y: usize,
    start_y_0: usize,
    end_y: usize,

    max_width: usize,
    buf_step: usize,
    width: usize,
    height: usize,
}

impl<S: Simd, T: VOrd, Op: MorphOperator<T> + VecMorphOperator<T>> FilterEngine<S, T, Op> {
    fn resize_ring_buf(&mut self, size: usize) {
        let actual = size.div_ceil(T::lanes::<S>());
        self.ring_buf.resize(actual, Zeroable::zeroed());
    }
    fn resize_src_row(&mut self, size: usize) {
        let actual = size.div_ceil(T::lanes::<S>());
        self.src_row.resize(actual, Zeroable::zeroed());
    }
}

impl<S: Simd, T: VOrd + Debug, Op: MorphOperator<T> + VecMorphOperator<T>> FilterEngine<S, T, Op> {
    pub fn new(
        row_filter: RowFilter<T, Op>,
        col_filter: ColFilter<T, Op>,
        border_value: &[T],
    ) -> Self {
        let ch = border_value.len();
        let ksize = (col_filter.ksize, row_filter.ksize);
        let anchor = (col_filter.anchor, row_filter.anchor);
        let border_length = (ksize.1 - 1).max(1);
        let mut const_border_value: Vec<T> = vec![Zeroable::zeroed(); border_length * ch];
        for elem in cast_slice_mut::<_, T>(&mut const_border_value).chunks_exact_mut(ch) {
            elem.copy_from_slice(border_value);
        }

        Self {
            ring_buf: Default::default(),
            src_row: Default::default(),
            rows: Default::default(),
            const_border_row: Default::default(),
            const_border_value,
            ksize,
            anchor,
            row_filter,
            col_filter,
            max_width: 0,
            buf_step: 0,
            dx1: 0,
            dx2: 0,
            row_count: 0,
            dst_y: 0,
            start_y: 0,
            start_y_0: 0,
            end_y: 0,
            width: 0,
            height: 0,
        }
    }

    pub fn apply(&mut self, simd: S, src: &[T], src_shape: Shape, dst: &mut [T], dst_shape: Shape) {
        let [_, w, ch] = src_shape.dims();
        let src_step = w * ch;
        let [_, w, ch] = dst_shape.dims();
        let dst_step = w * ch;
        self.start(simd, src_shape);
        let y = self.start_y;
        self.proceed(
            simd,
            &src[y * src_step..],
            src_step,
            self.end_y - self.start_y,
            dst,
            dst_step,
            ch,
        );
    }

    pub fn start(&mut self, simd: S, shape: Shape) -> usize {
        let [height, width, ch] = shape.dims();

        let max_buf_rows = (self.ksize.0 + 3)
            .max(self.anchor.0)
            .max((self.ksize.0 - self.anchor.0 - 1) * 2 + 1);

        if self.max_width < width || max_buf_rows != self.rows.len() {
            self.rows.resize(max_buf_rows, null_mut());
            self.max_width = self.max_width.max(width);
            self.resize_src_row((self.max_width + self.ksize.1 - 1) * ch);

            self.const_border_row.resize(
                ((self.max_width + self.ksize.1 - 1) * ch).div_ceil(T::lanes::<S>()),
                Zeroable::zeroed(),
            );
            let mut n = self.const_border_value.len();
            let n1 = (self.max_width + self.ksize.1 - 1) * ch;
            let const_val = &self.const_border_value;
            let dst = cast_slice_mut(&mut self.const_border_row);
            let t_dst = cast_slice_mut::<_, T>(&mut self.src_row);

            for i in (0..n1).step_by(n) {
                n = n.min(n1 - i);
                t_dst[i..i + n].copy_from_slice(&const_val[..n]);
            }

            self.row_filter
                .apply(simd, cast_slice(&self.src_row), dst, self.max_width, ch);

            let max_buf_step = self.max_width.next_multiple_of(align_of::<T::Vector<S>>()) * ch;

            self.resize_ring_buf(max_buf_step * self.rows.len());
        }

        let const_val = &self.const_border_value;

        self.buf_step = width.next_multiple_of(align_of::<T::Vector<S>>()) * ch;

        self.dx1 = self.anchor.1;
        self.dx2 = self.ksize.1 - self.anchor.1 - 1;

        if self.dx1 > 0 || self.dx2 > 0 {
            let nr = 1;
            for _ in 0..nr {
                let dst = cast_slice_mut::<_, T>(&mut self.src_row);
                memcpy(dst, const_val, self.dx1 * ch);
                let right = (width + self.ksize.1 - 1 - self.dx2) * ch;
                memcpy(&mut dst[right..], const_val, self.dx2 * ch);
            }
        }

        self.end_y = height;
        self.width = width;
        self.height = height;

        self.start_y
    }

    #[allow(clippy::too_many_arguments)]
    pub fn proceed(
        &mut self,
        simd: S,
        src: &[T],
        src_step: usize,
        mut count: usize,
        dst: &mut [T],
        dst_step: usize,
        ch: usize,
    ) -> usize {
        let buf_rows = self.rows.len();
        let kheight = self.ksize.0;
        let kwidth = self.ksize.1;
        let ay = self.anchor.0 as isize;
        let dx1 = self.dx1;
        let dx2 = self.dx2;
        let width1 = self.width + kwidth - 1;
        count = count.min(self.remaining_input_rows());
        let mut dst_off = 0;
        let mut src_off = 0;
        let mut dy = 0;
        let mut i;
        let brows = &mut self.rows;

        let src_row = cast_slice_mut::<_, T>(&mut self.src_row);
        let ring_buf = cast_slice_mut::<_, T>(&mut self.ring_buf);

        loop {
            let dcount = buf_rows as isize - ay - self.start_y as isize - self.row_count as isize;
            let mut dcount = if dcount > 0 {
                dcount as usize
            } else {
                buf_rows + 1 - kheight
            };
            dcount = dcount.min(count);
            count -= dcount;

            while dcount > 0 {
                let bi = (self.start_y - self.start_y_0 + self.row_count) % buf_rows;
                let brow = &mut ring_buf[bi * self.buf_step..];

                if self.row_count + 1 > buf_rows {
                    self.row_count -= 1;
                    self.start_y += 1;
                }
                self.row_count += 1;

                memcpy(
                    &mut src_row[dx1 * ch..],
                    &src[src_off..],
                    (width1 - dx2 - dx1) * ch,
                );

                self.row_filter.apply(simd, src_row, brow, self.width, ch);

                dcount -= 1;
                src_off += src_step;
            }

            let max_i = buf_rows.min(self.height - (self.dst_y + dy) + (kheight - 1));
            i = 0;
            while i < max_i {
                let src_y = border_interpolate((self.dst_y + dy + i) as isize - ay, self.height);
                if src_y < 0 {
                    brows[i] = self.const_border_row.as_ptr() as _;
                } else {
                    if src_y as usize >= self.start_y + self.row_count {
                        break;
                    }
                    let bi = (src_y as usize - self.start_y_0) % buf_rows;
                    brows[i] = unsafe { ring_buf.as_ptr().add(bi * self.buf_step) };
                }

                i += 1;
            }
            if i < kheight {
                break;
            }
            i -= kheight - 1;
            self.col_filter.apply(
                simd,
                brows,
                &mut dst[dst_off..],
                dst_step,
                i,
                self.width * ch,
            );

            dst_off += dst_step * i;
            dy += i;
        }

        self.dst_y += dy;
        dy
    }

    fn remaining_input_rows(&self) -> usize {
        self.end_y - self.start_y - self.row_count
    }
}

fn memcpy<T: Copy>(to: &mut [T], from: &[T], len: usize) {
    to[..len].copy_from_slice(&from[..len]);
}

fn border_interpolate(mut p: isize, len: usize) -> isize {
    if p >= len as isize {
        p = -1;
    }
    p
}
