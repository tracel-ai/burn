use std::{fmt::Debug, ptr::null_mut};

use burn_tensor::Shape;
use bytemuck::{Zeroable, cast_slice, cast_slice_mut};
use macerator::{Simd, VOrd, Vector};

use crate::{BorderType, Point, Size};

use super::filter::{
    MorphColumnFilter, MorphColumnVec, MorphFilter, MorphOperator, MorphRowFilter, MorphRowVec,
    MorphVec, VecMorphOperator,
};

pub type RowFilter<T, Op> = MorphRowFilter<T, Op, MorphRowVec<T, Op>>;
pub type ColFilter<T, Op> = MorphColumnFilter<T, Op, MorphColumnVec<T, Op>>;
pub type Filter2D<T, Op> = MorphFilter<T, Op, MorphVec<T, Op>>;

pub enum Filter<T: VOrd, Op: MorphOperator<T> + VecMorphOperator<T>> {
    Separable {
        row_filter: RowFilter<T, Op>,
        col_filter: ColFilter<T, Op>,
    },
    Fallback(Filter2D<T, Op>),
}

pub struct FilterEngine<S: Simd, T: VOrd, Op: MorphOperator<T> + VecMorphOperator<T>> {
    /// Vector aligned ring buffer to serve as intermediate, since image isn't always aligned
    ring_buf: Vec<Vector<S, T>>,
    /// Vector aligned row buffer to serve as intermediate, since image isn't always aligned
    src_row: Vec<Vector<S, T>>,
    const_border_value: Vec<T>,
    const_border_row: Vec<Vector<S, T>>,
    border_table: Vec<usize>,
    /// Pointers to each row offset in the ring buffer
    rows: Vec<*const T>,

    filter: Filter<T, Op>,

    ksize: Size,
    anchor: Point,
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
    border_type: BorderType,
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
    fn is_separable(&self) -> bool {
        matches!(self.filter, Filter::Separable { .. })
    }
}

impl<S: Simd, T: VOrd + Debug, Op: MorphOperator<T> + VecMorphOperator<T>> FilterEngine<S, T, Op> {
    pub fn new(
        filter: Filter<T, Op>,
        border_type: BorderType,
        border_value: &[T],
        ch: usize,
    ) -> Self {
        let (ksize, anchor) = match &filter {
            Filter::Separable {
                row_filter,
                col_filter,
            } => {
                let ksize = Size::new(row_filter.ksize, col_filter.ksize);
                let anchor = Point::new(row_filter.anchor, col_filter.anchor);
                (ksize, anchor)
            }
            Filter::Fallback(f) => (f.ksize, f.anchor),
        };

        let mut border_table = Vec::new();
        let border_length = (ksize.width - 1).max(1);
        let mut const_border_value = Vec::new();
        if matches!(border_type, BorderType::Constant) {
            const_border_value = vec![Zeroable::zeroed(); border_length * ch];
            for elem in cast_slice_mut::<_, T>(&mut const_border_value).chunks_exact_mut(ch) {
                elem.copy_from_slice(border_value);
            }
        } else {
            border_table = vec![0; border_length * ch];
        }

        Self {
            ring_buf: Default::default(),
            src_row: Default::default(),
            rows: Default::default(),
            border_type,
            const_border_row: Default::default(),
            const_border_value,
            border_table,
            ksize,
            anchor,
            filter,
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

    pub fn apply(&mut self, tensor: &mut [T], src_shape: Shape) {
        let [_, w, ch] = src_shape.dims();
        let src_step = w * ch;
        self.start(src_shape);
        let y = self.start_y;
        self.proceed(
            &mut tensor[y * src_step..],
            src_step,
            self.end_y - self.start_y,
            ch,
        );
    }

    pub fn start(&mut self, shape: Shape) -> usize {
        let [height, width, ch] = shape.dims();

        let max_buf_rows = (self.ksize.height + 3)
            .max(self.anchor.y)
            .max((self.ksize.height - self.anchor.y - 1) * 2 + 1);
        let k_offs = if !self.is_separable() {
            self.ksize.width - 1
        } else {
            0
        };
        let is_sep = self.is_separable();

        if self.max_width < width || max_buf_rows != self.rows.len() {
            self.rows.resize(max_buf_rows, null_mut());
            self.max_width = self.max_width.max(width);
            self.resize_src_row((self.max_width + self.ksize.width - 1) * ch);

            if matches!(self.border_type, BorderType::Constant) {
                self.const_border_row.resize(
                    ((self.max_width + self.ksize.width - 1) * ch).div_ceil(T::lanes::<S>()),
                    Zeroable::zeroed(),
                );
                let mut n = self.const_border_value.len();
                let n1 = (self.max_width + self.ksize.width - 1) * ch;
                let const_val = &self.const_border_value;
                let dst = cast_slice_mut(&mut self.const_border_row);
                let t_dst = if is_sep {
                    cast_slice_mut::<_, T>(&mut self.src_row)
                } else {
                    alias_slice_mut(dst)
                };

                for i in (0..n1).step_by(n) {
                    n = n.min(n1 - i);
                    t_dst[i..i + n].copy_from_slice(&const_val[..n]);
                }

                if let Filter::Separable { row_filter, .. } = &self.filter {
                    row_filter.apply::<S>(cast_slice(&self.src_row), dst, self.max_width, ch);
                }
            }

            let max_buf_step =
                (self.max_width + k_offs).next_multiple_of(align_of::<Vector<S, T>>()) * ch;

            self.resize_ring_buf(max_buf_step * self.rows.len());
        }

        let const_val = &self.const_border_value;

        self.buf_step = (width + k_offs).next_multiple_of(align_of::<Vector<S, T>>()) * ch;

        self.dx1 = self.anchor.x;
        self.dx2 = self.ksize.width - self.anchor.x - 1;

        if self.dx1 > 0 || self.dx2 > 0 {
            if matches!(self.border_type, BorderType::Constant) {
                let nr = if self.is_separable() {
                    1
                } else {
                    self.rows.len()
                };
                for i in 0..nr {
                    let dst = if self.is_separable() {
                        cast_slice_mut::<_, T>(&mut self.src_row)
                    } else {
                        &mut cast_slice_mut::<_, T>(&mut self.ring_buf)[self.buf_step * i..]
                    };
                    memcpy(dst, const_val, self.dx1 * ch);
                    let right = (width + self.ksize.width - 1 - self.dx2) * ch;
                    memcpy(&mut dst[right..], const_val, self.dx2 * ch);
                }
            } else {
                for i in 0..self.dx1 as isize {
                    let p0 = border_interpolate(i - self.dx1 as isize, width, self.border_type);
                    let p0 = p0 as usize * ch;
                    for j in 0..ch {
                        self.border_table[i as usize * ch + j] = p0 + j;
                    }
                }
                for i in 0..self.dx2 {
                    let p0 = border_interpolate((width + i) as isize, width, self.border_type)
                        as usize
                        * ch;
                    for j in 0..ch {
                        self.border_table[(i + self.dx1) * ch + j] = p0 + j;
                    }
                }
            }
        }

        self.row_count = 0;
        self.dst_y = 0;
        self.start_y = 0;
        self.start_y_0 = 0;
        self.end_y = height;
        self.width = width;
        self.height = height;

        self.start_y
    }

    #[allow(clippy::too_many_arguments)]
    pub fn proceed(
        &mut self,

        src: &mut [T],
        src_step: usize,
        mut count: usize,
        ch: usize,
    ) -> usize {
        let buf_rows = self.rows.len();
        let kheight = self.ksize.height;
        let kwidth = self.ksize.width;
        let ay = self.anchor.y as isize;
        let dx1 = self.dx1;
        let dx2 = self.dx2;
        let width1 = self.width + kwidth - 1;
        let btab = &self.border_table;
        let make_border = (dx1 > 0 || dx2 > 0) && !matches!(self.border_type, BorderType::Constant);
        let is_sep = self.is_separable();

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
                let row = if is_sep {
                    &mut src_row[..]
                } else {
                    alias_slice_mut(brow)
                };

                if self.row_count + 1 > buf_rows {
                    self.row_count -= 1;
                    self.start_y += 1;
                }
                self.row_count += 1;

                memcpy(
                    &mut row[dx1 * ch..],
                    &src[src_off..],
                    (width1 - dx2 - dx1) * ch,
                );

                if make_border {
                    for i in 0..dx1 * ch {
                        row[i] = src[src_off + btab[i]];
                    }
                    for i in 0..dx2 * ch {
                        row[i + (width1 - dx2) * ch] = src[src_off + btab[i + dx1 * ch]];
                    }
                }

                if let Filter::Separable { row_filter, .. } = &self.filter {
                    row_filter.apply::<S>(row, brow, self.width, ch);
                }

                dcount -= 1;
                src_off += src_step;
            }

            let max_i = buf_rows.min(self.height - (self.dst_y + dy) + (kheight - 1));
            i = 0;
            while i < max_i {
                let src_y = border_interpolate(
                    (self.dst_y + dy + i) as isize - ay,
                    self.height,
                    self.border_type,
                );
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
            match &mut self.filter {
                Filter::Separable { col_filter, .. } => {
                    col_filter.apply::<S>(brows, &mut src[dst_off..], src_step, i, self.width * ch)
                }
                Filter::Fallback(filter) => {
                    filter.apply::<S>(brows, &mut src[dst_off..], src_step, i, self.width, ch)
                }
            }

            dst_off += src_step * i;
            dy += i;
        }

        self.dst_y += dy;
        dy
    }

    fn remaining_input_rows(&self) -> usize {
        self.end_y - self.start_y - self.row_count
    }
}

#[track_caller]
fn memcpy<T: Copy>(to: &mut [T], from: &[T], len: usize) {
    to[..len].copy_from_slice(&from[..len]);
}

/// Unsafely alias slice. Needed for the conditional slice targets that depend on the filter. The
/// same slice shouldn't be used multiple times at once
fn alias_slice_mut<'b, T>(slice: &mut [T]) -> &'b mut [T] {
    let ptr = slice.as_mut_ptr();
    let len = slice.len();
    unsafe { core::slice::from_raw_parts_mut(ptr, len) }
}

fn border_interpolate(mut p: isize, len: usize, btype: BorderType) -> isize {
    let len = len as isize;
    if p < len && p >= 0 {
        return p;
    }
    match btype {
        BorderType::Constant => -1,
        BorderType::Replicate if p < 0 => 0,
        BorderType::Replicate => len - 1,
        BorderType::Reflect | BorderType::Reflect101 => {
            let delta = matches!(btype, BorderType::Reflect101) as isize;
            if len == 1 {
                return 0;
            }
            loop {
                if p < 0 {
                    p = -p - 1 + delta;
                } else {
                    p = len - 1 - (p - len) - delta;
                }
                if p < len && p >= 0 {
                    break;
                }
            }
            p
        }
        BorderType::Wrap => {
            if p < 0 {
                p -= ((p - len + 1) / len) * len;
            }
            if p >= len {
                p %= len;
            }
            p
        }
    }
}
