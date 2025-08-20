use core::slice;
use std::{marker::PhantomData, ptr::null};

use burn_tensor::Element;
use macerator::{
    Scalar, Simd, VOrd, Vector, vload, vload_low, vload_unaligned, vstore, vstore_low,
    vstore_unaligned,
};

use crate::{Point, Size, backends::cpu::MinMax};

pub trait MorphOperator<T> {
    fn apply(a: T, b: T) -> T;
}

pub trait VecMorphOperator<T: Scalar> {
    fn apply<S: Simd>(a: Vector<S, T>, b: Vector<S, T>) -> Vector<S, T>;
}

pub struct MinOp;
pub struct MaxOp;

impl<T: MinMax> MorphOperator<T> for MinOp {
    fn apply(a: T, b: T) -> T {
        MinMax::min(a, b)
    }
}

impl<T: VOrd> VecMorphOperator<T> for MinOp {
    fn apply<S: Simd>(a: Vector<S, T>, b: Vector<S, T>) -> Vector<S, T> {
        T::vmin(a, b)
    }
}

impl<T: MinMax> MorphOperator<T> for MaxOp {
    fn apply(a: T, b: T) -> T {
        MinMax::max(a, b)
    }
}

impl<T: VOrd> VecMorphOperator<T> for MaxOp {
    fn apply<S: Simd>(a: Vector<S, T>, b: Vector<S, T>) -> Vector<S, T> {
        T::vmax(a, b)
    }
}

pub struct MorphRowFilter<T: Scalar, S: MorphOperator<T>, Vec: VecRow<T>> {
    pub ksize: usize,
    pub anchor: usize,
    vec: Vec,
    _t: PhantomData<T>,
    _scalar: PhantomData<S>,
}

impl<T: Scalar, SOp: MorphOperator<T>, Vec: VecRow<T>> MorphRowFilter<T, SOp, Vec> {
    pub fn new(ksize: usize, anchor: usize) -> Self {
        let vec = Vec::new(ksize, anchor);
        Self {
            ksize,
            anchor,
            vec,
            _t: PhantomData,
            _scalar: PhantomData,
        }
    }

    pub fn apply<S: Simd>(&self, src: &[T], dst: &mut [T], width: usize, ch: usize) {
        let k_size = self.ksize * ch;

        if k_size == ch {
            let width = width * ch;
            dst[..width].copy_from_slice(&src[..width]);
            return;
        }

        let i0 = self.vec.apply::<S>(src, dst, width, ch);
        let width = width * ch;

        for k in 0..ch {
            let mut last_i = i0;
            for i in (i0..width.saturating_sub(ch * 2)).step_by(ch * 2) {
                let mut m = src[k + i + ch];
                let mut last_j = ch * 2;
                for j in (ch * 2..k_size).step_by(ch) {
                    m = SOp::apply(m, src[k + i + j]);
                    last_j = j + ch;
                }
                dst[k + i] = SOp::apply(m, src[k + i]);
                dst[k + i + ch] = SOp::apply(m, src[k + i + last_j]);
                last_i = i + ch * 2;
            }

            for i in (last_i..width).step_by(ch) {
                let mut m = src[k + i];
                for j in (ch..k_size).step_by(ch) {
                    m = SOp::apply(m, src[k + i + j]);
                }
                dst[k + i] = m;
            }
        }
    }
}

pub struct MorphRowVec<T: Scalar, Op: VecMorphOperator<T>> {
    k_size: usize,
    _t: PhantomData<T>,
    _op: PhantomData<Op>,
}

pub trait VecRow<T: Scalar> {
    fn new(ksize: usize, anchor: usize) -> Self;
    fn apply<S: Simd>(&self, src: &[T], dst: &mut [T], width: usize, channels: usize) -> usize;
}

impl<T: Scalar, Op: VecMorphOperator<T>> VecRow<T> for MorphRowVec<T, Op> {
    fn apply<S: Simd>(&self, src: &[T], dst: &mut [T], width: usize, ch: usize) -> usize {
        let src = src.as_ptr();
        let dst = dst.as_mut_ptr();
        let k_size = self.k_size * ch;
        let width = (width * ch) as isize;
        let lanes = T::lanes::<S>();

        // Safety: everything here is unsafe. Test thoroughly.
        unsafe {
            let mut x = 0;
            while x as isize <= width - 4 * lanes as isize {
                let mut s0 = vload(src.add(x));
                let mut s1 = vload(src.add(x + lanes));
                let mut s2 = vload(src.add(x + 2 * lanes));
                let mut s3 = vload(src.add(x + 3 * lanes));
                for k in (ch..k_size).step_by(ch) {
                    let x = x + k;
                    s0 = Op::apply::<S>(s0, vload_unaligned(src.add(x)));
                    s1 = Op::apply::<S>(s1, vload_unaligned(src.add(x + lanes)));
                    s2 = Op::apply::<S>(s2, vload_unaligned(src.add(x + 2 * lanes)));
                    s3 = Op::apply::<S>(s3, vload_unaligned(src.add(x + 3 * lanes)));
                }
                vstore(dst.add(x), s0);
                vstore(dst.add(x + lanes), s1);
                vstore(dst.add(x + 2 * lanes), s2);
                vstore(dst.add(x + 3 * lanes), s3);
                x += 4 * lanes;
            }
            if x as isize <= width - 2 * lanes as isize {
                let mut s0 = vload(src.add(x));
                let mut s1 = vload(src.add(x + lanes));
                for k in (ch..k_size).step_by(ch) {
                    s0 = Op::apply::<S>(s0, vload_unaligned(src.add(x + k)));
                    s1 = Op::apply::<S>(s1, vload_unaligned(src.add(x + k + lanes)));
                }
                vstore(dst.add(x), s0);
                vstore(dst.add(x + lanes), s1);
                x += 2 * lanes;
            }
            if x as isize <= width - lanes as isize {
                let mut s = vload(src.add(x));
                for k in (ch..k_size).step_by(ch) {
                    s = Op::apply::<S>(s, vload_unaligned(src.add(x + k)));
                }
                vstore(dst.add(x), s);
                x += lanes;
            }
            if x as isize <= width - lanes as isize / 2 {
                let mut s = vload_low(src.add(x));
                for k in (ch..k_size).step_by(ch) {
                    s = Op::apply::<S>(s, vload_low(src.add(x + k)));
                }
                vstore_low(dst.add(x), s);
                x += lanes / 2;
            }
            x - x % ch
        }
    }

    fn new(k_size: usize, _anchor: usize) -> Self {
        Self {
            k_size,
            _t: PhantomData,
            _op: PhantomData,
        }
    }
}

pub trait VecColumn<T: Scalar> {
    fn new(ksize: usize, anchor: usize) -> Self;
    fn apply<S: Simd>(
        &self,

        src: &[*const T],
        dst: &mut [T],
        dst_step: usize,
        height: usize,
        width: usize,
    ) -> usize;
}

pub struct MorphColumnVec<T: Scalar, Op: VecMorphOperator<T>> {
    k_size: usize,
    _t: PhantomData<T>,
    _op: PhantomData<Op>,
}

impl<T: VOrd, Op: VecMorphOperator<T>> VecColumn<T> for MorphColumnVec<T, Op> {
    fn new(k_size: usize, _anchor: usize) -> Self {
        Self {
            k_size,
            _t: PhantomData,
            _op: PhantomData,
        }
    }

    fn apply<S: Simd>(
        &self,

        src: &[*const T],
        dst: &mut [T],
        dst_step: usize,
        mut count: usize,
        width: usize,
    ) -> usize {
        let ksize = self.k_size;
        let width = width as isize;
        let mut dst = dst.as_mut_ptr();
        let lanes = T::lanes::<S>();
        let mut y = 0;
        let mut x = 0;

        // Safety: everything here is unsafe. Test thoroughly.
        unsafe {
            while count > 1 && ksize > 1 {
                x = 0;
                while x as isize <= width - 4 * lanes as isize {
                    let sptr = src[y + 1].add(x);
                    let mut s0 = vload(sptr);
                    let mut s1 = vload(sptr.add(lanes));
                    let mut s2 = vload(sptr.add(2 * lanes));
                    let mut s3 = vload(sptr.add(3 * lanes));

                    for k in 2..ksize {
                        let sptr = src[y + k].add(x);
                        s0 = Op::apply::<S>(s0, vload(sptr));
                        s1 = Op::apply::<S>(s1, vload(sptr.add(lanes)));
                        s2 = Op::apply::<S>(s2, vload(sptr.add(2 * lanes)));
                        s3 = Op::apply::<S>(s3, vload(sptr.add(3 * lanes)));
                    }

                    // Row 1
                    {
                        let sptr = src[y].add(x);
                        let s0 = Op::apply(s0, vload(sptr));
                        let s1 = Op::apply(s1, vload(sptr.add(lanes)));
                        let s2 = Op::apply(s2, vload(sptr.add(2 * lanes)));
                        let s3 = Op::apply(s3, vload(sptr.add(3 * lanes)));
                        vstore_unaligned(dst.add(x), s0);
                        vstore_unaligned(dst.add(x + lanes), s1);
                        vstore_unaligned(dst.add(x + 2 * lanes), s2);
                        vstore_unaligned(dst.add(x + 3 * lanes), s3);
                    }

                    // Row 2
                    {
                        let sptr = src[y + ksize].add(x);
                        let s0 = Op::apply(s0, vload(sptr));
                        let s1 = Op::apply(s1, vload(sptr.add(lanes)));
                        let s2 = Op::apply(s2, vload(sptr.add(2 * lanes)));
                        let s3 = Op::apply(s3, vload(sptr.add(3 * lanes)));
                        vstore_unaligned(dst.add(dst_step + x), s0);
                        vstore_unaligned(dst.add(dst_step + x + lanes), s1);
                        vstore_unaligned(dst.add(dst_step + x + 2 * lanes), s2);
                        vstore_unaligned(dst.add(dst_step + x + 3 * lanes), s3);
                    }
                    x += 4 * lanes;
                }
                if x as isize <= width - 2 * lanes as isize {
                    let sptr = src[y + 1].add(x);
                    let mut s0 = vload(sptr);
                    let mut s1 = vload(sptr.add(lanes));

                    for k in 2..ksize {
                        let sptr = src[y + k].add(x);
                        s0 = Op::apply::<S>(s0, vload(sptr));
                        s1 = Op::apply::<S>(s1, vload(sptr.add(lanes)));
                    }

                    // Row 1
                    {
                        let sptr = src[y].add(x);
                        let s0 = Op::apply(s0, vload(sptr));
                        let s1 = Op::apply(s1, vload(sptr.add(lanes)));
                        vstore_unaligned(dst.add(x), s0);
                        vstore_unaligned(dst.add(x + lanes), s1);
                    }

                    // Row 2
                    {
                        let sptr = src[y + ksize].add(x);
                        let s0 = Op::apply(s0, vload(sptr));
                        let s1 = Op::apply(s1, vload(sptr.add(lanes)));
                        vstore_unaligned(dst.add(dst_step + x), s0);
                        vstore_unaligned(dst.add(dst_step + x + lanes), s1);
                    }
                    x += 2 * lanes;
                }
                if x as isize <= width - lanes as isize {
                    let mut s0 = vload(src[y + 1].add(x));
                    for k in 2..ksize {
                        s0 = Op::apply::<S>(s0, vload(src[y + k].add(x)));
                    }
                    // Row 1
                    {
                        let sptr = src[y].add(x);
                        vstore_unaligned(dst.add(x), Op::apply(s0, vload(sptr)));
                    }

                    // Row 2
                    {
                        let sptr = src[y + ksize].add(x);
                        let s0 = Op::apply(s0, vload(sptr));
                        vstore_unaligned(dst.add(dst_step + x), s0);
                    }
                    x += lanes;
                }
                if x as isize <= width - lanes as isize / 2 {
                    let mut s0 = vload_low(src[y + 1].add(x));
                    for k in 2..ksize {
                        s0 = Op::apply::<S>(s0, vload_low(src[y + k].add(x)));
                    }
                    // Row 1
                    {
                        let sptr = src[y].add(x);
                        let s0 = Op::apply(s0, vload_low(sptr));
                        vstore_low(dst.add(x), s0);
                    }

                    // Row 2
                    {
                        let sptr = src[y + ksize].add(x);
                        let s0 = Op::apply(s0, vload_low(sptr));
                        vstore_low(dst.add(dst_step + x), s0);
                    }
                    x += lanes / 2;
                }

                count -= 2;
                dst = dst.add(dst_step * 2);
                y += 2;
            }

            while count > 0 {
                x = 0;
                while x as isize <= width - 4 * lanes as isize {
                    let sptr = src[y].add(x);
                    let mut s0 = vload(sptr);
                    let mut s1 = vload(sptr.add(lanes));
                    let mut s2 = vload(sptr.add(2 * lanes));
                    let mut s3 = vload(sptr.add(3 * lanes));

                    for k in 1..ksize {
                        let sptr = src[y + k].add(x);
                        s0 = Op::apply::<S>(s0, vload(sptr));
                        s1 = Op::apply::<S>(s1, vload(sptr.add(lanes)));
                        s2 = Op::apply::<S>(s2, vload(sptr.add(2 * lanes)));
                        s3 = Op::apply::<S>(s3, vload(sptr.add(3 * lanes)));
                    }

                    vstore_unaligned(dst.add(x), s0);
                    vstore_unaligned(dst.add(x + lanes), s1);
                    vstore_unaligned(dst.add(x + 2 * lanes), s2);
                    vstore_unaligned(dst.add(x + 3 * lanes), s3);

                    x += 4 * lanes;
                }
                if x as isize <= width - 2 * lanes as isize {
                    let sptr = src[y].add(x);
                    let mut s0 = vload(sptr);
                    let mut s1 = vload(sptr.add(lanes));

                    for k in 1..ksize {
                        let sptr = src[y + k].add(x);
                        s0 = Op::apply::<S>(s0, vload(sptr));
                        s1 = Op::apply::<S>(s1, vload(sptr.add(lanes)));
                    }

                    vstore_unaligned(dst.add(x), s0);
                    vstore_unaligned(dst.add(x + lanes), s1);
                    x += 2 * lanes;
                }
                if x as isize <= width - lanes as isize {
                    let mut s0 = vload(src[y].add(x));

                    for k in 1..ksize {
                        s0 = Op::apply::<S>(s0, vload(src[y + k].add(x)));
                    }

                    vstore_unaligned(dst.add(x), s0);
                    x += lanes;
                }
                if x as isize <= width - lanes as isize / 2 {
                    let mut s0 = vload_low(src[y].add(x));

                    for k in 1..ksize {
                        s0 = Op::apply::<S>(s0, vload_low(src[y + k].add(x)));
                    }

                    vstore_low(dst.add(x), s0);
                    x += lanes / 2;
                }

                count -= 1;
                dst = dst.add(dst_step);
                y += 1;
            }
        }
        x
    }
}

pub struct MorphColumnFilter<T: Scalar, Op: MorphOperator<T>, VecOp: VecColumn<T>> {
    pub ksize: usize,
    pub anchor: usize,
    vec: VecOp,
    _t: PhantomData<T>,
    _op: PhantomData<Op>,
}

impl<T: Scalar, Op: MorphOperator<T>, VecOp: VecColumn<T>> MorphColumnFilter<T, Op, VecOp> {
    pub fn new(ksize: usize, anchor: usize) -> Self {
        let vec = VecOp::new(ksize, anchor);
        Self {
            ksize,
            anchor,
            vec,
            _t: PhantomData,
            _op: PhantomData,
        }
    }

    pub fn apply<S: Simd>(
        &self,

        src: &[*const T],
        dst: &mut [T],
        dst_step: usize,
        mut count: usize,
        width: usize,
    ) {
        let ksize = self.ksize;
        let x0 = self.vec.apply::<S>(src, dst, dst_step, count, width);
        let width = width as isize;

        let mut d = 0;
        let mut x;
        let mut y = 0;

        let slice = |row: *const T| unsafe { slice::from_raw_parts(row, width as usize) };

        while ksize > 1 && count > 1 {
            x = x0;

            while x as isize <= width - 4 {
                let row = slice(src[y + 1]);
                let mut s0 = row[x];
                let mut s1 = row[x + 1];
                let mut s2 = row[x + 2];
                let mut s3 = row[x + 3];

                for k in 2..ksize {
                    let row = slice(src[y + k]);
                    s0 = Op::apply(s0, row[x]);
                    s1 = Op::apply(s1, row[x + 1]);
                    s2 = Op::apply(s2, row[x + 2]);
                    s3 = Op::apply(s3, row[x + 3]);
                }

                let row = slice(src[y]);
                dst[d + x] = Op::apply(s0, row[x]);
                dst[d + x + 1] = Op::apply(s1, row[x + 1]);
                dst[d + x + 2] = Op::apply(s2, row[x + 2]);
                dst[d + x + 3] = Op::apply(s3, row[x + 3]);

                let row = slice(src[y + ksize]);
                dst[d + dst_step + x] = Op::apply(s0, row[x]);
                dst[d + dst_step + x + 1] = Op::apply(s1, row[x + 1]);
                dst[d + dst_step + x + 2] = Op::apply(s2, row[x + 2]);
                dst[d + dst_step + x + 3] = Op::apply(s3, row[x + 3]);

                x += 4;
            }
            while (x as isize) < width {
                let mut s0 = slice(src[y + 1])[x];
                for k in 2..ksize {
                    s0 = Op::apply(s0, slice(src[y + k])[x]);
                }
                dst[d + x] = Op::apply(s0, slice(src[y])[x]);
                dst[d + dst_step + x] = Op::apply(s0, slice(src[y + ksize])[x]);

                x += 1;
            }

            count -= 2;
            d += 2 * dst_step;
            y += 2;
        }

        while count > 0 {
            x = x0;

            while x as isize <= width - 4 {
                let row = slice(src[y]);
                let mut s0 = row[x];
                let mut s1 = row[x + 1];
                let mut s2 = row[x + 2];
                let mut s3 = row[x + 3];

                for k in 1..ksize {
                    let row = slice(src[y + k]);
                    s0 = Op::apply(s0, row[x]);
                    s1 = Op::apply(s1, row[x + 1]);
                    s2 = Op::apply(s2, row[x + 2]);
                    s3 = Op::apply(s3, row[x + 3]);
                }

                dst[d + x] = s0;
                dst[d + x + 1] = s1;
                dst[d + x + 2] = s2;
                dst[d + x + 3] = s3;

                x += 4;
            }
            while (x as isize) < width {
                let mut s0 = slice(src[y])[x];
                for k in 1..ksize {
                    s0 = Op::apply(s0, slice(src[y + k])[x]);
                }

                dst[d + x] = s0;

                x += 1;
            }

            count -= 1;
            d += dst_step;
            y += 1;
        }
    }
}

pub trait VecFilter<T: Scalar> {
    fn apply<S: Simd>(src: &[*const T], nz: usize, dst: &mut [T], width: usize) -> usize;
}

pub struct MorphVec<T: Scalar, Op: VecMorphOperator<T>>(PhantomData<(T, Op)>);

impl<T: Scalar, Op: VecMorphOperator<T>> VecFilter<T> for MorphVec<T, Op> {
    fn apply<S: Simd>(src: &[*const T], nz: usize, dst: &mut [T], width: usize) -> usize {
        let dst = dst.as_mut_ptr();
        let mut i = 0;
        let lanes = T::lanes::<S>();
        let width = width as isize;

        // Safety: everything here is unsafe. Test thoroughly.
        unsafe {
            while i as isize <= width - 4 * lanes as isize {
                let sptr = src[0].add(i);
                let mut s0 = vload_unaligned(sptr);
                let mut s1 = vload_unaligned(sptr.add(lanes));
                let mut s2 = vload_unaligned(sptr.add(2 * lanes));
                let mut s3 = vload_unaligned(sptr.add(3 * lanes));
                for sptr in src[1..nz].iter().map(|sptr| sptr.add(i)) {
                    s0 = Op::apply::<S>(s0, vload_unaligned(sptr));
                    s1 = Op::apply::<S>(s1, vload_unaligned(sptr.add(lanes)));
                    s2 = Op::apply::<S>(s2, vload_unaligned(sptr.add(2 * lanes)));
                    s3 = Op::apply::<S>(s3, vload_unaligned(sptr.add(3 * lanes)));
                }
                vstore_unaligned(dst.add(i), s0);
                vstore_unaligned(dst.add(i + lanes), s1);
                vstore_unaligned(dst.add(i + 2 * lanes), s2);
                vstore_unaligned(dst.add(i + 3 * lanes), s3);
                i += 4 * lanes;
            }
            if i as isize <= width - 2 * lanes as isize {
                let sptr = src[0].add(i);
                let mut s0 = vload_unaligned(sptr);
                let mut s1 = vload_unaligned(sptr.add(lanes));
                for sptr in src[1..nz].iter().map(|sptr| sptr.add(i)) {
                    s0 = Op::apply::<S>(s0, vload_unaligned(sptr));
                    s1 = Op::apply::<S>(s1, vload_unaligned(sptr.add(lanes)));
                }
                vstore_unaligned(dst.add(i), s0);
                vstore_unaligned(dst.add(i + lanes), s1);
                i += 2 * lanes;
            }
            if i as isize <= width - lanes as isize {
                let mut s0 = vload_unaligned(src[0].add(i));
                for sptr in src[1..nz].iter().map(|sptr| sptr.add(i)) {
                    s0 = Op::apply::<S>(s0, vload_unaligned(sptr));
                }
                vstore_unaligned(dst.add(i), s0);
                i += lanes;
            }
            if i as isize <= width - lanes as isize / 2 {
                let mut s = vload_low(src[0].add(i));
                for sptr in src[1..nz].iter().map(|sptr| sptr.add(i)) {
                    s = Op::apply::<S>(s, vload_low(sptr));
                }
                vstore_low(dst.add(i), s);
                i += lanes / 2;
            }
        }
        i
    }
}

pub struct MorphFilter<T: Scalar, Op: MorphOperator<T>, VecOp: VecFilter<T>> {
    pub ksize: Size,
    pub anchor: Point,
    coords: Vec<Point>,
    ptrs: Vec<*const T>,
    _op: PhantomData<(Op, VecOp)>,
}

impl<T: Scalar, Op: MorphOperator<T>, VecOp: VecFilter<T>> MorphFilter<T, Op, VecOp> {
    pub fn new<B: Element>(kernel: &[B], ksize: Size, anchor: Point) -> Self {
        let coords = process_2d_kernel(kernel, ksize);
        let ptrs = vec![null(); coords.len()];

        Self {
            ksize,
            anchor,
            coords,
            ptrs,
            _op: PhantomData,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply<S: Simd>(
        &mut self,

        src: &[*const T],
        dst: &mut [T],
        dst_step: usize,
        mut count: usize,
        width: usize,
        ch: usize,
    ) {
        let nz = self.coords.len();
        let width = (width * ch) as isize;
        let pt = &self.coords;
        let kp = &mut self.ptrs;

        let mut dst_off = 0;
        let mut src_off = 0;

        let slice = |ptr: *const T| unsafe { slice::from_raw_parts(ptr, width as usize) };

        unsafe {
            while count > 0 {
                for k in 0..nz {
                    kp[k] = src[src_off + pt[k].y].add(pt[k].x * ch);
                }

                let mut i = VecOp::apply::<S>(kp, nz, &mut dst[dst_off..], width as usize);
                while i as isize <= width - 4 {
                    let sptr = slice(kp[0].add(i));
                    let mut s0 = sptr[0];
                    let mut s1 = sptr[1];
                    let mut s2 = sptr[2];
                    let mut s3 = sptr[3];

                    for sptr in kp[1..nz].iter().map(|sptr| slice(sptr.add(i))) {
                        s0 = Op::apply(s0, sptr[0]);
                        s1 = Op::apply(s1, sptr[1]);
                        s2 = Op::apply(s2, sptr[2]);
                        s3 = Op::apply(s3, sptr[3]);
                    }

                    dst[dst_off + i] = s0;
                    dst[dst_off + i + 1] = s1;
                    dst[dst_off + i + 2] = s2;
                    dst[dst_off + i + 3] = s3;
                    i += 4;
                }
                for i in i..width as usize {
                    let mut s0 = *kp[0].add(i);
                    for v in kp[1..nz].iter().map(|sptr| *sptr.add(i)) {
                        s0 = Op::apply(s0, v);
                    }
                    dst[dst_off + i] = s0;
                }

                count -= 1;
                dst_off += dst_step;
                src_off += 1;
            }
        }
    }
}

fn process_2d_kernel<B: Element>(kernel: &[B], ksize: Size) -> Vec<Point> {
    let Size { width, height } = ksize;

    let mut nz = kernel.iter().filter(|it| it.to_bool()).count();
    if nz == 0 {
        nz = 1;
    }

    let mut coords = vec![Point::new(0, 0); nz];
    let mut k = 0;

    for y in 0..height {
        let krow = &kernel[y * width..];
        for (x, _) in krow[..width].iter().enumerate().filter(|it| it.1.to_bool()) {
            coords[k] = Point::new(x, y);
            k += 1;
        }
    }

    coords
}
