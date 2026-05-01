use std::fmt::Debug;

use burn_core::tensor::{
    Bool, BoolStore, DType, Device, Element, ElementLimits, Scalar, Shape, Tensor, TensorData,
    backend::Backend, cast::ToElement,
};
use filter::{MaxOp, MinOp, MorphOperator, VecMorphOperator};
use filter_engine::{ColFilter, Filter, Filter2D, FilterEngine, RowFilter};
use macerator::{Simd, VOrd};

use crate::{BorderType, MorphOptions, Point, Size};

use super::MinMax;

mod filter;
mod filter_engine;

/// A morphology operation.
/// TODO: Implement composite ops
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MorphOp {
    Erode,
    Dilate,
}

pub enum MorphKernel<B: Element> {
    Rect {
        size: Size,
        anchor: Point,
    },
    Other {
        kernel: Vec<B>,
        size: Size,
        anchor: Point,
    },
}

pub fn morph<B: Backend>(
    input: TensorData,
    kernel: TensorData,
    op: MorphOp,
    opts: MorphOptions,
) -> TensorData {
    let [kh, kw] = kernel.shape.dims();

    let kernel = kernel.into_vec::<B::BoolElem>().unwrap();
    let is_rect = kernel.iter().all(|it| it.to_bool());
    let anchor = opts.anchor.unwrap_or(Point::new(kw / 2, kh / 2));
    let iter = opts.iterations;
    let btype = opts.border_type;
    let bvalue = opts.border_value;

    let size = Size::new(kw, kh);
    let kernel = if is_rect {
        MorphKernel::Rect { size, anchor }
    } else {
        MorphKernel::Other {
            kernel,
            size,
            anchor,
        }
    };

    let shape = input.shape.clone();
    let data = input;
    match data.dtype {
        DType::F64 => morph_typed::<B, f64>(data, shape, kernel, op, iter, btype, bvalue),
        DType::F32 | DType::Flex32 => {
            morph_typed::<B, f32>(data, shape, kernel, op, iter, btype, bvalue)
        }
        DType::F16 | DType::BF16 => morph_typed::<B, f32>(
            data.convert::<f32>(),
            shape,
            kernel,
            op,
            iter,
            btype,
            bvalue,
        ),
        DType::I64 => morph_typed::<B, i64>(data, shape, kernel, op, iter, btype, bvalue),
        DType::I32 => morph_typed::<B, i32>(data, shape, kernel, op, iter, btype, bvalue),
        DType::I16 => morph_typed::<B, i16>(data, shape, kernel, op, iter, btype, bvalue),
        DType::I8 => morph_typed::<B, i8>(data, shape, kernel, op, iter, btype, bvalue),
        DType::U64 => morph_typed::<B, u64>(data, shape, kernel, op, iter, btype, bvalue),
        DType::U32 | DType::Bool(BoolStore::U32) => {
            morph_typed::<B, u32>(data, shape, kernel, op, iter, btype, bvalue)
        }
        DType::U16 => morph_typed::<B, u16>(data, shape, kernel, op, iter, btype, bvalue),
        DType::U8 | DType::Bool(BoolStore::U8) => {
            morph_typed::<B, u8>(data, shape, kernel, op, iter, btype, bvalue)
        }
        DType::Bool(BoolStore::Native) => {
            morph_bool::<B>(data, shape, kernel, op, iter, btype, bvalue)
        }
        DType::QFloat(_) => unimplemented!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn morph_typed<B: Backend, T: VOrd + MinMax + Element + ElementLimits>(
    mut input: TensorData,
    shape: Shape,
    kernel: MorphKernel<B::BoolElem>,
    op: MorphOp,
    iter: usize,
    btype: BorderType,
    bvalue: Option<Vec<Scalar>>,
) -> TensorData {
    let data = input.as_mut_slice::<T>().unwrap();
    let bvalue = border_value(btype, bvalue, op, &shape);
    run_morph(data, shape, kernel, op, iter, btype, &bvalue);
    input
}

#[allow(clippy::too_many_arguments)]
fn morph_bool<B: Backend>(
    mut input: TensorData,
    shape: Shape,
    kernel: MorphKernel<B::BoolElem>,
    op: MorphOp,
    iter: usize,
    btype: BorderType,
    bvalue: Option<Vec<Scalar>>,
) -> TensorData {
    let data = input.as_mut_slice::<bool>().unwrap();
    // SAFETY: Morph can't produce invalid boolean values
    let data = unsafe { core::mem::transmute::<&mut [bool], &mut [u8]>(data) };
    let bvalue = border_value(btype, bvalue, op, &shape);
    run_morph(data, shape.clone(), kernel, op, iter, btype, &bvalue);
    input
}

fn border_value<T: Element + ElementLimits>(
    btype: BorderType,
    bvalue: Option<Vec<Scalar>>,
    op: MorphOp,
    shape: &Shape,
) -> Vec<T> {
    let [_, _, ch] = shape.dims();
    match (btype, bvalue) {
        (BorderType::Constant, Some(value)) => value.into_iter().map(|v| v.elem::<T>()).collect(),
        (BorderType::Constant, None) => match op {
            MorphOp::Erode => vec![T::MAX; ch],
            MorphOp::Dilate => vec![T::MIN; ch],
        },
        _ => vec![],
    }
}

fn run_morph<T: VOrd + MinMax + Element, B: Element>(
    input: &mut [T],
    shape: Shape,
    kernel: MorphKernel<B>,
    op: MorphOp,
    iter: usize,
    btype: BorderType,
    bvalue: &[T],
) {
    match op {
        MorphOp::Erode => {
            let filter = filter::<T, MinOp, B>(kernel);
            dispatch_morph(input, shape, filter, btype, bvalue, iter);
        }
        MorphOp::Dilate => {
            let filter = filter::<T, MaxOp, B>(kernel);
            dispatch_morph(input, shape, filter, btype, bvalue, iter);
        }
    };
}

fn filter<T: VOrd + MinMax, Op: MorphOperator<T> + VecMorphOperator<T>, B: Element>(
    kernel: MorphKernel<B>,
) -> Filter<T, Op> {
    match kernel {
        MorphKernel::Rect { size, anchor } => {
            let row_filter = RowFilter::new(size.width, anchor.x);
            let col_filter = ColFilter::new(size.height, anchor.y);
            Filter::Separable {
                row_filter,
                col_filter,
            }
        }
        MorphKernel::Other {
            kernel,
            size,
            anchor,
        } => {
            let filter = Filter2D::new(&kernel, size, anchor);
            Filter::Fallback(filter)
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[macerator::with_simd]
fn dispatch_morph<
    'a,
    S: Simd,
    T: VOrd + MinMax + Debug,
    Op: MorphOperator<T> + VecMorphOperator<T>,
>(
    buffer: &'a mut [T],
    buffer_shape: Shape,
    filter: filter_engine::Filter<T, Op>,
    border_type: BorderType,
    border_value: &'a [T],
    iterations: usize,
) where
    'a: 'a,
{
    let [_, _, ch] = buffer_shape.dims();
    let mut engine = FilterEngine::<S, _, _>::new(filter, border_type, border_value, ch);
    engine.apply(buffer, buffer_shape.clone());
    for _ in 1..iterations {
        engine.apply(buffer, buffer_shape.clone());
    }
}

/// Shape of the structuring element
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelShape {
    /// Rectangular kernel
    Rect,
    /// Cross shaped kernel
    Cross,
    /// Ellipse shaped kernel
    Ellipse,
}

/// Create a structuring element tensor for use with morphology ops
pub fn create_structuring_element(
    shape: KernelShape,
    ksize: Size,
    anchor: Option<Point>,
    device: &Device,
) -> Tensor<2, Bool> {
    fn create_kernel(shape: KernelShape, ksize: Size, anchor: Option<Point>) -> Vec<bool> {
        let anchor = anchor.unwrap_or(Point::new(ksize.width / 2, ksize.height / 2));
        let mut r = 0;
        let mut c = 0;
        let mut inv_r2 = 0.0;

        if (ksize.width == 1 && ksize.height == 1) || shape == KernelShape::Rect {
            return vec![true; ksize.height * ksize.width];
        }

        if shape == KernelShape::Ellipse {
            r = ksize.height / 2;
            c = ksize.width / 2;
            inv_r2 = if r > 0 { 1.0 / (r * r) as f64 } else { 0.0 }
        }

        let mut elem = vec![false; ksize.height * ksize.width];

        for i in 0..ksize.height {
            let mut j1 = 0;
            let mut j2 = 0;
            if shape == KernelShape::Cross && i == anchor.y {
                j2 = ksize.width;
            } else if shape == KernelShape::Cross {
                j1 = anchor.x;
                j2 = j1 + 1;
            } else {
                let dy = i as isize - r as isize;
                if dy.abs() <= r as isize {
                    let dx = (c as f64 * ((r * r - (dy * dy) as usize) as f64 * inv_r2).sqrt())
                        .round() as isize;
                    j1 = (c as isize - dx).max(0) as usize;
                    j2 = (c + dx as usize + 1).min(ksize.width);
                }
            }

            for j in j1..j2 {
                elem[i * ksize.width + j] = true;
            }
        }
        elem
    }

    let elem = create_kernel(shape, ksize, anchor);

    let data = TensorData::new(elem, [ksize.height, ksize.width]);
    Tensor::from_data(data, device)
}
