use std::fmt::Debug;

use burn_tensor::{
    BasicOps, Bool, DType, Element, Shape, Tensor, TensorData, backend::Backend, cast::ToElement,
    ops::BoolTensor,
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

pub fn morph<B: Backend, K: BasicOps<B>>(
    input: Tensor<B, 3, K>,
    kernel: BoolTensor<B>,
    op: MorphOp,
    opts: MorphOptions<B, K>,
) -> Tensor<B, 3, K> {
    let device = input.device();

    let kernel = Tensor::<B, 2, Bool>::new(kernel);
    let kshape = kernel.shape().dims();
    let [kh, kw] = kshape;

    let kernel = kernel.into_data().into_vec::<B::BoolElem>().unwrap();
    let is_rect = kernel.iter().all(|it| it.to_bool());
    let anchor = opts.anchor.unwrap_or(Point::new(kw / 2, kh / 2));
    let iter = opts.iterations;
    let btype = opts.border_type;
    let bvalue = opts.border_value.map(|it| it.into_data());

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

    let shape = input.shape();
    let data = input.into_data();
    match data.dtype {
        DType::F64 => {
            morph_typed::<B, K, f64>(data, shape, kernel, op, iter, btype, bvalue, &device)
        }
        DType::F32 | DType::Flex32 => {
            morph_typed::<B, K, f32>(data, shape, kernel, op, iter, btype, bvalue, &device)
        }
        DType::F16 | DType::BF16 => morph_typed::<B, K, f32>(
            data.convert::<f32>(),
            shape,
            kernel,
            op,
            iter,
            btype,
            bvalue,
            &device,
        ),
        DType::I64 => {
            morph_typed::<B, K, i64>(data, shape, kernel, op, iter, btype, bvalue, &device)
        }
        DType::I32 => {
            morph_typed::<B, K, i32>(data, shape, kernel, op, iter, btype, bvalue, &device)
        }
        DType::I16 => {
            morph_typed::<B, K, i16>(data, shape, kernel, op, iter, btype, bvalue, &device)
        }
        DType::I8 => morph_typed::<B, K, i8>(data, shape, kernel, op, iter, btype, bvalue, &device),
        DType::U64 => {
            morph_typed::<B, K, u64>(data, shape, kernel, op, iter, btype, bvalue, &device)
        }
        DType::U32 => {
            morph_typed::<B, K, u32>(data, shape, kernel, op, iter, btype, bvalue, &device)
        }
        DType::U16 => {
            morph_typed::<B, K, u16>(data, shape, kernel, op, iter, btype, bvalue, &device)
        }
        DType::U8 => morph_typed::<B, K, u8>(data, shape, kernel, op, iter, btype, bvalue, &device),
        DType::Bool => morph_bool::<B, K>(data, shape, kernel, op, iter, btype, bvalue, &device),
        DType::QFloat(_) => unimplemented!(),
        DType::Complex32 | DType::Complex64 => {
            panic!("Morphological operations are not supported for complex tensors")
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn morph_typed<B: Backend, K: BasicOps<B>, T: VOrd + MinMax + Element>(
    mut input: TensorData,
    shape: Shape,
    kernel: MorphKernel<B::BoolElem>,
    op: MorphOp,
    iter: usize,
    btype: BorderType,
    bvalue: Option<TensorData>,
    device: &B::Device,
) -> Tensor<B, 3, K> {
    let data = input.as_mut_slice::<T>().unwrap();
    let bvalue = border_value(btype, bvalue, op, &shape);
    run_morph(data, shape, kernel, op, iter, btype, &bvalue);
    Tensor::from_data(input, device)
}

#[allow(clippy::too_many_arguments)]
fn morph_bool<B: Backend, K: BasicOps<B>>(
    mut input: TensorData,
    shape: Shape,
    kernel: MorphKernel<B::BoolElem>,
    op: MorphOp,
    iter: usize,
    btype: BorderType,
    bvalue: Option<TensorData>,
    device: &B::Device,
) -> Tensor<B, 3, K> {
    let data = input.as_mut_slice::<bool>().unwrap();
    // SAFETY: Morph can't produce invalid boolean values
    let data = unsafe { core::mem::transmute::<&mut [bool], &mut [u8]>(data) };
    let bvalue = border_value(btype, bvalue, op, &shape);
    run_morph(data, shape.clone(), kernel, op, iter, btype, &bvalue);
    Tensor::from_data(input, device)
}

fn border_value<T: Element>(
    btype: BorderType,
    bvalue: Option<TensorData>,
    op: MorphOp,
    shape: &Shape,
) -> Vec<T> {
    let [_, _, ch] = shape.dims();
    match (btype, bvalue) {
        (BorderType::Constant, Some(value)) => value.convert::<T>().into_vec().unwrap(),
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
pub fn create_structuring_element<B: Backend>(
    shape: KernelShape,
    ksize: Size,
    anchor: Option<Point>,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
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
