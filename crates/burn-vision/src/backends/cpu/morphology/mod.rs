use std::fmt::Debug;

use burn_tensor::{
    backend::Backend,
    cast::ToElement,
    ops::BoolTensor,
    quantization::{QuantizationScheme, QuantizationType},
    BasicOps, Bool, DType, Element, Shape, Tensor, TensorData,
};
use filter::{MaxOp, MinOp, MorphOperator, VecMorphOperator};
use filter_engine::{ColFilter, Filter, Filter2D, FilterEngine, RowFilter};
use macerator::VOrd;
use pulp::Simd;

use crate::{BorderType, MorphOptions};

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
        shape: [usize; 2],
        anchor: (usize, usize),
    },
    Other {
        kernel: Vec<B>,
        shape: [usize; 2],
        anchor: (usize, usize),
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
    let anchor = opts.anchor.unwrap_or((kh / 2, kw / 2));
    let iter = opts.iterations;
    let btype = opts.border_type;
    let bvalue = opts.border_value.map(|it| it.into_data());

    let kernel = if is_rect {
        MorphKernel::Rect {
            shape: kshape,
            anchor,
        }
    } else {
        MorphKernel::Other {
            kernel,
            shape: kshape,
            anchor,
        }
    };

    let shape = input.shape();
    let data = input.into_data();
    match data.dtype {
        DType::F64 => {
            morph_typed::<B, K, f64>(data, shape, kernel, op, iter, btype, bvalue, &device)
        }
        DType::F32 => {
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
        DType::QFloat(scheme) => match scheme {
            QuantizationScheme::PerTensorAffine(QuantizationType::QInt8) => {
                morph_typed::<B, K, i8>(data, shape, kernel, op, iter, btype, bvalue, &device)
            }
            QuantizationScheme::PerTensorSymmetric(QuantizationType::QInt8) => {
                morph_typed::<B, K, i8>(data, shape, kernel, op, iter, btype, bvalue, &device)
            }
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn morph_typed<B: Backend, K: BasicOps<B>, T: VOrd + MinMax + Element>(
    input: TensorData,
    shape: Shape,
    kernel: MorphKernel<B::BoolElem>,
    op: MorphOp,
    iter: usize,
    btype: BorderType,
    bvalue: Option<TensorData>,
    device: &B::Device,
) -> Tensor<B, 3, K> {
    let [h, w, ch] = shape.dims();
    let mut data = input.into_vec::<T>().unwrap();
    let bvalue = border_value(btype, bvalue, op, &shape);
    run_morph(&mut data, shape, kernel, op, iter, btype, &bvalue);
    let data = TensorData::new(data, Shape::new([h, w, ch]));
    Tensor::from_data(data, device)
}

#[allow(clippy::too_many_arguments)]
fn morph_bool<B: Backend, K: BasicOps<B>>(
    input: TensorData,
    shape: Shape,
    kernel: MorphKernel<B::BoolElem>,
    op: MorphOp,
    iter: usize,
    btype: BorderType,
    bvalue: Option<TensorData>,
    device: &B::Device,
) -> Tensor<B, 3, K> {
    let input = input.into_vec::<bool>().unwrap();
    let mut data = bytemuck::cast_vec::<_, u8>(input);
    let bvalue = border_value(btype, bvalue, op, &shape);
    run_morph(&mut data, shape.clone(), kernel, op, iter, btype, &bvalue);
    let [h, w, ch] = shape.dims();
    // SAFETY: Morph can't produce invalid boolean values
    let data = unsafe { core::mem::transmute::<Vec<u8>, Vec<bool>>(data) };
    let data = TensorData::new(data, Shape::new([h, w, ch]));
    Tensor::from_data(data, device)
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
        MorphKernel::Rect { shape, anchor } => {
            let row_filter = RowFilter::new(shape[1], anchor.1);
            let col_filter = ColFilter::new(shape[0], anchor.0);
            Filter::Separable {
                row_filter,
                col_filter,
            }
        }
        MorphKernel::Other {
            kernel,
            shape,
            anchor,
        } => {
            let filter = Filter2D::new(&kernel, shape, anchor);
            Filter::Fallback(filter)
        }
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[pulp::with_simd(dispatch_morph = pulp::Arch::new())]
fn run_morph_simd<S: Simd, T: VOrd + MinMax + Debug, Op: MorphOperator<T> + VecMorphOperator<T>>(
    simd: S,
    buffer: &mut [T],
    buffer_shape: Shape,
    filter: filter_engine::Filter<T, Op>,
    border_type: BorderType,
    border_value: &[T],
    iterations: usize,
) {
    let mut engine = FilterEngine::new(filter, border_type, border_value);
    engine.apply(simd, buffer, buffer_shape.clone());
    for _ in 1..iterations {
        engine.apply(simd, buffer, buffer_shape.clone());
    }
}
