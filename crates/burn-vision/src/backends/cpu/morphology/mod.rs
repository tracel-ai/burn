use std::fmt::Debug;

use burn_tensor::{
    backend::Backend,
    cast::ToElement,
    ops::BoolTensor,
    quantization::{QuantizationScheme, QuantizationType},
    BasicOps, Bool, DType, Element, Shape, Tensor, TensorData,
};
use filter::{MorphOperator, VecMorphOperator};
use filter_engine::{DilateCol, DilateRow, ErodeCol, ErodeRow, FilterEngine};
use macerator::VOrd;
use ndarray::Array2;
use pulp::Simd;

use super::MinMax;

mod filter;
mod filter_engine;

/// A morphology operation.
/// TODO: Implement composite ops
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
        kernel: Array2<B>,
        anchor: (usize, usize),
    },
}

impl<B: Element> MorphKernel<B> {
    pub fn ksize(&self) -> (usize, usize) {
        match self {
            MorphKernel::Rect { shape, .. } => (shape[0], shape[1]),
            MorphKernel::Other { kernel, .. } => kernel.dim(),
        }
    }

    pub fn anchor(&self) -> (usize, usize) {
        match self {
            MorphKernel::Rect { anchor, .. } => *anchor,
            MorphKernel::Other { anchor, .. } => *anchor,
        }
    }
}

pub fn morph<B: Backend, K: BasicOps<B>>(
    input: Tensor<B, 3, K>,
    kernel: BoolTensor<B>,
    op: MorphOp,
) -> Tensor<B, 3, K> {
    let device = input.device();

    let kernel = Tensor::<B, 2, Bool>::new(kernel);
    let k_shape = kernel.shape().dims();
    let [kh, kw] = k_shape;

    let data = kernel.into_data().into_vec::<B::BoolElem>().unwrap();
    let kernel = unsafe { Array2::from_shape_vec_unchecked(k_shape, data) };
    let is_rect = kernel.iter().all(|it| it.to_bool());
    let anchor_x = kw / 2;
    let anchor_y = kh / 2;

    let kernel = if is_rect {
        MorphKernel::Rect {
            shape: k_shape,
            anchor: (anchor_x, anchor_y),
        }
    } else {
        MorphKernel::Other {
            kernel,
            anchor: (anchor_x, anchor_y),
        }
    };

    let shape = input.shape();
    let data = input.into_data();
    match data.dtype {
        DType::F64 => morph_typed::<B, K, f64>(data, shape, kernel, op, &device),
        DType::F32 => morph_typed::<B, K, f32>(data, shape, kernel, op, &device),
        DType::F16 | DType::BF16 => {
            morph_typed::<B, K, f32>(data.convert::<f32>(), shape, kernel, op, &device)
        }
        DType::I64 => morph_typed::<B, K, i64>(data, shape, kernel, op, &device),
        DType::I32 => morph_typed::<B, K, i32>(data, shape, kernel, op, &device),
        DType::I16 => morph_typed::<B, K, i16>(data, shape, kernel, op, &device),
        DType::I8 => morph_typed::<B, K, i8>(data, shape, kernel, op, &device),
        DType::U64 => morph_typed::<B, K, u64>(data, shape, kernel, op, &device),
        DType::U32 => morph_typed::<B, K, u32>(data, shape, kernel, op, &device),
        DType::U16 => morph_typed::<B, K, u16>(data, shape, kernel, op, &device),
        DType::U8 => morph_typed::<B, K, u8>(data, shape, kernel, op, &device),
        DType::Bool => morph_bool::<B, K>(data, shape, kernel, op, &device),
        DType::QFloat(scheme) => match scheme {
            QuantizationScheme::PerTensorAffine(QuantizationType::QInt8) => {
                morph_typed::<B, K, i8>(data, shape, kernel, op, &device)
            }
            QuantizationScheme::PerTensorSymmetric(QuantizationType::QInt8) => {
                morph_typed::<B, K, i8>(data, shape, kernel, op, &device)
            }
        },
    }
}

fn morph_typed<B: Backend, K: BasicOps<B>, T: VOrd + MinMax + Element>(
    input: TensorData,
    shape: Shape,
    kernel: MorphKernel<B::BoolElem>,
    op: MorphOp,
    device: &B::Device,
) -> Tensor<B, 3, K> {
    let [h, w, ch] = shape.dims();
    let mut data = input.into_vec::<T>().unwrap();
    run_morph(&mut data, shape, kernel, op);
    let data = TensorData::new(data, Shape::new([h, w, ch]));
    Tensor::from_data(data, device)
}

fn morph_bool<B: Backend, K: BasicOps<B>>(
    input: TensorData,
    shape: Shape,
    kernel: MorphKernel<B::BoolElem>,
    op: MorphOp,
    device: &B::Device,
) -> Tensor<B, 3, K> {
    let input = input.into_vec::<bool>().unwrap();
    let mut data = bytemuck::cast_vec::<_, u8>(input);
    run_morph(&mut data, shape.clone(), kernel, op);
    let [h, w, ch] = shape.dims();
    // SAFETY: Morph can't produce invalid boolean values
    let data = unsafe { core::mem::transmute::<Vec<u8>, Vec<bool>>(data) };
    let data = TensorData::new(data, Shape::new([h, w, ch]));
    Tensor::from_data(data, device)
}

fn run_morph<T: VOrd + MinMax + Element, B: Element>(
    input: &mut [T],
    input_shape: Shape,
    kernel: MorphKernel<B>,
    op: MorphOp,
) {
    let [_, _, ch] = input_shape.dims();
    let border_value = match op {
        MorphOp::Erode => vec![T::MAX; ch],
        MorphOp::Dilate => vec![T::MIN; ch],
    };
    let ksize = kernel.ksize();
    let anchor = kernel.anchor();
    match op {
        MorphOp::Erode => {
            let row_filter = ErodeRow::<T>::new(ksize.1, anchor.1);
            let col_filter = ErodeCol::<T>::new(ksize.0, anchor.0);
            dispatch_morph(input, input_shape, row_filter, col_filter, &border_value);
        }
        MorphOp::Dilate => {
            let row_filter = DilateRow::<T>::new(ksize.1, anchor.1);
            let col_filter = DilateCol::<T>::new(ksize.0, anchor.0);
            dispatch_morph(input, input_shape, row_filter, col_filter, &border_value);
        }
    };
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
#[pulp::with_simd(dispatch_morph = pulp::Arch::new())]
fn run_morph_simd<S: Simd, T: VOrd + MinMax + Debug, Op: MorphOperator<T> + VecMorphOperator<T>>(
    simd: S,
    buffer: &mut [T],
    buffer_shape: Shape,
    row_filter: filter_engine::RowFilter<T, Op>,
    col_filter: filter_engine::ColFilter<T, Op>,
    border_value: &[T],
) {
    let mut engine = FilterEngine::new(row_filter, col_filter, border_value);
    engine.apply(simd, buffer, buffer_shape);
}
