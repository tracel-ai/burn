use burn_ndarray::NdArrayTensor;
use burn_tensor::{
    backend::Backend,
    cast::ToElement,
    ops::BoolTensor,
    quantization::{QuantizationScheme, QuantizationType},
    BasicOps, Bool, DType, Element, Shape, Tensor, TensorData,
};
use cubecl::prelude::Array;
use macerator::VOrd;
use ndarray::Array4;
use pulp::Simd;

use super::MinMax;

mod filter;

/// A morphology operation.
/// TODO: Implement composite ops
pub enum MorphOp {
    Erode,
    Dilate,
}

pub enum MorphKernel<B: Element> {
    Rect {
        shape: [usize; 4],
        anchor: (usize, usize),
    },
    Other {
        kernel: Array4<B>,
        anchor: (usize, usize),
    },
}

pub fn morph<B: Backend, K: BasicOps<B>>(
    input: Tensor<B, 4, K>,
    kernel: BoolTensor<B>,
    op: MorphOp,
) -> Tensor<B, 4, K> {
    let device = input.device();

    let kernel = Tensor::<B, 4, Bool>::new(kernel);
    let k_shape = kernel.shape().dims();
    let [_, _, kh, kw] = k_shape;

    let data = kernel.into_data().into_vec::<B::BoolElem>().unwrap();
    let kernel = unsafe { Array4::from_shape_vec_unchecked(k_shape, data) };
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

    let dims = input.shape().dims::<4>();
    let data = input.into_data();
    match data.dtype {
        DType::F64 => morph_typed::<B, K, f64>(data, kernel, op, &device),
        DType::F32 => morph_typed::<B, K, f32>(data, kernel, op, &device),
        DType::F16 | DType::BF16 => {
            morph_typed::<B, K, f32>(data.convert::<f32>(), kernel, op, &device)
        }
        DType::I64 => morph_typed::<B, K, i64>(data, kernel, op, &device),
        DType::I32 => morph_typed::<B, K, i32>(data, kernel, op, &device),
        DType::I16 => morph_typed::<B, K, i16>(data, kernel, op, &device),
        DType::I8 => morph_typed::<B, K, i8>(data, kernel, op, &device),
        DType::U64 => morph_typed::<B, K, u64>(data, kernel, op, &device),
        DType::U32 => morph_typed::<B, K, u32>(data, kernel, op, &device),
        DType::U16 => morph_typed::<B, K, u16>(data, kernel, op, &device),
        DType::U8 => morph_typed::<B, K, u8>(data, kernel, op, &device),
        DType::Bool => morph_bool::<B, K>(data, kernel, op, &device),
        DType::QFloat(scheme) => match scheme {
            QuantizationScheme::PerTensorAffine(QuantizationType::QInt8) => {
                morph_typed::<B, K, i8>(data, kernel, op, &device)
            }
            QuantizationScheme::PerTensorSymmetric(QuantizationType::QInt8) => {
                morph_typed::<B, K, i8>(data, kernel, op, &device)
            }
        },
    }
}

fn morph_typed<B: Backend, K: BasicOps<B>, T: VOrd + MinMax + Element>(
    input: TensorData,
    kernel: MorphKernel<B::BoolElem>,
    op: MorphOp,
    device: &B::Device,
) -> Tensor<B, 4, K> {
    let shape = input.shape.clone();
    let input = input.into_vec::<T>().unwrap();
    let input = unsafe {
        Array4::from_shape_vec_unchecked([shape[0], shape[1], shape[2], shape[3]], input)
    };
    let out = run_morph(input, kernel, op);
    let (b, ch, h, w) = out.dim();
    let (data, _) = out.into_raw_vec_and_offset();
    let data = TensorData::new(data, Shape::new([b, ch, h, w]));
    Tensor::from_data(data, device)
}

fn morph_bool<B: Backend, K: BasicOps<B>>(
    input: TensorData,
    kernel: MorphKernel<B::BoolElem>,
    op: MorphOp,
    device: &B::Device,
) -> Tensor<B, 4, K> {
    let shape = input.shape.clone();
    let input = input.into_vec::<bool>().unwrap();
    let input = bytemuck::cast_vec::<_, u8>(input);
    let input = unsafe {
        Array4::from_shape_vec_unchecked([shape[0], shape[1], shape[2], shape[3]], input)
    };
    let out = run_morph(input, kernel, op);
    let (b, ch, h, w) = out.dim();
    let (data, _) = out.into_raw_vec_and_offset();
    // SAFETY: Morph can't produce invalid boolean values
    let data = unsafe { core::mem::transmute::<Vec<u8>, Vec<bool>>(data) };
    let data = TensorData::new(data, Shape::new([b, ch, h, w]));
    Tensor::from_data(data, device)
}

fn run_morph<T: VOrd + MinMax, B: Element>(
    input: Array4<T>,
    kernel: MorphKernel<B>,
    op: MorphOp,
) -> Array4<T> {
    todo!()
}

struct FilterEngine<S: Simd, T: VOrd> {
    // Vector aligned ring buffer to serve as intermediate, since image isn't always aligned
    ring_buf: Vec<T::Vector<S>>,
}
