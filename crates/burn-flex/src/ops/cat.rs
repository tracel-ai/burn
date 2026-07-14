//! Concatenation operations for FlexTensor.

use alloc::vec::Vec;
use burn_backend::{DType, Element};
use burn_std::{Bytes, Shape, bf16, f16};

use crate::{FlexTensor, Layout};

/// Concatenate tensors along the given dimension.
pub fn cat(tensors: Vec<FlexTensor>, dim: usize) -> FlexTensor {
    assert!(!tensors.is_empty(), "cat: cannot concatenate empty list");
    if tensors.len() == 1 {
        return tensors.into_iter().next().unwrap();
    }

    let dtype = tensors[0].dtype();
    match dtype {
        DType::F32 => cat_impl::<f32>(tensors, dim),
        DType::F64 => cat_impl::<f64>(tensors, dim),
        DType::F16 => cat_impl::<f16>(tensors, dim),
        DType::BF16 => cat_impl::<bf16>(tensors, dim),
        DType::I64 => cat_impl::<i64>(tensors, dim),
        DType::I32 => cat_impl::<i32>(tensors, dim),
        DType::I16 => cat_impl::<i16>(tensors, dim),
        DType::I8 => cat_impl::<i8>(tensors, dim),
        DType::U64 => cat_impl::<u64>(tensors, dim),
        DType::U32 => cat_impl::<u32>(tensors, dim),
        DType::U16 => cat_impl::<u16>(tensors, dim),
        DType::U8 | DType::Bool(_) => cat_impl::<u8>(tensors, dim),
        _ => panic!("cat: unsupported dtype {:?}", dtype),
    }
}

fn cat_impl<E: Element + bytemuck::Pod>(tensors: Vec<FlexTensor>, dim: usize) -> FlexTensor {
    let dtype = tensors[0].dtype();
    let first_shape = tensors[0].layout().shape();
    let ndims = first_shape.num_dims();

    assert!(
        dim < ndims,
        "cat: dim {} out of bounds for {} dimensions",
        dim,
        ndims
    );

    // Compute output shape: sum along cat dim, others must match
    let mut out_dims = first_shape.to_vec();
    out_dims[dim] = 0;
    for t in &tensors {
        assert_eq!(
            t.dtype(),
            dtype,
            "cat: dtype mismatch: expected {:?}, got {:?}",
            dtype,
            t.dtype()
        );
        let s = t.layout().shape();
        assert_eq!(s.num_dims(), ndims, "cat: dimension count mismatch");
        for (d, out_d) in out_dims.iter_mut().enumerate() {
            if d == dim {
                *out_d += s[d];
            } else {
                assert_eq!(
                    s[d], first_shape[d],
                    "cat: shape mismatch at dim {d}: expected {}, got {}",
                    first_shape[d], s[d]
                );
            }
        }
    }

    let out_shape = Shape::from(out_dims.clone());
    let total_elements: usize = out_shape.num_elements();

    if total_elements == 0 {
        let bytes = Bytes::from_elems::<E>(Vec::new());
        return FlexTensor::new(bytes, Layout::contiguous(out_shape), dtype);
    }

    let mut output: Vec<E> = Vec::with_capacity(total_elements);

    // Fast path: dim 0, all contiguous
    if dim == 0
        && tensors
            .iter()
            .all(|t| t.layout().contiguous_offsets().is_some())
    {
        for t in &tensors {
            let (start, end) = t.layout().contiguous_offsets().unwrap();
            let data: &[E] = t.storage();
            output.extend_from_slice(&data[start..end]);
        }
    } else {
        // General path: iterate over outer/inner chunks
        // outer_size = product of dims before `dim`
        // inner_size = product of dims after `dim`
        let outer_size: usize = out_dims[..dim].iter().product();
        let inner_size: usize = out_dims[dim + 1..].iter().product();

        // Make all tensors contiguous for simple indexing
        let contiguous: Vec<FlexTensor> = tensors.into_iter().map(|t| t.to_contiguous()).collect();

        for outer in 0..outer_size {
            for t in &contiguous {
                let data: &[E] = t.storage();
                let t_dim_size = t.layout().shape()[dim];
                let t_start = t.layout().start_offset();
                // Each tensor's chunk for this outer index:
                // offset = t_start + outer * t_dim_size * inner_size
                let chunk_start = t_start + outer * t_dim_size * inner_size;
                let chunk_len = t_dim_size * inner_size;
                output.extend_from_slice(&data[chunk_start..chunk_start + chunk_len]);
            }
        }
    }

    let bytes = Bytes::from_elems(output);
    FlexTensor::new(bytes, Layout::contiguous(out_shape), dtype)
}
