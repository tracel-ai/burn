use std::cmp::max;
use std::marker::PhantomData;

use crate::{
    Candle, CandleDevice, CandleTensor,
    element::{CandleElement, FloatCandleElement, IntCandleElement},
};
use burn_tensor::ops::unfold::{calculate_unfold_shape, calculate_unfold_windows};
use burn_tensor::{Element, Shape, TensorData, TensorMetadata, backend::Backend};
use candle_core::{Layout, WithDType};
use half::{bf16, f16};

use super::tensor;

pub fn cat(tensors: Vec<CandleTensor>, dim: usize) -> CandleTensor {
    let tensors: Vec<candle_core::Tensor> = tensors.into_iter().map(|t| t.tensor).collect();
    CandleTensor::new(candle_core::Tensor::cat(&tensors, dim).unwrap())
}

pub fn from_data<E: CandleElement>(data: TensorData, device: &CandleDevice) -> CandleTensor {
    CandleTensor::from_data::<E>(data, device.clone())
}
pub fn into_data(tensor: CandleTensor) -> TensorData {
    fn tensor_data_from_dtype<T: WithDType + Element>(tensor: &CandleTensor) -> TensorData {
        TensorData::new(
            tensor.tensor.flatten_all().unwrap().to_vec1::<T>().unwrap(),
            tensor.shape(),
        )
    }

    match tensor.tensor.dtype() {
        candle_core::DType::BF16 => tensor_data_from_dtype::<bf16>(&tensor),
        candle_core::DType::F16 => tensor_data_from_dtype::<f16>(&tensor),
        candle_core::DType::F32 => tensor_data_from_dtype::<f32>(&tensor),
        candle_core::DType::F64 => tensor_data_from_dtype::<f64>(&tensor),
        candle_core::DType::U8 => tensor_data_from_dtype::<u8>(&tensor),
        candle_core::DType::U32 => tensor_data_from_dtype::<u32>(&tensor),
        candle_core::DType::I64 => tensor_data_from_dtype::<i64>(&tensor),
    }
}

pub fn to_device(tensor: CandleTensor, device: &CandleDevice) -> CandleTensor {
    CandleTensor::new(tensor.tensor.to_device(&(device.clone()).into()).unwrap())
}

pub fn empty(shape: Shape, device: &CandleDevice, dtype: candle_core::DType) -> CandleTensor {
    zeros(shape, device, dtype)
}

pub fn zeros(shape: Shape, device: &CandleDevice, dtype: candle_core::DType) -> CandleTensor {
    CandleTensor::new(
        candle_core::Tensor::zeros(shape.dims, dtype, &(device.clone()).into()).unwrap(),
    )
}

pub fn ones(shape: Shape, device: &CandleDevice, dtype: candle_core::DType) -> CandleTensor {
    CandleTensor::new(
        candle_core::Tensor::ones(shape.dims, dtype, &(device.clone()).into()).unwrap(),
    )
}

pub fn swap_dims(mut tensor: CandleTensor, dim1: usize, dim2: usize) -> CandleTensor {
    CandleTensor::new(tensor.tensor.transpose(dim1, dim2).unwrap())
}

pub fn permute(tensor: CandleTensor, axes: &[usize]) -> CandleTensor {
    CandleTensor::new(tensor.tensor.permute(axes).unwrap())
}

pub fn flip(tensor: CandleTensor, axes: &[usize]) -> CandleTensor {
    // FIXME: Replace with an appropriate method when Candle provides one.
    let mut tensor = tensor.tensor;
    for &axis in axes {
        // Ensure tensor is contiguous before index_select (required by Candle)
        tensor = tensor.contiguous().unwrap();

        let indexes = candle_core::Tensor::arange_step(
            tensor.dim(axis).unwrap() as i64 - 1,
            -1,
            -1,
            tensor.device(),
        )
        .unwrap();
        tensor = tensor.index_select(&indexes, axis).unwrap();
    }

    CandleTensor::new(tensor)
}

pub fn reshape(tensor: CandleTensor, shape: Shape) -> CandleTensor {
    CandleTensor::new(tensor.tensor.reshape(shape.dims).unwrap())
}

pub fn device(tensor: &CandleTensor) -> CandleDevice {
    tensor.tensor.device().clone().into()
}

pub fn shape(tensor: &CandleTensor) -> Shape {
    tensor.shape()
}

pub fn slice(tensor: CandleTensor, ranges: &[std::ops::Range<usize>]) -> CandleTensor {
    let mut narrow_tensor = tensor.tensor;
    for (i, range) in ranges.iter().enumerate().take(ranges.len()) {
        narrow_tensor = narrow_tensor
            .narrow(i, range.start, range.end - range.start)
            .unwrap()
    }
    CandleTensor::new(narrow_tensor)
}

pub fn slice_with_steps(tensor: CandleTensor, slices: &[burn_tensor::Slice]) -> CandleTensor {
    let mut result_tensor = tensor.tensor;

    for (dim, slice) in slices.iter().enumerate() {
        if slice.step == 1 {
            // Use narrow for step=1 (more efficient)
            // Convert slice to range using tensor shape
            let dim_size = result_tensor.dim(dim).unwrap();
            let range = slice.to_range(dim_size);
            let start = range.start;
            let length = range.end - range.start;
            result_tensor = result_tensor.narrow(dim, start, length).unwrap();
        } else {
            // Use index_select for step != 1
            let dim_size = result_tensor.dim(dim).unwrap();
            let range = slice.to_range(dim_size);
            let start = range.start;
            let end = range.end;
            let step = slice.step;

            // Generate indices based on step direction
            let indices_vec = if step > 0 {
                // Forward stepping
                let step_usize = step as usize;
                (start..end).step_by(step_usize).collect::<Vec<_>>()
            } else {
                // Backward stepping (negative step)
                let step_usize = step.unsigned_abs();
                // Start from end-1 and go backwards
                let mut indices = Vec::new();
                let mut idx = end - 1;
                while idx >= start && idx < end {
                    // Check for underflow
                    indices.push(idx);
                    if idx >= step_usize {
                        idx -= step_usize;
                    } else {
                        break;
                    }
                }
                indices
            };

            // Convert indices to tensor and use index_select
            let indices_len = indices_vec.len();
            let device = result_tensor.device();
            let indices = candle_core::Tensor::from_vec(
                indices_vec.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                indices_len,
                device,
            )
            .unwrap();

            result_tensor = result_tensor.index_select(&indices, dim).unwrap();
        }
    }

    CandleTensor::new(result_tensor)
}

pub fn slice_assign(
    tensor: CandleTensor,
    slices: &[burn_tensor::Slice],
    value: CandleTensor,
) -> CandleTensor {
    // Check if all slices have step=1 (candle's native slice_assign requirement)
    let all_unit_steps = slices.iter().all(|s| s.step == 1);

    if all_unit_steps {
        // Convert Slice to Range for candle's native slice_assign
        let ranges: Vec<std::ops::Range<usize>> = slices
            .iter()
            .enumerate()
            .map(|(dim, slice)| {
                let dim_size = tensor.tensor.dim(dim).unwrap_or(usize::MAX);
                slice.to_range(dim_size)
            })
            .collect();

        CandleTensor::new(tensor.tensor.slice_assign(&ranges, &value.tensor).unwrap())
    } else {
        // Implement slice_assign with steps using scatter operations
        slice_assign_with_steps_workaround(tensor, slices, value)
    }
}

/// Implements slice_assign for non-unit steps using index operations
fn slice_assign_with_steps_workaround(
    tensor: CandleTensor,
    slices: &[burn_tensor::Slice],
    value: CandleTensor,
) -> CandleTensor {
    let shape = tensor.shape();
    let ndims = shape.num_dims();
    let device = tensor.tensor.device();

    // Generate indices for each dimension based on slice specifications
    let indices_per_dim = generate_slice_indices(slices, &shape.dims);

    // Early return if no elements to assign
    let total_elements: usize = indices_per_dim.iter().map(|v| v.len()).product();
    if total_elements == 0 {
        return tensor;
    }

    // Flatten tensors and get metadata
    let value_flat = value.tensor.flatten_all().unwrap();
    let strides = tensor.tensor.stride();
    let tensor_shape = tensor.tensor.dims();

    // Use a macro to handle different dtypes without code duplication
    macro_rules! apply_slice_assign {
        ($dtype:ty, $to_vec_fn:ident) => {{
            let mut tensor_vec: Vec<$dtype> =
                tensor.tensor.flatten_all().unwrap().$to_vec_fn().unwrap();
            let value_vec: Vec<$dtype> = value_flat.$to_vec_fn().unwrap();

            // Apply assignments using cartesian product of indices
            for (value_idx, &value) in value_vec.iter().enumerate() {
                let flat_idx = compute_flat_index(value_idx, &indices_per_dim, &strides);
                if flat_idx < tensor_vec.len() {
                    tensor_vec[flat_idx] = value;
                }
            }

            candle_core::Tensor::from_vec(tensor_vec, tensor_shape, device).unwrap()
        }};
    }

    use candle_core::DType;
    let result = match tensor.tensor.dtype() {
        DType::F32 => apply_slice_assign!(f32, to_vec1),
        DType::F64 => apply_slice_assign!(f64, to_vec1),
        DType::I64 => apply_slice_assign!(i64, to_vec1),
        DType::U32 => apply_slice_assign!(u32, to_vec1),
        DType::U8 => apply_slice_assign!(u8, to_vec1),
        _ => panic!(
            "Unsupported dtype {:?} for slice_assign with steps",
            tensor.tensor.dtype()
        ),
    };

    CandleTensor::new(result)
}

/// Generate indices for each dimension based on slice specifications
fn generate_slice_indices(slices: &[burn_tensor::Slice], tensor_dims: &[usize]) -> Vec<Vec<usize>> {
    let ndims = tensor_dims.len();
    let mut indices_per_dim = Vec::with_capacity(ndims);

    // Process provided slices
    for (dim_idx, slice) in slices.iter().enumerate() {
        let dim_size = tensor_dims[dim_idx];
        let range = slice.to_range(dim_size);
        let indices = generate_stepped_indices(range.start, range.end, slice.step);
        indices_per_dim.push(indices);
    }

    // Fill remaining dimensions with full ranges
    for &dim_size in tensor_dims.iter().skip(slices.len()) {
        indices_per_dim.push((0..dim_size).collect());
    }

    indices_per_dim
}

/// Generate indices for a single dimension with stepping
fn generate_stepped_indices(start: usize, end: usize, step: isize) -> Vec<usize> {
    if step > 0 {
        // Forward stepping
        (start..end).step_by(step as usize).collect()
    } else if step < 0 {
        // Backward stepping: start from end-1 and go backwards
        let step_size = step.unsigned_abs();
        let mut indices = Vec::new();
        let mut idx = end.saturating_sub(1);

        while idx >= start && idx < end {
            indices.push(idx);
            if idx >= step_size {
                idx -= step_size;
            } else {
                break;
            }
        }
        indices
    } else {
        // This branch should never be reached since step is validated to be non-zero
        panic!("Step cannot be zero")
    }
}

/// Compute flat index from multi-dimensional indices using cartesian product logic
fn compute_flat_index(
    value_idx: usize,
    indices_per_dim: &[Vec<usize>],
    strides: &[usize],
) -> usize {
    let mut flat_idx = 0;
    let mut remainder = value_idx;

    // Convert value_idx to multi-dimensional indices and compute flat tensor index
    for dim in (0..indices_per_dim.len()).rev() {
        let dim_size = indices_per_dim[dim].len();
        let idx_in_dim = remainder % dim_size;
        remainder /= dim_size;

        let actual_idx = indices_per_dim[dim][idx_in_dim];
        flat_idx += actual_idx * strides[dim];
    }

    flat_idx
}

pub fn narrow(tensor: CandleTensor, dim: usize, start: usize, length: usize) -> CandleTensor {
    let tensor = tensor.tensor.narrow(dim, start, length);
    match tensor {
        Ok(tensor) => CandleTensor::new(tensor),
        Err(e) => panic!("error narrow from Candle"),
    }
}

pub fn chunk(tensor: CandleTensor, chunks: usize, dim: usize) -> Vec<CandleTensor> {
    let tensors = tensor.tensor.chunk(chunks, dim);
    match tensors {
        Ok(tensors) => tensors.into_iter().map(CandleTensor::new).collect(),
        Err(e) => panic!("error chunk from Candle"),
    }
}

pub fn expand(tensor: CandleTensor, shape: Shape) -> CandleTensor {
    CandleTensor::new(tensor.tensor.broadcast_as(shape.dims).unwrap())
}

pub fn unfold(tensor: CandleTensor, dim: usize, size: usize, step: usize) -> CandleTensor {
    let result_shape = calculate_unfold_shape(tensor.shape(), dim, size, step);
    let windows = result_shape[dim];

    let mut select_ranges = tensor.shape().into_ranges();
    let new_axis = select_ranges.len();

    let mut stack = Vec::with_capacity(windows);
    for widx in 0..windows {
        let start = widx * step;
        let end = start + size;
        select_ranges[dim] = start..end;

        let mut window_slice = slice(tensor.clone(), &select_ranges);

        window_slice = swap_dims(window_slice, dim, new_axis);
        let window_slice = CandleTensor::new(window_slice.tensor.unsqueeze(new_axis).unwrap());

        stack.push(window_slice);
    }
    cat(stack, dim)
}

pub fn sign(tensor: CandleTensor) -> CandleTensor {
    CandleTensor::new(tensor.tensor.sign().unwrap())
}

pub fn mask_where_broadcasted(
    tensor: CandleTensor,
    mask: CandleTensor,
    value: CandleTensor,
) -> CandleTensor {
    let shape = tensor
        .tensor
        .shape()
        .broadcast_shape_binary_op(mask.tensor.shape(), "where_cond")
        .unwrap();

    let mut tensor = tensor.tensor;
    let mut mask = mask.tensor;
    let mut value = value.tensor;

    if shape != *tensor.shape() {
        tensor = tensor.broadcast_as(shape.clone()).unwrap();
    }
    if shape != *mask.shape() {
        mask = mask.broadcast_as(shape.clone()).unwrap();
    }
    if shape != *value.shape() {
        value = value.broadcast_as(shape).unwrap();
    }

    CandleTensor::new(mask.where_cond(&value, &tensor).unwrap())
}

pub fn cross(lhs: CandleTensor, rhs: CandleTensor, dim: usize) -> CandleTensor {
    let shape_lhs = lhs.shape();
    let shape_rhs = rhs.shape();
    let ndims = shape_lhs.num_dims();

    // Broadcast the shapes except along dim
    let mut broadcast_shape = vec![0; ndims];
    for (i, item) in broadcast_shape.iter_mut().enumerate().take(ndims) {
        if i == dim {
            *item = shape_lhs.dims[i];
        } else {
            let l = shape_lhs.dims[i];
            let r = shape_rhs.dims[i];
            if l == r {
                *item = l;
            } else if l == 1 {
                *item = r;
            } else if r == 1 {
                *item = l;
            } else {
                panic!("Tensors are not broadcastable along dimension {}", i);
            }
        }
    }

    // Broadcast lhs and rhs
    let lhs_broadcast = if shape_lhs == Shape::from(broadcast_shape.clone()) {
        lhs
    } else {
        expand(lhs, Shape::from(broadcast_shape.clone()))
    };
    let rhs_broadcast = if shape_rhs == Shape::from(broadcast_shape.clone()) {
        rhs
    } else {
        expand(rhs, Shape::from(broadcast_shape.clone()))
    };

    // Now, move dim to the last dimension
    let mut perm = (0..ndims).collect::<Vec<_>>();
    perm.remove(dim);
    perm.push(dim);

    let lhs_permuted = permute(lhs_broadcast, &perm);
    let rhs_permuted = permute(rhs_broadcast, &perm);

    // Reshape to (*, 3)
    let total_elements = lhs_permuted.shape().num_elements();
    let batch_size = total_elements / 3;
    let lhs_reshaped = reshape(lhs_permuted, Shape::new([batch_size, 3]));
    let rhs_reshaped = reshape(rhs_permuted, Shape::new([batch_size, 3]));

    // Extract components using narrow and squeeze
    let lhs_0 = CandleTensor::new(
        lhs_reshaped
            .tensor
            .narrow(1, 0, 1)
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );
    let lhs_1 = CandleTensor::new(
        lhs_reshaped
            .tensor
            .narrow(1, 1, 1)
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );
    let lhs_2 = CandleTensor::new(
        lhs_reshaped
            .tensor
            .narrow(1, 2, 1)
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );
    let rhs_0 = CandleTensor::new(
        rhs_reshaped
            .tensor
            .narrow(1, 0, 1)
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );
    let rhs_1 = CandleTensor::new(
        rhs_reshaped
            .tensor
            .narrow(1, 1, 1)
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );
    let rhs_2 = CandleTensor::new(
        rhs_reshaped
            .tensor
            .narrow(1, 2, 1)
            .unwrap()
            .squeeze(1)
            .unwrap(),
    );

    // Compute cross product components
    let result_0 = CandleTensor::new(
        lhs_1
            .tensor
            .mul(&rhs_2.tensor)
            .unwrap()
            .sub(&lhs_2.tensor.mul(&rhs_1.tensor).unwrap())
            .unwrap(),
    );
    let result_1 = CandleTensor::new(
        lhs_2
            .tensor
            .mul(&rhs_0.tensor)
            .unwrap()
            .sub(&lhs_0.tensor.mul(&rhs_2.tensor).unwrap())
            .unwrap(),
    );
    let result_2 = CandleTensor::new(
        lhs_0
            .tensor
            .mul(&rhs_1.tensor)
            .unwrap()
            .sub(&lhs_1.tensor.mul(&rhs_0.tensor).unwrap())
            .unwrap(),
    );

    // Stack the components
    let result_0_unsqueezed = CandleTensor::new(result_0.tensor.unsqueeze(1).unwrap());
    let result_1_unsqueezed = CandleTensor::new(result_1.tensor.unsqueeze(1).unwrap());
    let result_2_unsqueezed = CandleTensor::new(result_2.tensor.unsqueeze(1).unwrap());
    let result = cat(
        vec![
            result_0_unsqueezed,
            result_1_unsqueezed,
            result_2_unsqueezed,
        ],
        1,
    );

    // Reshape back to the broadcast shape with dim at the end
    let mut result_shape = broadcast_shape;
    result_shape.remove(dim);
    result_shape.push(3);
    let result_reshaped = reshape(result, Shape::from(result_shape));

    // Permute back
    let mut inv_perm = vec![0; ndims];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }
    permute(result_reshaped, &inv_perm)
}
