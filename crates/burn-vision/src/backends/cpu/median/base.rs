use core::cmp::Ordering;

use burn_core::{
    backend::{
        BackendTensor, DispatchAutodiffContext, DispatchDevice, DispatchTensor, DispatchTensorKind,
        flex::FlexTensor,
    },
    tensor::{DType, Element, Tensor, TensorData, kind::Ordered},
};
use burn_std::{bf16, f16};
use bytemuck::Pod;
use macerator::VOrd;

use super::{autodiff, histogram, network};

/// Dispatch only Flex tensors here, leaving other backends on the decomposition.
pub(crate) fn try_median<K: Ordered>(images: &Tensor<4, K>, size: usize) -> Option<Tensor<4, K>> {
    let device = images.device().without_autodiff();
    if !matches!(device.as_dispatch(), DispatchDevice::Flex(_))
        || matches!(images.dtype(), DType::QFloat(_))
        || images.dims().contains(&0)
        || size == 0
    {
        return None;
    }
    size.checked_mul(size)?;

    let dispatch = images.clone().into_dispatch();
    let context = dispatch.autodiff;
    if context != DispatchAutodiffContext::Disabled && images.dtype().is_float() {
        let images = Tensor::<4>::from_dispatch(dispatch);
        return autodiff::median(&images, size)
            .map(|output| Tensor::from_dispatch(output.into_dispatch()));
    }

    // Integer tensors carry an autodiff context but do not participate in the graph.
    let dispatch = if context != DispatchAutodiffContext::Disabled {
        Tensor::<4, burn_core::tensor::Int>::from_dispatch(dispatch)
            .inner()
            .into_dispatch()
    } else {
        dispatch
    };
    let output = match dispatch.kind {
        DispatchTensorKind::Flex(BackendTensor::Float(input)) => {
            BackendTensor::Float(filter(input, size)?)
        }
        DispatchTensorKind::Flex(BackendTensor::Int(input)) => {
            BackendTensor::Int(filter(input, size)?)
        }
        _ => return None,
    };
    Some(Tensor::from_dispatch(DispatchTensor {
        kind: DispatchTensorKind::Flex(output),
        autodiff: context,
    }))
}

fn filter(input: FlexTensor, size: usize) -> Option<FlexTensor> {
    let input = input.to_contiguous();
    Some(match input.dtype() {
        DType::F32 => {
            let mut keys = Vec::new();
            filter_typed(
                input,
                size,
                f32::total_cmp,
                |plane, output, height, width| {
                    network::filter_3x3(plane, output, height, width, &mut keys)
                },
                histogram::try_filter,
            )
        }
        DType::F64 => {
            let mut keys = Vec::new();
            filter_typed(
                input,
                size,
                f64::total_cmp,
                |plane, output, height, width| {
                    network::filter_f64(plane, output, height, width, &mut keys)
                },
                |_, _, _, _, _| false,
            )
        }
        DType::F16 => filter_half(input, size, f16::to_f32, f16::from_f32),
        DType::BF16 => filter_half(input, size, bf16::to_f32, bf16::from_f32),
        dtype if dtype.is_int() || dtype.is_uint() => {
            crate::dispatch_int_dtype!(dtype.into(), |E| filter_integer::<E>(input, size))
        }
        _ => return None,
    })
}

fn filter_integer<E: Element + Pod + Ord + VOrd>(input: FlexTensor, size: usize) -> FlexTensor {
    filter_typed(
        input,
        size,
        E::cmp,
        network::filter_ord,
        histogram::try_integer::<E>,
    )
}

fn filter_half<H: Element + Pod>(
    input: FlexTensor,
    size: usize,
    to_f32: fn(H) -> f32,
    from_f32: fn(f32) -> H,
) -> FlexTensor {
    let shape = input.layout().shape().clone();
    let values = input
        .as_slice::<H>()
        .unwrap()
        .iter()
        .copied()
        .map(to_f32)
        .collect::<Vec<_>>();
    let input = FlexTensor::from_data(TensorData::new(values, shape.clone()));
    let filtered = filter(input, size).expect("F32 median is supported");
    let values = filtered
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .copied()
        .map(from_f32)
        .collect::<Vec<_>>();
    FlexTensor::from_data(TensorData::new(values, shape))
}

fn filter_typed<E: Element + Pod>(
    input: FlexTensor,
    size: usize,
    compare: impl Fn(&E, &E) -> Ordering + Copy,
    mut filter_3x3: impl FnMut(&[E], &mut [E], usize, usize),
    mut filter_histogram: impl FnMut(&[E], &mut [E], usize, usize, usize) -> bool,
) -> FlexTensor {
    if size == 1 {
        return input;
    }
    let shape = input.layout().shape().clone();
    let [_, _, height, width] = shape.dims();
    let values = input.as_slice::<E>().expect("Contiguous median input");
    let mut output = vec![values[0]; values.len()];
    let mut scratch = Vec::new();
    let plane_len = height * width;

    for (plane, output) in values
        .chunks_exact(plane_len)
        .zip(output.chunks_exact_mut(plane_len))
    {
        if plane.iter().all(|value| compare(value, &plane[0]).is_eq()) {
            output.copy_from_slice(plane);
        } else if size == 3 {
            filter_3x3(plane, output, height, width);
        } else if size < 5 || !filter_histogram(plane, output, height, width, size) {
            scratch.resize(size * size, plane[0]);
            select_windows(plane, output, [height, width], size, &mut scratch, compare);
        }
    }
    FlexTensor::from_data(TensorData::new(output, shape))
}

/// Map a window offset to an edge-replicated image coordinate.
pub(super) fn coordinate(position: usize, offset: usize, reach: usize, length: usize) -> usize {
    (position + offset).saturating_sub(reach).min(length - 1)
}

fn select_windows<E: Copy>(
    input: &[E],
    output: &mut [E],
    [height, width]: [usize; 2],
    size: usize,
    scratch: &mut [E],
    compare: impl Fn(&E, &E) -> Ordering + Copy,
) {
    let reach = size / 2;
    let middle = scratch.len() / 2;
    for y in 0..height {
        for x in 0..width {
            for (dy, row) in scratch.chunks_exact_mut(size).enumerate() {
                let sy = coordinate(y, dy, reach, height);
                let source = &input[sy * width..(sy + 1) * width];
                if x >= reach && size <= width && x - reach <= width - size {
                    row.copy_from_slice(&source[x - reach..x - reach + size]);
                } else {
                    for (dx, value) in row.iter_mut().enumerate() {
                        *value = source[coordinate(x, dx, reach, width)];
                    }
                }
            }
            // Large flat regions need no partitioning.
            let first = scratch[0];
            output[y * width + x] = if scratch.iter().all(|value| compare(value, &first).is_eq()) {
                first
            } else {
                *scratch.select_nth_unstable_by(middle, compare).1
            };
        }
    }
}
