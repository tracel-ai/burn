use alloc::{vec, vec::Vec};
use ndarray::Array4;

use super::padding::{apply_padding2d, apply_padding_4d};
use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor, NdArrayBackend, NdArrayDevice};

use burn_tensor::{
    ops::{IntTensorOps, TensorOps},
    Data, ElementConversion, Shape,
};

use libm::ceilf;

pub(crate) fn max_pool2d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> (NdArrayTensor<E, 4>, NdArrayTensor<i64, 4>) {
    let [kernel_height, kernel_width] = kernel_size;
    let [padding_height, padding_width] = padding;
    let [stride_height, stride_width] = stride;
    let [batch_size, in_channels, in_height, in_width] = x.shape().dims;

    let out_height = ((in_height + 2 * padding_height - kernel_height) / stride_height) + 1;
    let out_width = ((in_width + 2 * padding_width - kernel_width) / stride_width) + 1;

    let x = apply_padding_4d(x, padding).array;

    let mut output = Array4::zeros((batch_size, in_channels, out_height, out_width));
    let mut indexes = Array4::<i64>::zeros((batch_size, in_channels, out_height, out_width));

    for b in 0..batch_size {
        for ic in 0..in_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut max_val = (-f32::INFINITY).elem::<E>();
                    let mut index = 0;

                    for kh in 0..kernel_height {
                        for kw in 0..kernel_width {
                            let ih = oh * stride_height + kh;
                            let iw = ow * stride_width + kw;

                            let val = x[[b, ic, ih, iw]];

                            if val > max_val {
                                max_val = val;

                                let ih = ih as i64 - padding_height as i64;
                                let iw = iw as i64 - padding_width as i64;

                                index = ih * in_height as i64 + iw;
                            }
                        }
                    }

                    output[[b, ic, oh, ow]] = max_val;
                    indexes[[b, ic, oh, ow]] = index;
                }
            }
        }
    }

    println!("out {:?}", output);
    let output = NdArrayTensor::new(output.into_dyn().into_shared());
    let indexes = NdArrayTensor::new(indexes.into_dyn().into_shared());

    (output, indexes)
}

/// This method is not the most efficient, but it serves as a basic implementation that is easy to understand.
/// A more optimized version should be used in its place.
pub(crate) fn max_pool2d_with_indexes_naive<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> (NdArrayTensor<E, 4>, NdArrayTensor<i64, 4>) {
    let mut batches = Vec::new();
    let mut batches_indexes = Vec::new();

    let [batch_size, channels, height, width] = x.shape().dims;

    for b in 0..batch_size {
        let mut batch = Vec::new();
        let mut batch_indexes = Vec::new();

        for c in 0..channels {
            let x = NdArrayBackend::index(x.clone(), [b..b + 1, c..c + 1, 0..height, 0..width]);
            let x = NdArrayBackend::reshape(x.clone(), Shape::new([height, width]));
            let x = apply_padding2d(x.clone(), padding);

            let (matrix, indexes) = max_pool2d_with_kernel(x, kernel_size, stride, padding);
            let [height, width] = matrix.shape().dims;

            let matrix = NdArrayBackend::reshape(matrix, Shape::new([1, 1, height, width]));
            let indexes =
                NdArrayBackend::<E>::int_reshape(indexes, Shape::new([1, 1, height, width]));

            batch.push(matrix);
            batch_indexes.push(indexes);
        }
        let batch = NdArrayBackend::cat(batch, 1);
        let batch_indexes = NdArrayBackend::<E>::int_cat(batch_indexes, 1);

        batches.push(batch);
        batches_indexes.push(batch_indexes);
    }

    (
        NdArrayBackend::cat(batches, 0),
        NdArrayBackend::<E>::int_cat(batches_indexes, 0),
    )
}

pub(crate) fn max_pool2d_backward_naive<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    _kernel_size: [usize; 2],
    _stride: [usize; 2],
    _padding: [usize; 2],
    output_grad: NdArrayTensor<E, 4>,
    indexes: NdArrayTensor<i64, 4>,
) -> NdArrayTensor<E, 4> {
    let [_batch_size, _channels, height, width] = output_grad.shape().dims;
    let [batch_size, channels, height_x, width_x] = x.shape().dims;

    let output_grad_flatten = NdArrayBackend::reshape(
        output_grad,
        Shape::new([batch_size, channels, height * width]),
    );
    let indexes_flatten = NdArrayBackend::<E>::int_reshape(
        indexes,
        Shape::new([batch_size, channels, height * width]),
    );
    let mut output_flatten = NdArrayBackend::zeros(
        Shape::new([batch_size, channels, height_x * width_x]),
        &NdArrayDevice::Cpu,
    );

    for b in 0..batch_size {
        for c in 0..channels {
            for i in 0..(height * width) {
                let index = NdArrayBackend::<E>::int_index(
                    indexes_flatten.clone(),
                    [b..b + 1, c..c + 1, i..i + 1],
                );
                let index = NdArrayBackend::<E>::int_into_data(index).value[0] as usize;

                let current_value = NdArrayBackend::index(
                    output_flatten.clone(),
                    [b..b + 1, c..c + 1, index..index + 1],
                );
                let output_grad = NdArrayBackend::index(
                    output_grad_flatten.clone(),
                    [b..b + 1, c..c + 1, i..i + 1],
                );
                let updated_value = NdArrayBackend::add(current_value, output_grad);

                output_flatten = NdArrayBackend::index_assign(
                    output_flatten.clone(),
                    [b..b + 1, c..c + 1, index..index + 1],
                    updated_value,
                );
            }
        }
    }

    NdArrayBackend::reshape(output_flatten, x.shape())
}

fn max_pool2d_with_kernel<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 2>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> (NdArrayTensor<E, 2>, NdArrayTensor<i64, 2>) {
    let [k1, k2] = kernel_size;
    let [p1, p2] = padding;
    let [height, width] = x.shape().dims;

    let height_new = ceilf((height - k1 + 1) as f32 / stride[0] as f32) as usize;
    let width_new = ceilf((width - k2 + 1) as f32 / stride[1] as f32) as usize;
    let mut output =
        NdArrayBackend::empty(Shape::new([height_new, width_new]), &NdArrayDevice::Cpu);
    let mut indexes =
        NdArrayBackend::<E>::int_empty(Shape::new([height_new, width_new]), &NdArrayDevice::Cpu);

    for i in 0..height_new {
        for j in 0..width_new {
            let i_x = i * stride[0];
            let j_x = j * stride[1];

            let x_ij = NdArrayBackend::index(x.clone(), [i_x..i_x + k1, j_x..j_x + k2]);
            let x_flatten = NdArrayBackend::reshape(x_ij, Shape::new([k1 * k2]));
            let index = NdArrayBackend::argmax(x_flatten.clone(), 0);
            let index = NdArrayBackend::<E>::int_into_data(index).value[0];
            let value = NdArrayBackend::into_data(x_flatten).value[index as usize];
            let value = NdArrayBackend::from_data(
                Data::new(vec![value], Shape::new([1, 1])),
                &NdArrayDevice::Cpu,
            );

            let index_i = index / k2 as i64;
            let index_j = index - (index_i * k2 as i64);
            let ii = i64::max(0, i_x as i64 - p1 as i64 + index_i);
            let jj = i64::max(0, j_x as i64 - p2 as i64 + index_j);
            let h = (height - (2 * p1)) as i64;
            let index = ii * h + jj;

            let index = NdArrayBackend::<E>::int_from_data(
                Data::new(vec![index], Shape::new([1, 1])),
                &NdArrayDevice::Cpu,
            );

            indexes = NdArrayBackend::<E>::int_index_assign(indexes, [i..i + 1, j..j + 1], index);
            output = NdArrayBackend::index_assign(output, [i..i + 1, j..j + 1], value);
        }
    }

    (output, indexes)
}
