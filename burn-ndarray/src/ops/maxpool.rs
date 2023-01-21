use super::padding::apply_padding2d;
use crate::{element::NdArrayElement, tensor::NdArrayTensor, NdArrayBackend, NdArrayDevice};
use burn_tensor::{ops::TensorOps, Data, Shape};

/// This method is not the most efficient, but it serves as a basic implementation that is easy to understand.
/// A more optimized version should be used in its place.
pub(crate) fn max_pool2d_with_indexes_naive<E: NdArrayElement>(
    x: &NdArrayTensor<E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> (NdArrayTensor<E, 4>, NdArrayTensor<i64, 4>) {
    let mut batches = Vec::new();
    let mut batches_indexes = Vec::new();

    let [batch_size, channels, heigth, width] = x.shape.dims;

    for b in 0..batch_size {
        let mut batch = Vec::new();
        let mut batch_indexes = Vec::new();

        for c in 0..channels {
            let x = NdArrayBackend::index(x, [b..b + 1, c..c + 1, 0..heigth, 0..width]);
            let x = NdArrayBackend::reshape(&x, Shape::new([heigth, width]));
            let x = apply_padding2d(&x, padding);

            let (matrix, indexes) = max_pool2d_with_kernel(x, kernel_size, stride, padding);
            let [heigth, width] = matrix.shape.dims;

            let matrix = NdArrayBackend::reshape(&matrix, Shape::new([1, 1, heigth, width]));
            let indexes = NdArrayBackend::reshape(&indexes, Shape::new([1, 1, heigth, width]));

            batch.push(matrix);
            batch_indexes.push(indexes);
        }
        let batch = NdArrayBackend::cat(&batch, 1);
        let batch_indexes = NdArrayBackend::cat(&batch_indexes, 1);

        batches.push(batch);
        batches_indexes.push(batch_indexes);
    }

    (
        NdArrayBackend::cat(&batches, 0),
        NdArrayBackend::cat(&batches_indexes, 0),
    )
}

pub(crate) fn max_pool2d_backward_naive<E: NdArrayElement>(
    x: &NdArrayTensor<E, 4>,
    _kernel_size: [usize; 2],
    _stride: [usize; 2],
    _padding: [usize; 2],
    output_grad: &NdArrayTensor<E, 4>,
    indexes: &NdArrayTensor<i64, 4>,
) -> NdArrayTensor<E, 4> {
    let [_batch_size, _channels, heigth, width] = output_grad.shape.dims;
    let [batch_size, channels, heigth_x, width_x] = x.shape.dims;

    let output_grad_flatten = NdArrayBackend::reshape(
        output_grad,
        Shape::new([batch_size, channels, heigth * width]),
    );
    let indexes_flatten =
        NdArrayBackend::reshape(indexes, Shape::new([batch_size, channels, heigth * width]));
    let mut output_flatten = NdArrayBackend::zeros(
        Shape::new([batch_size, channels, heigth_x * width_x]),
        NdArrayDevice::Cpu,
    );

    for b in 0..batch_size {
        for c in 0..channels {
            for i in 0..(heigth * width) {
                let index = NdArrayBackend::index(&indexes_flatten, [b..b + 1, c..c + 1, i..i + 1]);
                let index = NdArrayBackend::into_data(index).value[0] as usize;

                let current_value =
                    NdArrayBackend::index(&output_flatten, [b..b + 1, c..c + 1, index..index + 1]);
                let output_grad =
                    NdArrayBackend::index(&output_grad_flatten, [b..b + 1, c..c + 1, i..i + 1]);
                let updated_value = NdArrayBackend::add(&current_value, &output_grad);

                output_flatten = NdArrayBackend::index_assign(
                    &output_flatten,
                    [b..b + 1, c..c + 1, index..index + 1],
                    &updated_value,
                );
            }
        }
    }

    NdArrayBackend::reshape(&output_flatten, x.shape)
}

fn max_pool2d_with_kernel<E: NdArrayElement>(
    x: NdArrayTensor<E, 2>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> (NdArrayTensor<E, 2>, NdArrayTensor<i64, 2>) {
    let [k1, k2] = kernel_size;
    let [p1, p2] = padding;
    let [heigth, width] = x.shape.dims;

    let heigth_new = f32::ceil((heigth - k1 + 1) as f32 / stride[0] as f32) as usize;
    let width_new = f32::ceil((width - k2 + 1) as f32 / stride[1] as f32) as usize;
    let mut output = NdArrayBackend::empty(Shape::new([heigth_new, width_new]), NdArrayDevice::Cpu);
    let mut indexes =
        NdArrayBackend::empty(Shape::new([heigth_new, width_new]), NdArrayDevice::Cpu);

    for i in 0..heigth_new {
        for j in 0..width_new {
            let i_x = i * stride[0];
            let j_x = j * stride[1];

            let x_ij = NdArrayBackend::index(&x, [i_x..i_x + k1, j_x..j_x + k2]);
            let x_flatten = NdArrayBackend::reshape(&x_ij, Shape::new([k1 * k2]));
            let index = NdArrayBackend::argmax(&x_flatten, 0);
            let index = NdArrayBackend::into_data(index).value[0];
            let value = NdArrayBackend::into_data(x_flatten).value[index as usize];
            let value = NdArrayBackend::from_data(
                Data::new(vec![value], Shape::new([1, 1])),
                NdArrayDevice::Cpu,
            );

            let index_i = index / k2 as i64;
            let index_j = index - (index_i * k2 as i64);
            let ii = i64::max(0, i_x as i64 - p1 as i64 + index_i);
            let jj = i64::max(0, j_x as i64 - p2 as i64 + index_j);
            let h = (heigth - (2 * p1)) as i64;
            let index = ii * h + jj;

            let index = NdArrayBackend::from_data(
                Data::new(vec![index], Shape::new([1, 1])),
                NdArrayDevice::Cpu,
            );

            indexes = NdArrayBackend::index_assign(&indexes, [i..i + 1, j..j + 1], &index);
            output = NdArrayBackend::index_assign(&output, [i..i + 1, j..j + 1], &value);
        }
    }

    (output, indexes)
}
