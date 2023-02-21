use alloc::vec::Vec;

use super::padding::apply_padding2d;
use crate::{element::NdArrayElement, tensor::NdArrayTensor, NdArrayBackend, NdArrayDevice};

use burn_tensor::{ops::TensorOps, Shape};

use libm::ceilf;

/// This method is not the most efficient, but it serves as a basic implementation that is easy to understand.
/// A more optimized version should be used in its place.
pub(crate) fn conv2d_naive<E: NdArrayElement>(
    x: NdArrayTensor<E, 4>,
    weight: NdArrayTensor<E, 4>,
    bias: Option<NdArrayTensor<E, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> NdArrayTensor<E, 4> {
    let [batch_size, channels_in, heigth, width] = x.shape().dims;
    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let x = NdArrayBackend::index(x.clone(), [b..b + 1, 0..channels_in, 0..heigth, 0..width]);
        let x = NdArrayBackend::reshape(x.clone(), Shape::new([channels_in, heigth, width]));

        results.push(conv2d_naive_no_batch_size(
            x,
            weight.clone(),
            bias.clone(),
            stride,
            padding,
        ));
    }

    NdArrayBackend::cat(results, 0)
}

pub(crate) fn conv2d_naive_no_batch_size<E: NdArrayElement>(
    x: NdArrayTensor<E, 3>,
    weight: NdArrayTensor<E, 4>,
    bias: Option<NdArrayTensor<E, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> NdArrayTensor<E, 4> {
    let [channels_out, channels_in, k1, k2] = weight.shape().dims;
    let [_, heigth, width] = x.shape().dims;
    let mut results = Vec::new();

    for co in 0..channels_out {
        let mut matrices = Vec::new();

        for ci in 0..channels_in {
            let kernel =
                NdArrayBackend::index(weight.clone(), [co..co + 1, ci..ci + 1, 0..k1, 0..k2]);
            let kernel = NdArrayBackend::reshape(kernel.clone(), Shape::new([k1, k2]));

            let x = NdArrayBackend::index(x.clone(), [ci..ci + 1, 0..heigth, 0..width]);
            let x = NdArrayBackend::reshape(x.clone(), Shape::new([heigth, width]));
            let x = apply_padding2d(x, padding);

            let matrix = conv2d_with_kernel(x, kernel, stride);
            let [heigth, width] = matrix.shape().dims;
            let matrix = NdArrayBackend::reshape(matrix.clone(), Shape::new([1, 1, heigth, width]));

            matrices.push(matrix);
        }
        let matrices = NdArrayBackend::cat(matrices, 1);
        let matrices = NdArrayBackend::sum_dim(matrices.clone(), 1);

        results.push(matrices);
    }

    let mut result = NdArrayBackend::cat(results, 1);

    if let Some(bias) = bias {
        let [size] = bias.shape().dims;
        let bias = NdArrayBackend::reshape(bias, Shape::new([1, size, 1, 1]));
        result = NdArrayBackend::add(result, bias);
    }

    result
}

fn conv2d_with_kernel<E: NdArrayElement>(
    x: NdArrayTensor<E, 2>,
    kernel: NdArrayTensor<E, 2>,
    stride: [usize; 2],
) -> NdArrayTensor<E, 2> {
    let [k1, k2] = kernel.shape().dims;
    let [heigth, width] = x.shape().dims;

    let heigth_new = ceilf((heigth - k1 + 1) as f32 / stride[0] as f32) as usize;
    let width_new = ceilf((width - k2 + 1) as f32 / stride[1] as f32) as usize;
    let mut output =
        NdArrayBackend::empty(Shape::new([heigth_new, width_new]), &NdArrayDevice::Cpu);

    for i in 0..heigth_new {
        for j in 0..width_new {
            let i_x = i * stride[0];
            let j_x = j * stride[1];

            let x_ij = NdArrayBackend::index(x.clone(), [i_x..i_x + k1, j_x..j_x + k2]);
            let value = NdArrayBackend::mul(x_ij, kernel.clone());
            let value = NdArrayBackend::sum(value);
            let value = NdArrayBackend::reshape(value, Shape::new([1, 1]));

            output = NdArrayBackend::index_assign(output, [i..i + 1, j..j + 1], value);
        }
    }

    output
}
