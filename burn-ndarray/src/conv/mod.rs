use crate::{element::NdArrayElement, tensor::NdArrayTensor, NdArrayBackend, NdArrayDevice};
use burn_tensor::{ops::TensorOps, Shape};

pub(crate) fn conv2d<E: NdArrayElement>(
    x: &NdArrayTensor<E, 3>,
    weight: &NdArrayTensor<E, 4>,
    bias: &Option<NdArrayTensor<E, 1>>,
    stride: [usize; 2],
    padding: [usize; 2],
) -> NdArrayTensor<E, 4> {
    let [channels_out, channels_in, k1, k2] = weight.shape.dims;
    let mut results = Vec::new();

    for co in 0..channels_out {
        let mut matrices = Vec::new();

        for ci in 0..channels_in {
            let kernel = NdArrayBackend::index(weight, [co..co + 1, ci..ci + 1, 0..k1, 0..k2]);
            let kernel = NdArrayBackend::reshape(&kernel, Shape::new([k1, k2]));

            let x = apply_padding(&x, ci, padding);

            let matrix = conv2d_with_kernel(x, kernel, stride);
            let [heigth, width] = matrix.shape.dims;
            let matrix = NdArrayBackend::reshape(&matrix, Shape::new([1, 1, heigth, width]));

            matrices.push(matrix);
        }
        let matrices = NdArrayBackend::cat(&matrices, 1);
        results.push(matrices);
    }

    let mut result = NdArrayBackend::cat(&results, 0);
    result = NdArrayBackend::sum_dim(&result, 0);

    if let Some(bias) = bias {
        let [size] = bias.shape.dims;
        let bias = NdArrayBackend::reshape(bias, Shape::new([1, size, 1, 1]));
        result = NdArrayBackend::add(&result, &bias);
    }

    result
}

fn apply_padding<E: NdArrayElement>(
    x: &NdArrayTensor<E, 3>,
    channel: usize,
    padding: [usize; 2],
) -> NdArrayTensor<E, 2> {
    let [_, heigth, width] = x.shape.dims;
    let heigth_new = heigth + (2 * padding[0]);
    let width_new = width + (2 * padding[1]);

    let x = NdArrayBackend::index(&x, [channel..channel + 1, 0..heigth, 0..width]);
    let x = NdArrayBackend::reshape(&x, Shape::new([heigth, width]));

    let mut x_new = NdArrayBackend::zeros(Shape::new([heigth_new, width_new]), NdArrayDevice::Cpu);
    x_new = NdArrayBackend::index_assign(
        &x_new,
        [
            padding[0]..heigth + padding[0],
            padding[1]..width + padding[1],
        ],
        &x,
    );

    x_new
}

fn conv2d_with_kernel<E: NdArrayElement>(
    x: NdArrayTensor<E, 2>,
    kernel: NdArrayTensor<E, 2>,
    stride: [usize; 2],
) -> NdArrayTensor<E, 2> {
    let [k1, k2] = kernel.shape.dims;
    let [heigth, width] = x.shape.dims;
    let heigth_new = heigth / stride[0] - k1 + 1;
    let width_new = width / stride[1] - k2 + 1;

    let mut output = NdArrayBackend::empty(Shape::new([heigth_new, width_new]), NdArrayDevice::Cpu);

    for i in 0..heigth_new {
        for j in 0..width_new {
            let x_ij = NdArrayBackend::index(&x, [i..i + k1, j..j + k2]);

            let value = NdArrayBackend::mul(&x_ij, &kernel);
            let value = NdArrayBackend::sum(&value);
            let value = NdArrayBackend::reshape(&value, Shape::new([1, 1]));

            output = NdArrayBackend::index_assign(&output, [i..i + 1, j..j + 1], &value);
        }
    }

    output
}
