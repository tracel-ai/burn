use crate::{element::FloatNdArrayElement, tensor::NdArrayTensor, NdArrayBackend, NdArrayDevice};
use burn_tensor::{ops::TensorOps, Shape};

pub(crate) fn apply_padding2d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 2>,
    padding: [usize; 2],
) -> NdArrayTensor<E, 2> {
    let [heigth, width] = x.shape().dims;

    let heigth_new = heigth + (2 * padding[0]);
    let width_new = width + (2 * padding[1]);

    let mut x_new = NdArrayBackend::zeros(Shape::new([heigth_new, width_new]), &NdArrayDevice::Cpu);
    x_new = NdArrayBackend::index_assign(
        x_new,
        [
            padding[0]..heigth + padding[0],
            padding[1]..width + padding[1],
        ],
        x,
    );

    x_new
}

pub(crate) fn apply_padding_4d<E: FloatNdArrayElement>(
    x: NdArrayTensor<E, 4>,
    padding: [usize; 2],
) -> NdArrayTensor<E, 4> {
    let [batch_size, input_channels, height, width] = x.shape().dims;
    let [padding_height, padding_width] = padding;
    let padded_height = height + 2 * padding_height;
    let padded_width = width + 2 * padding_width;

    let mut x_new = NdArrayBackend::zeros(
        Shape::new([batch_size, input_channels, padded_height, padded_width]),
        &NdArrayDevice::Cpu,
    );
    x_new = NdArrayBackend::index_assign(
        x_new,
        [
            0..batch_size,
            0..input_channels,
            padding_height..(height + padding_height),
            padding_width..width + padding_width,
        ],
        x,
    );

    x_new
}
