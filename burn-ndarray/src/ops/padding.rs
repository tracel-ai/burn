use crate::{element::NdArrayElement, tensor::NdArrayTensor, NdArrayBackend, NdArrayDevice};
use burn_tensor::{ops::TensorOps, Shape};

pub(crate) fn apply_padding2d<E: NdArrayElement>(
    x: &NdArrayTensor<E, 2>,
    padding: [usize; 2],
) -> NdArrayTensor<E, 2> {
    let [heigth, width] = x.shape().dims;

    let heigth_new = heigth + (2 * padding[0]);
    let width_new = width + (2 * padding[1]);

    let mut x_new = NdArrayBackend::zeros(Shape::new([heigth_new, width_new]), NdArrayDevice::Cpu);
    x_new = NdArrayBackend::index_assign(
        &x_new,
        [
            padding[0]..heigth + padding[0],
            padding[1]..width + padding[1],
        ],
        x,
    );

    x_new
}
