macro_rules! keepdim {
    (
        $D:expr,
        $dim:expr,
        $self:expr,
        mean
    ) => {{
        let tensor: NdArrayTensor<E, $D> = mean_dim($self.clone(), $dim);
        let mut shape = $self.shape();
        shape.dims[$dim] = 1;
        NdArrayOps::reshape(tensor.clone(), shape)
    }};
    (
        $D:expr,
        $dim:expr,
        $self:expr,
        sum
    ) => {{
        let tensor: NdArrayTensor<E, $D> = sum_dim($self.clone(), $dim);
        let mut shape = $self.shape();
        shape.dims[$dim] = 1;
        NdArrayOps::reshape(tensor, shape)
    }};
}

pub(crate) use keepdim;
use ndarray::Axis;

use crate::{element::NdArrayElement, tensor::NdArrayTensor};

pub(crate) fn mean_dim<E: NdArrayElement, const D1: usize, const D2: usize>(
    tensor: NdArrayTensor<E, D1>,
    dim: usize,
) -> NdArrayTensor<E, D2> {
    let array = tensor.array.mean_axis(Axis(dim)).unwrap().into_shared();

    NdArrayTensor { array }
}

pub(crate) fn sum_dim<E: NdArrayElement, const D1: usize, const D2: usize>(
    tensor: NdArrayTensor<E, D1>,
    dim: usize,
) -> NdArrayTensor<E, D2> {
    let array = tensor.array.sum_axis(Axis(dim)).into_shared();

    NdArrayTensor { array }
}
