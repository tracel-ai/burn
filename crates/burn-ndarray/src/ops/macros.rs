macro_rules! keepdim {
    (
        $dim:expr,
        $self:expr,
        mean
    ) => {{
        let tensor: NdArrayTensor<E> = mean_dim($self.clone(), $dim);
        let mut shape = $self.shape();
        shape.dims[$dim] = 1;
        NdArrayOps::reshape(tensor.clone(), shape)
    }};
    (
        $dim:expr,
        $self:expr,
        sum
    ) => {{
        let tensor: NdArrayTensor<E> = sum_dim($self.clone(), $dim);
        let mut shape = $self.shape();
        shape.dims[$dim] = 1;
        NdArrayOps::reshape(tensor, shape)
    }};
    (
        $dim:expr,
        $self:expr,
        prod
    ) => {{
        let tensor: NdArrayTensor<E> = prod_dim($self.clone(), $dim);
        let mut shape = $self.shape();
        shape.dims[$dim] = 1;
        NdArrayOps::reshape(tensor, shape)
    }};
}

use burn_tensor::ElementConversion;
pub(crate) use keepdim;
use ndarray::Axis;

use crate::{element::NdArrayElement, tensor::NdArrayTensor};

pub(crate) fn mean_dim<E: NdArrayElement>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    let array = tensor.array.mean_axis(Axis(dim)).unwrap().into_shared();

    NdArrayTensor { array }
}

pub(crate) fn sum_dim<E: NdArrayElement>(tensor: NdArrayTensor<E>, dim: usize) -> NdArrayTensor<E> {
    let array = tensor.array.sum_axis(Axis(dim)).into_shared();

    NdArrayTensor { array }
}

pub(crate) fn prod_dim<E: NdArrayElement>(
    tensor: NdArrayTensor<E>,
    dim: usize,
) -> NdArrayTensor<E> {
    let array = tensor
        .array
        .fold_axis(Axis(dim), 1.elem::<E>(), |acc, &x| acc.mul(x.elem()))
        .into_shared();

    NdArrayTensor { array }
}
