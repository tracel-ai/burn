macro_rules! keepdim {
    (
        $dim:expr,
        $self:expr,
        mean
    ) => {{
        let tensor: SharedArray<E> = mean_dim($self.clone(), $dim);
        let mut shape = $self.shape().into_shape();
        shape.dims[$dim] = 1;
        NdArrayOps::reshape(tensor.clone(), shape)
    }};
    (
        $dim:expr,
        $self:expr,
        sum
    ) => {{
        let tensor: SharedArray<E> = sum_dim($self.clone(), $dim);
        let mut shape = $self.shape().into_shape();
        shape.dims[$dim] = 1;
        NdArrayOps::reshape(tensor, shape)
    }};
    (
        $dim:expr,
        $self:expr,
        prod
    ) => {{
        let tensor: SharedArray<E> = prod_dim($self.clone(), $dim);
        let mut shape = $self.shape().into_shape();
        shape.dims[$dim] = 1;
        NdArrayOps::reshape(tensor, shape)
    }};
}

use burn_tensor::ElementConversion;
pub(crate) use keepdim;
use ndarray::{Axis, Zip};

use crate::{SharedArray, element::NdArrayElement};

pub(crate) fn mean_dim<E: NdArrayElement>(tensor: SharedArray<E>, dim: usize) -> SharedArray<E> {
    tensor.mean_axis(Axis(dim)).unwrap().into_shared()
}

pub(crate) fn sum_dim<E: NdArrayElement>(tensor: SharedArray<E>, dim: usize) -> SharedArray<E> {
    tensor.sum_axis(Axis(dim)).into_shared()
}

pub(crate) fn prod_dim<E: NdArrayElement>(tensor: SharedArray<E>, dim: usize) -> SharedArray<E> {
    tensor
        .fold_axis(Axis(dim), 1.elem::<E>(), |acc, &x| acc.mul(x.elem()))
        .into_shared()
}

pub(crate) fn cumsum_dim<E: NdArrayElement>(
    tensor: SharedArray<E>,
    dim: usize,
) -> SharedArray<E> {
    let axis = Axis(dim);
    let shape = tensor.shape().to_vec();
    let mut result = tensor.to_owned();

    // Compute cumulative sum along the specified axis
    let dim_size = shape[dim];
    for i in 1..dim_size {
        let prev = result.index_axis(axis, i - 1).to_owned();
        let mut current = result.index_axis_mut(axis, i);
        Zip::from(&mut current)
            .and(&prev)
            .for_each(|c, &p| *c = c.add(p.elem()));
    }

    result.into_shared()
}
