use std::{future::Future, ops::Range};

use crate::backend::SparseBackend;
use burn_tensor::{backend::Backend, BasicOps, Shape, TensorData, TensorKind};

/// A type-level representation of the kind of a sparse (float) tensor.
#[derive(Clone, Debug)]
pub struct Sparse;

impl<B: SparseBackend> TensorKind<B> for Sparse {
    type Primitive<const D: usize> = B::SparseTensorPrimitive<D>;
    fn name() -> &'static str {
        "Sparse"
    }
}

impl<B: SparseBackend> BasicOps<B> for Sparse {
    type Elem = B::FloatElem;

    fn into_data_async<const D: usize>(
        tensor: Self::Primitive<D>,
    ) -> impl Future<Output = TensorData> + Send {
        B::sparse_into_data(tensor)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        B::sparse_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::sparse_to_device(tensor, device)
    }

    fn from_data<const D: usize>(
        data: TensorData,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::sparse_from_data(data, device)
    }

    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        B::sparse_shape(tensor)
    }

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::sparse_empty(shape, device)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::sparse_slice(tensor, ranges)
    }
}
