use crate::decorator::SparseCSR;
use crate::decorator::SparseDecorator;
use burn_tensor::{backend::Backend, sparse_backend::SparseBackend};
use core::marker::PhantomData;

#[derive(Debug, Default, Clone)]
pub struct SparseCSRTensor<B: Backend, const D: usize> {
    _b: PhantomData<B>,
}

impl<B> SparseBackend for SparseDecorator<B, SparseCSR>
where
    B: Backend,
{
    type SparseTensorPrimitive<const D: usize> = SparseCSRTensor<B, D>;

    fn sparse_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &burn_tensor::Device<Self>,
    ) -> burn_tensor::ops::SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_to_sparse<const D: usize>(
        dense: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }

    fn sparse_to_dense<const D: usize>(
        sparse: Self::SparseTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D> {
        todo!()
    }

    fn sparse_spmm<const D: usize>(
        lhs: Self::SparseTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
    ) -> Self::FloatTensorPrimitive<D> {
        todo!()
    }

    fn sparse_sddmm<const D: usize>(
        lhs: Self::SparseTensorPrimitive<D>,
        rhs: Self::FloatTensorPrimitive<D>,
    ) -> Self::SparseTensorPrimitive<D> {
        todo!()
    }

    fn sparse_slice<const D1: usize, const D2: usize>(
        tensor: burn_tensor::ops::SparseTensor<Self, D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> burn_tensor::ops::SparseTensor<Self, D1> {
        todo!()
    }

    fn sparse_device<const D: usize>(
        tensor: &burn_tensor::ops::SparseTensor<Self, D>,
    ) -> burn_tensor::Device<Self> {
        todo!()
    }

    fn sparse_to_device<const D: usize>(
        tensor: burn_tensor::ops::SparseTensor<Self, D>,
        device: &burn_tensor::Device<Self>,
    ) -> burn_tensor::ops::SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_shape<const D: usize>(
        tensor: &burn_tensor::ops::SparseTensor<Self, D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn sparse_into_data<const D: usize>(
        tensor: burn_tensor::ops::SparseTensor<Self, D>,
    ) -> impl std::future::Future<Output = burn_tensor::TensorData> + Send {
        async { todo!() }
    }

    fn sparse_from_data<const D: usize>(
        data: burn_tensor::TensorData,
        device: &burn_tensor::Device<Self>,
    ) -> burn_tensor::ops::SparseTensor<Self, D> {
        todo!()
    }
}
