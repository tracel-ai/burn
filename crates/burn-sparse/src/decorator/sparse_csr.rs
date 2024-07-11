use crate::backend::SparseBackend;
use crate::backend::SparseTensor;
use crate::decorator::SparseCSR;
use crate::decorator::SparseDecorator;
use burn_tensor::backend::Backend;
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
    ) -> SparseTensor<Self, D> {
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
        tensor: SparseTensor<Self, D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> SparseTensor<Self, D1> {
        todo!()
    }

    fn sparse_device<const D: usize>(tensor: &SparseTensor<Self, D>) -> burn_tensor::Device<Self> {
        todo!()
    }

    fn sparse_to_device<const D: usize>(
        tensor: SparseTensor<Self, D>,
        device: &burn_tensor::Device<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_shape<const D: usize>(tensor: &SparseTensor<Self, D>) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn sparse_into_data<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> impl std::future::Future<Output = burn_tensor::TensorData> + Send {
        async { todo!() }
    }

    fn sparse_from_data<const D: usize>(
        data: burn_tensor::TensorData,
        device: &burn_tensor::Device<Self>,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_reshape<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> SparseTensor<Self, D2> {
        todo!()
    }

    fn sparse_transpose<const D: usize>(tensor: SparseTensor<Self, D>) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_swap_dims<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_permute<const D: usize>(
        tensor: SparseTensor<Self, D>,
        axes: &[usize],
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_flip<const D: usize>(
        tensor: SparseTensor<Self, D>,
        axes: &[usize],
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_slice_assign<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: SparseTensor<Self, D1>,
    ) -> SparseTensor<Self, D1> {
        todo!()
    }

    fn sparse_repeat<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_cat<const D: usize>(
        tensors: Vec<SparseTensor<Self, D>>,
        dim: usize,
    ) -> SparseTensor<Self, D> {
        todo!()
    }

    fn sparse_equal<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_not_equal<const D: usize>(
        lhs: SparseTensor<Self, D>,
        rhs: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_any<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, 1> {
        todo!()
    }

    fn sparse_any_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_all<const D: usize>(
        tensor: SparseTensor<Self, D>,
    ) -> burn_tensor::ops::BoolTensor<Self, 1> {
        todo!()
    }

    fn sparse_all_dim<const D: usize>(
        tensor: SparseTensor<Self, D>,
        dim: usize,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        todo!()
    }

    fn sparse_expand<const D1: usize, const D2: usize>(
        tensor: SparseTensor<Self, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> SparseTensor<Self, D2> {
        todo!()
    }
}
