use burn_tensor::{
    backend::Backend,
    ops::{SparseBoolOps, SparseTensorOps},
    SparseRepr,
};

use super::coo::COO;
type R = COO;

impl<B: Backend> SparseBoolOps<R, B> for R {
    fn bool_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &burn_tensor::Device<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_shape<const D: usize>(
        tensor: &<R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D2> {
        todo!()
    }

    fn bool_transpose<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_swap_dims<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
        dim1: usize,
        dim2: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_permute<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
        axes: &[usize],
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_flip<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
        axes: &[usize],
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D1> {
        todo!()
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D1>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D1> {
        todo!()
    }

    fn bool_device<const D: usize>(
        tensor: &<R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
    ) -> burn_tensor::Device<B> {
        todo!()
    }

    fn bool_to_device<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
        device: &burn_tensor::Device<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_into_data<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
    ) -> impl std::future::Future<Output = burn_tensor::TensorData> + Send {
        async { todo!() }
    }

    fn bool_from_data<const D: usize>(
        data: burn_tensor::TensorData,
        device: &burn_tensor::Device<B>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_repeat_dim<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
        dim: usize,
        times: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_cat<const D: usize>(
        tensors: Vec<<R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>>,
        dim: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_equal<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_not_equal<const D: usize>(
        lhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
        rhs: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_any<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, 1> {
        todo!()
    }

    fn bool_any_dim<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
        dim: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_all<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, 1> {
        todo!()
    }

    fn bool_all_dim<const D: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D>,
        dim: usize,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_expand<const D1: usize, const D2: usize>(
        tensor: <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <R as SparseRepr<B>>::Primitive<burn_tensor::Bool, D2> {
        todo!()
    }
}
