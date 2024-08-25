use super::coo::COO;
use burn_tensor::{
    backend::Backend,
    ops::{SparseBoolOps, SparseTensorOps},
    SparseStorage,
};

impl<B: Backend> SparseBoolOps<COO, B> for COO {
    fn bool_to_sparse<const D: usize>(
        dense: <B as Backend>::BoolTensorPrimitive<D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &burn_tensor::Device<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_shape<const D: usize>(
        tensor: &<COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D2> {
        todo!()
    }

    fn bool_transpose<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_swap_dims<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
        dim1: usize,
        dim2: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_permute<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
        axes: &[usize],
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_flip<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
        axes: &[usize],
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D1> {
        todo!()
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D1>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D1> {
        todo!()
    }

    fn bool_device<const D: usize>(
        tensor: &<COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
    ) -> burn_tensor::Device<B> {
        todo!()
    }

    fn bool_to_device<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
        device: &burn_tensor::Device<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_repeat_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
        dim: usize,
        times: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_cat<const D: usize>(
        tensors: Vec<<COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_equal<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
        rhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_not_equal<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
        rhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_any<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, 1> {
        todo!()
    }

    fn bool_any_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_all<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, 1> {
        todo!()
    }

    fn bool_all_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn bool_expand<const D1: usize, const D2: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D2> {
        todo!()
    }
}
