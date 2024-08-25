use super::coo::COO;
use burn_tensor::{backend::Backend, ops::SparseIntOps, SparseStorage};

impl<B: Backend> SparseIntOps<COO, B> for COO {
    fn int_empty<const D: usize>(
        shape: burn_tensor::Shape<D>,
        device: &burn_tensor::Device<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D> {
        todo!()
    }

    fn int_shape<const D: usize>(
        tensor: &<COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
    ) -> burn_tensor::Shape<D> {
        todo!()
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D2> {
        todo!()
    }

    fn int_transpose<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D> {
        todo!()
    }

    fn int_swap_dims<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
        dim1: usize,
        dim2: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D> {
        todo!()
    }

    fn int_permute<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
        axes: &[usize],
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D> {
        todo!()
    }

    fn int_flip<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
        axes: &[usize],
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D> {
        todo!()
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D1>,
        indices: [std::ops::Range<usize>; D2],
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D1> {
        todo!()
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D1>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D1> {
        todo!()
    }

    fn int_device<const D: usize>(
        tensor: &<COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
    ) -> burn_tensor::Device<B> {
        todo!()
    }

    fn int_to_device<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
        device: &burn_tensor::Device<B>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D> {
        todo!()
    }

    fn int_repeat_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
        dim: usize,
        times: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D> {
        todo!()
    }

    fn int_cat<const D: usize>(
        tensors: Vec<<COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D> {
        todo!()
    }

    fn int_equal<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
        rhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn int_not_equal<const D: usize>(
        lhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
        rhs: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn int_any<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, 1> {
        todo!()
    }

    fn int_any_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn int_all<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, 1> {
        todo!()
    }

    fn int_all_dim<const D: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D>,
        dim: usize,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Bool, D> {
        todo!()
    }

    fn int_expand<const D1: usize, const D2: usize>(
        tensor: <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D1>,
        shape: burn_tensor::Shape<D2>,
    ) -> <COO as SparseStorage<B>>::SparsePrimitive<burn_tensor::Int, D2> {
        todo!()
    }
}
