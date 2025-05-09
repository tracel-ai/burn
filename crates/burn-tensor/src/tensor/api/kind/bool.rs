use crate::{
    AssignOps, BasicOps, Bool, ComparisonOps, CreationOps, Element, ElementConversion,
    ReductionOps, Shape, TensorData, ViewOps, backend::Backend,
};

// BasicOps: CreationOps + AssignOps + ComparisonOps + ReductionOps + ViewOps

impl<B: Backend> CreationOps<B> for Bool {
    fn empty(shape: Shape, device: &B::Device) -> Self::Primitive {
        B::bool_empty(shape, device)
    }

    fn zeros(shape: Shape, device: &B::Device) -> Self::Primitive {
        // B::bool_zeros(shape, device)
        todo!()
    }

    fn ones(shape: Shape, device: &B::Device) -> Self::Primitive {
        // B::bool_ones(shape, device)
        todo!()
    }

    fn full<E: ElementConversion>(
        shape: Shape,
        fill_value: E,
        device: &B::Device,
    ) -> Self::Primitive {
        // B::bool_full(shape, fill_value.elem(), device)
        todo!()
    }
}

impl<B: Backend> AssignOps<B> for Bool {
    fn slice_assign(
        tensor: Self::Primitive,
        ranges: &[core::ops::Range<usize>],
        value: Self::Primitive,
    ) -> Self::Primitive {
        B::bool_slice_assign(tensor, ranges, value)
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
        values: Self::Primitive,
    ) -> Self::Primitive {
        // B::bool_scatter(dim, tensor, indices, values)
        todo!()
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        source: Self::Primitive,
    ) -> Self::Primitive {
        // B::bool_mask_where(tensor, mask, source)
        todo!()
    }

    fn mask_fill(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        value: Self::Elem,
    ) -> Self::Primitive {
        // B::bool_mask_fill(tensor, mask, value)
        todo!()
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: B::IntTensorPrimitive,
        values: Self::Primitive,
    ) -> Self::Primitive {
        // B::bool_select_assign(tensor, dim, indices, values)
        todo!()
    }
}

impl<B: Backend> ComparisonOps<B> for Bool {
    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::bool_equal(lhs, rhs)
    }

    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive {
        B::bool_not_equal(lhs, rhs)
    }

    fn equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        // B::bool_equal_elem(lhs, rhs)
        todo!()
    }

    fn not_equal_elem(lhs: Self::Primitive, rhs: Self::Elem) -> B::BoolTensorPrimitive {
        // B::bool_not_equal_elem(lhs, rhs)
        todo!()
    }
}

impl<B: Backend> ReductionOps<B> for Bool {
    fn any(tensor: Self::Primitive) -> <B as Backend>::BoolTensorPrimitive {
        B::bool_any(tensor)
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> <B as Backend>::BoolTensorPrimitive {
        B::bool_any_dim(tensor, dim)
    }

    fn all(tensor: Self::Primitive) -> <B as Backend>::BoolTensorPrimitive {
        B::bool_all(tensor)
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> <B as Backend>::BoolTensorPrimitive {
        B::bool_all_dim(tensor, dim)
    }
}

impl<B: Backend> ViewOps<B> for Bool {
    fn transpose(tensor: Self::Primitive) -> Self::Primitive {
        B::bool_transpose(tensor)
    }

    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive {
        B::bool_swap_dims(tensor, dim1, dim2)
    }

    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        B::bool_permute(tensor, axes)
    }

    fn slice(tensor: Self::Primitive, ranges: &[core::ops::Range<usize>]) -> Self::Primitive {
        B::bool_slice(tensor, ranges)
    }

    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        B::bool_expand(tensor, shape)
    }
}

impl<B: Backend> BasicOps<B> for Bool {
    fn device(tensor: &Self::Primitive) -> <B as Backend>::Device {
        B::bool_device(tensor)
    }

    fn to_device(tensor: Self::Primitive, device: &<B as Backend>::Device) -> Self::Primitive {
        B::bool_to_device(tensor, device)
    }

    fn register_transaction(tr: &mut crate::Transaction<B>, tensor: Self::Primitive) {
        tr.register_bool(tensor);
    }

    fn from_data(data: crate::TensorData, device: &<B as Backend>::Device) -> Self::Primitive {
        B::bool_from_data(data.convert::<B::IntElem>(), device)
    }

    fn from_data_dtype(
        data: crate::TensorData,
        device: &<B as Backend>::Device,
        dtype: crate::DType,
    ) -> Self::Primitive {
        // Backends only use one bool representation dtype
        if dtype != B::BoolElem::dtype() {
            panic!("Expected bool dtype, got {dtype:?}")
        }

        B::bool_from_data(data.convert_dtype(dtype), device)
    }

    async fn into_data_async(tensor: Self::Primitive) -> TensorData {
        B::bool_into_data(tensor).await
    }

    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        B::bool_reshape(tensor, shape)
    }

    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        B::bool_flip(tensor, axes)
    }

    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive {
        B::bool_repeat_dim(tensor, dim, times)
    }

    fn cat(vectors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive {
        B::bool_cat(vectors, dim)
    }

    fn gather(
        dim: usize,
        tensor: Self::Primitive,
        indices: <B as Backend>::IntTensorPrimitive,
    ) -> Self::Primitive {
        // B::bool_gather(dim, tensor, indices)
        todo!()
    }

    fn select(
        tensor: Self::Primitive,
        dim: usize,
        indices: <B as Backend>::IntTensorPrimitive,
    ) -> Self::Primitive {
        // B::bool_select(tensor, dim, indices)
        todo!()
    }
}
