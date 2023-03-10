use alloc::vec::Vec;
use core::ops::Range;

use crate::{backend::Backend, tensor::Shape, Data};

/// Int Tensor API for basic and numeric operations, see [tensor](crate::Tensor)
/// for documentation on each function.
pub trait IntTensorOps<B: Backend> {
    fn int_empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> B::IntTensorPrimitive<D>;
    fn int_shape<const D: usize>(tensor: &B::IntTensorPrimitive<D>) -> Shape<D>;
    fn int_into_data<const D: usize>(tensor: B::IntTensorPrimitive<D>) -> Data<B::IntElem, D>;
    fn int_to_data<const D: usize>(tensor: &B::IntTensorPrimitive<D>) -> Data<B::IntElem, D> {
        Self::int_into_data(tensor.clone())
    }
    fn int_from_data<const D: usize>(
        data: Data<B::IntElem, D>,
        device: &B::Device,
    ) -> B::IntTensorPrimitive<D>;
    fn int_device<const D: usize>(tensor: &B::IntTensorPrimitive<D>) -> B::Device;
    fn int_to_device<const D: usize>(
        tensor: B::IntTensorPrimitive<D>,
        device: &B::Device,
    ) -> B::IntTensorPrimitive<D>;
    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: B::IntTensorPrimitive<D1>,
        shape: Shape<D2>,
    ) -> B::IntTensorPrimitive<D2>;
    fn int_index<const D1: usize, const D2: usize>(
        tensor: B::IntTensorPrimitive<D1>,
        indexes: [Range<usize>; D2],
    ) -> B::IntTensorPrimitive<D1>;
    fn int_index_assign<const D1: usize, const D2: usize>(
        tensor: B::IntTensorPrimitive<D1>,
        indexes: [Range<usize>; D2],
        value: B::IntTensorPrimitive<D1>,
    ) -> B::IntTensorPrimitive<D1>;
    fn int_repeat<const D: usize>(
        tensor: B::IntTensorPrimitive<D>,
        dim: usize,
        times: usize,
    ) -> B::IntTensorPrimitive<D> {
        let mut shape = Self::int_shape(&tensor);
        if shape.dims[dim] != 1 {
            panic!("Can only repeat dimension with dim=1");
        }
        shape.dims[dim] = times;

        let mut i = 0;
        let indexes_select_all = [0; D].map(|_| {
            let start = 0;
            let end = shape.dims[i];
            i += 1;
            start..end
        });

        let mut tensor_output = Self::int_empty(shape, &Self::int_device(&tensor));
        for i in 0..times {
            let mut indexes = indexes_select_all.clone();
            indexes[dim] = i..i + 1;
            tensor_output = Self::int_index_assign(tensor_output, indexes, tensor.clone());
        }

        tensor_output
    }
    fn int_cat<const D: usize>(
        tensors: Vec<B::IntTensorPrimitive<D>>,
        dim: usize,
    ) -> B::IntTensorPrimitive<D>;
    fn int_equal<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntTensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;
    fn int_equal_elem<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntElem,
    ) -> B::BoolTensorPrimitive<D>;
    fn int_greater<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntTensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;
    fn int_greater_elem<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntElem,
    ) -> B::BoolTensorPrimitive<D>;
    fn int_greater_equal<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntTensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;
    fn int_greater_equal_elem<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntElem,
    ) -> B::BoolTensorPrimitive<D>;
    fn int_lower<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntTensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;
    fn int_lower_elem<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntElem,
    ) -> B::BoolTensorPrimitive<D>;
    fn int_lower_equal<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntTensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;
    fn int_lower_equal_elem<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntElem,
    ) -> B::BoolTensorPrimitive<D>;

    // ====  NUMERIC ==== //
    fn int_add<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntTensorPrimitive<D>,
    ) -> B::IntTensorPrimitive<D>;
    fn int_add_scalar<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntElem,
    ) -> B::IntTensorPrimitive<D>;
    fn int_sub<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntTensorPrimitive<D>,
    ) -> B::IntTensorPrimitive<D>;
    fn int_sub_scalar<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntElem,
    ) -> B::IntTensorPrimitive<D>;
    fn int_mul<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntTensorPrimitive<D>,
    ) -> B::IntTensorPrimitive<D>;
    fn int_mul_scalar<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntElem,
    ) -> B::IntTensorPrimitive<D>;
    fn int_div<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntTensorPrimitive<D>,
    ) -> B::IntTensorPrimitive<D>;
    fn int_div_scalar<const D: usize>(
        lhs: B::IntTensorPrimitive<D>,
        rhs: B::IntElem,
    ) -> B::IntTensorPrimitive<D>;
    fn int_neg<const D: usize>(tensor: B::IntTensorPrimitive<D>) -> B::IntTensorPrimitive<D>;
    fn int_zeros<const D: usize>(shape: Shape<D>, device: &B::Device) -> B::IntTensorPrimitive<D>;
    fn int_ones<const D: usize>(shape: Shape<D>, device: &B::Device) -> B::IntTensorPrimitive<D>;
    fn int_sum<const D: usize>(tensor: B::IntTensorPrimitive<D>) -> B::IntTensorPrimitive<1>;
    fn int_sum_dim<const D: usize>(
        tensor: B::IntTensorPrimitive<D>,
        dim: usize,
    ) -> B::IntTensorPrimitive<D>;
    fn int_mean<const D: usize>(tensor: B::IntTensorPrimitive<D>) -> B::IntTensorPrimitive<1>;
    fn int_mean_dim<const D: usize>(
        tensor: B::IntTensorPrimitive<D>,
        dim: usize,
    ) -> B::IntTensorPrimitive<D>;
}
