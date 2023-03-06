use alloc::vec::Vec;
use core::ops::Range;

use crate::{backend::Backend, tensor::Shape, Data};

/// Bool Tensor API for basic operations, see [tensor](crate::Tensor)
/// for documentation on each function.
pub trait BoolTensorOps<B: Backend> {
    fn bool_empty<const D: usize>(shape: Shape<D>, device: &B::Device)
        -> B::BoolTensorPrimitive<D>;
    fn bool_shape<const D: usize>(tensor: &B::BoolTensorPrimitive<D>) -> Shape<D>;
    fn bool_into_data<const D: usize>(tensor: B::BoolTensorPrimitive<D>) -> Data<bool, D>;
    fn bool_to_data<const D: usize>(tensor: &B::BoolTensorPrimitive<D>) -> Data<bool, D> {
        Self::bool_into_data(tensor.clone())
    }
    fn bool_from_data<const D: usize>(
        data: Data<bool, D>,
        device: &B::Device,
    ) -> B::BoolTensorPrimitive<D>;
    fn bool_into_int<const D: usize>(tensor: B::BoolTensorPrimitive<D>)
        -> B::IntTensorPrimitive<D>;
    fn bool_device<const D: usize>(tensor: &B::BoolTensorPrimitive<D>) -> B::Device;
    fn bool_to_device<const D: usize>(
        tensor: B::BoolTensorPrimitive<D>,
        device: &B::Device,
    ) -> B::BoolTensorPrimitive<D>;
    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: B::BoolTensorPrimitive<D1>,
        shape: Shape<D2>,
    ) -> B::BoolTensorPrimitive<D2>;
    fn bool_index<const D1: usize, const D2: usize>(
        tensor: B::BoolTensorPrimitive<D1>,
        indexes: [Range<usize>; D2],
    ) -> B::BoolTensorPrimitive<D1>;
    fn bool_index_assign<const D1: usize, const D2: usize>(
        tensor: B::BoolTensorPrimitive<D1>,
        indexes: [Range<usize>; D2],
        value: B::BoolTensorPrimitive<D1>,
    ) -> B::BoolTensorPrimitive<D1>;
    fn bool_repeat<const D: usize>(
        tensor: B::BoolTensorPrimitive<D>,
        dim: usize,
        times: usize,
    ) -> B::BoolTensorPrimitive<D> {
        let mut shape = Self::bool_shape(&tensor);
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

        let mut tensor_output = Self::bool_empty(shape, &Self::bool_device(&tensor));
        for i in 0..times {
            let mut indexes = indexes_select_all.clone();
            indexes[dim] = i..i + 1;
            tensor_output = Self::bool_index_assign(tensor_output, indexes, tensor.clone());
        }

        tensor_output
    }
    fn bool_cat<const D: usize>(
        tensors: Vec<B::BoolTensorPrimitive<D>>,
        dim: usize,
    ) -> B::BoolTensorPrimitive<D>;
    fn bool_equal<const D: usize>(
        lhs: B::BoolTensorPrimitive<D>,
        rhs: B::BoolTensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;
    fn bool_equal_elem<const D: usize>(
        lhs: B::BoolTensorPrimitive<D>,
        rhs: bool,
    ) -> B::BoolTensorPrimitive<D>;
}
