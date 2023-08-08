use alloc::vec::Vec;
use core::ops::Range;

use crate::{backend::Backend, tensor::Shape, Data};

/// Bool Tensor API for basic operations, see [tensor](crate::Tensor)
/// for documentation on each function.
pub trait BoolTensorOps<B: Backend> {
    /// Creates a new bool tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The boolean tensor with the given shape.
    fn bool_empty<const D: usize>(shape: Shape<D>, device: &B::Device)
        -> B::BoolTensorPrimitive<D>;

    /// Returns the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn bool_shape<const D: usize>(tensor: &B::BoolTensorPrimitive<D>) -> Shape<D>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn bool_into_data<const D: usize>(tensor: B::BoolTensorPrimitive<D>) -> Data<bool, D>;

    /// Gets the data from the tensor.
    ///
    ///
    /// # Arguments
    ///
    /// * `data` - The data structure.
    ///
    /// # Returns
    ///
    /// The data cloned from the data structure.
    fn bool_to_data<const D: usize>(tensor: &B::BoolTensorPrimitive<D>) -> Data<bool, D> {
        Self::bool_into_data(tensor.clone())
    }

    /// Creates a tensor from the data structure.
    ///
    /// # Arguments
    ///
    /// * `data` - The data structure.
    /// * `device` - The device to create the tensor on.
    ///
    /// # Returns
    ///
    /// The tensor with the data.
    fn bool_from_data<const D: usize>(
        data: Data<bool, D>,
        device: &B::Device,
    ) -> B::BoolTensorPrimitive<D>;

    /// Converts bool tensor to int tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The int tensor with the same data as the bool tensor.
    fn bool_into_int<const D: usize>(tensor: B::BoolTensorPrimitive<D>)
        -> B::IntTensorPrimitive<D>;

    /// Converts bool tensor to float tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The float tensor with the same data as the bool tensor.
    fn bool_into_float<const D: usize>(tensor: B::BoolTensorPrimitive<D>) -> B::TensorPrimitive<D>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn bool_device<const D: usize>(tensor: &B::BoolTensorPrimitive<D>) -> B::Device;

    /// Moves the tensor to the device.
    fn bool_to_device<const D: usize>(
        tensor: B::BoolTensorPrimitive<D>,
        device: &B::Device,
    ) -> B::BoolTensorPrimitive<D>;

    /// Reshapes the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `shape` - The new shape.
    ///
    /// # Returns
    ///
    /// The tensor with the new shape.
    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: B::BoolTensorPrimitive<D1>,
        shape: Shape<D2>,
    ) -> B::BoolTensorPrimitive<D2>;

    /// Gets the values from the tensor for the given ranges.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `ranges` - The ranges to get the values from.
    ///
    /// # Returns
    ///
    /// The tensor with the values for the given ranges.
    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: B::BoolTensorPrimitive<D1>,
        ranges: [Range<usize>; D2],
    ) -> B::BoolTensorPrimitive<D1>;

    /// Sets the values in the tensor for the given ranges.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `ranges` - The ranges to set the values for.
    /// * `value` - The values to set.
    ///
    /// # Returns
    ///
    /// The tensor with the values set for the given ranges.
    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: B::BoolTensorPrimitive<D1>,
        ranges: [Range<usize>; D2],
        value: B::BoolTensorPrimitive<D1>,
    ) -> B::BoolTensorPrimitive<D1>;

    /// Repeats one dimension of the tensor a given number of times along that dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension to repeat.
    /// * `times` - The number of times to repeat the dimension.
    ///
    /// # Returns
    ///
    /// The tensor with the dimension repeated.
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
        let ranges_select_all = [0; D].map(|_| {
            let start = 0;
            let end = shape.dims[i];
            i += 1;
            start..end
        });

        let mut tensor_output = Self::bool_empty(shape, &Self::bool_device(&tensor));
        for i in 0..times {
            let mut ranges = ranges_select_all.clone();
            ranges[dim] = i..i + 1;
            tensor_output = Self::bool_slice_assign(tensor_output, ranges, tensor.clone());
        }

        tensor_output
    }

    /// Concatenates the tensors along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `tensors` - The tensors to concatenate.
    /// * `dim` - The dimension to concatenate along.
    ///
    /// # Returns
    ///
    /// The tensor with the tensors concatenated along the given dimension.
    fn bool_cat<const D: usize>(
        tensors: Vec<B::BoolTensorPrimitive<D>>,
        dim: usize,
    ) -> B::BoolTensorPrimitive<D>;

    /// Equates the two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The tensor with the result of the equate.
    fn bool_equal<const D: usize>(
        lhs: B::BoolTensorPrimitive<D>,
        rhs: B::BoolTensorPrimitive<D>,
    ) -> B::BoolTensorPrimitive<D>;

    /// Equates the tensor with the element.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side element.
    ///
    /// # Returns
    ///
    /// The tensor with the result of the equate.
    fn bool_equal_elem<const D: usize>(
        lhs: B::BoolTensorPrimitive<D>,
        rhs: bool,
    ) -> B::BoolTensorPrimitive<D>;
}
