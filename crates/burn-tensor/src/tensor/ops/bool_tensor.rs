use super::{cat::cat_with_slice_assign, BoolTensor, Device, FloatTensor, IntTensor};
use crate::{
    backend::Backend, chunk, narrow, tensor::Shape, Bool, Data, ElementConversion, Tensor,
};
use alloc::vec::Vec;
use burn_common::reader::Reader;
use core::ops::Range;

#[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
use crate::argwhere;

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
    fn bool_empty<const D: usize>(shape: Shape<D>, device: &Device<B>) -> BoolTensor<B, D>;

    /// Returns the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn bool_shape<const D: usize>(tensor: &BoolTensor<B, D>) -> Shape<D>;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn bool_into_data<const D: usize>(tensor: BoolTensor<B, D>) -> Reader<Data<bool, D>>;

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
    fn bool_to_data<const D: usize>(tensor: &BoolTensor<B, D>) -> Reader<Data<bool, D>> {
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
    fn bool_from_data<const D: usize>(data: Data<bool, D>, device: &Device<B>) -> BoolTensor<B, D>;

    /// Converts bool tensor to int tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The int tensor with the same data as the bool tensor.
    fn bool_into_int<const D: usize>(tensor: BoolTensor<B, D>) -> IntTensor<B, D>;

    /// Converts bool tensor to float tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The float tensor with the same data as the bool tensor.
    fn bool_into_float<const D: usize>(tensor: BoolTensor<B, D>) -> FloatTensor<B, D>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn bool_device<const D: usize>(tensor: &BoolTensor<B, D>) -> Device<B>;

    /// Moves the tensor to the device.
    fn bool_to_device<const D: usize>(
        tensor: BoolTensor<B, D>,
        device: &Device<B>,
    ) -> BoolTensor<B, D>;

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
        tensor: BoolTensor<B, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<B, D2>;

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
        tensor: BoolTensor<B, D1>,
        ranges: [Range<usize>; D2],
    ) -> BoolTensor<B, D1>;

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
        tensor: BoolTensor<B, D1>,
        ranges: [Range<usize>; D2],
        value: BoolTensor<B, D1>,
    ) -> BoolTensor<B, D1>;

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
        tensor: BoolTensor<B, D>,
        dim: usize,
        times: usize,
    ) -> BoolTensor<B, D> {
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
    fn bool_cat<const D: usize>(tensors: Vec<BoolTensor<B, D>>, dim: usize) -> BoolTensor<B, D> {
        cat_with_slice_assign::<B, D, Bool>(
            tensors
                .into_iter()
                .map(Tensor::<B, D, Bool>::from_primitive)
                .collect(),
            dim,
        )
        .into_primitive()
    }

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
    fn bool_equal<const D: usize>(lhs: BoolTensor<B, D>, rhs: BoolTensor<B, D>)
        -> BoolTensor<B, D>;

    /// Element-wise non-equality comparison.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The tensor with the result of the comparison.
    fn bool_not_equal<const D: usize>(
        lhs: BoolTensor<B, D>,
        rhs: BoolTensor<B, D>,
    ) -> BoolTensor<B, D> {
        let equal_tensor = B::bool_equal(lhs, rhs);
        B::bool_not(equal_tensor)
    }

    /// Inverses boolean values.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The tensor with the result of the negation.
    fn bool_not<const D: usize>(tensor: BoolTensor<B, D>) -> BoolTensor<B, D>;

    /// Transposes a bool tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to transpose.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    fn bool_transpose<const D: usize>(tensor: BoolTensor<B, D>) -> BoolTensor<B, D> {
        Self::bool_swap_dims(tensor, D - 2, D - 1)
    }

    /// Swaps two dimensions of a bool tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to swap the dimensions of.
    /// * `dim1` - The first dimension to swap.
    /// * `dim2` - The second dimension to swap.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions swapped.
    fn bool_swap_dims<const D: usize>(
        tensor: BoolTensor<B, D>,
        dim1: usize,
        dim2: usize,
    ) -> BoolTensor<B, D>;

    /// Permutes the dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to permute the dimensions of.
    /// * `axes` - The new order of the dimensions.
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    fn bool_permute<const D: usize>(tensor: BoolTensor<B, D>, axes: [usize; D])
        -> BoolTensor<B, D>;

    /// Reverse the order of elements in a tensor along the given axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to reverse.
    /// * `axes` - The axes to reverse.
    ///
    /// The tensor with the elements reversed.
    fn bool_flip<const D: usize>(tensor: BoolTensor<B, D>, axes: &[usize]) -> BoolTensor<B, D>;

    /// Returns a new tensor with the given dimension narrowed to the given range.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which the tensor will be narrowed.
    /// * `start` - The starting point of the given range.
    /// * `length` - The ending point of the given range.
    /// # Panics
    ///
    /// - If the dimension is greater than the number of dimensions of the tensor.
    /// - If the given range exceeds the number of elements on the given dimension.
    ///
    /// # Returns
    ///
    /// A new tensor with the given dimension narrowed to the given range.
    fn bool_narrow<const D: usize>(
        tensor: BoolTensor<B, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> BoolTensor<B, D> {
        narrow::<B, D, Bool>(tensor, dim, start, length)
    }

    /// Split the tensor along the given dimension into chunks.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `chunks` - The number of chunks to be produced
    /// * `times` - The dimension along which the tensor will be split.
    ///
    /// # Returns
    ///
    /// A vector of tensors
    fn bool_chunk<const D: usize>(
        tensor: BoolTensor<B, D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<BoolTensor<B, D>> {
        chunk::<B, D, Bool>(tensor, chunks, dim)
    }

    /// Tests if any element in the boolean `tensor` evaluates to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, True if any element in the tensor is True, False otherwise.
    fn bool_any<const D: usize>(tensor: BoolTensor<B, D>) -> BoolTensor<B, 1> {
        let sum = B::int_sum(B::bool_into_int(tensor));
        B::int_greater_elem(sum, 0.elem())
    }

    /// Tests if any element in the boolean `tensor` evaluates to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, D, Bool>` with the same size as input `tensor`, except in the `dim` axis
    /// where the size is 1. The elem in the `dim` axis is True if any element along this dim in the input
    /// evaluates to True, False otherwise.

    fn bool_any_dim<const D: usize>(tensor: BoolTensor<B, D>, dim: usize) -> BoolTensor<B, D> {
        let sum = B::int_sum_dim(B::bool_into_int(tensor), dim);
        B::int_greater_elem(sum, 0.elem())
    }

    /// Tests if all elements in the boolean `tensor` evaluate to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, 1, Bool>` with a single element, True if all elements in the input tensor
    /// evaluate to True, False otherwise.
    fn bool_all<const D: usize>(tensor: BoolTensor<B, D>) -> BoolTensor<B, 1> {
        let num_elems = B::bool_shape(&tensor).num_elements();
        let sum = B::int_sum(B::bool_into_int(tensor));
        B::int_equal_elem(sum, (num_elems as i32).elem())
    }

    /// Tests if all elements in the boolean `tensor` evaluate to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, D, Bool>` with the same size as input `tensor`, except in the `dim` axis
    /// where the size is 1. The elem in the `dim` axis is True if all elements along this dim in the input
    /// evaluates to True, False otherwise.

    fn bool_all_dim<const D: usize>(tensor: BoolTensor<B, D>, dim: usize) -> BoolTensor<B, D> {
        let num_elems = B::bool_shape(&tensor).dims[dim];
        let sum = B::int_sum_dim(B::bool_into_int(tensor), dim);
        B::int_equal_elem(sum, (num_elems as i32).elem())
    }

    /// Compute the indices of the elements that are non-zero, grouped by element.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A vector of tensors, one for each dimension of the given tensor, containing the indices of
    /// the non-zero elements in that dimension.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn bool_argwhere<const D: usize>(tensor: BoolTensor<B, D>) -> IntTensor<B, 2> {
        argwhere::<B, D>(tensor)
    }

    /// Compute the indices of the elements that are non-zero.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor.
    ///
    /// # Returns
    ///
    /// A vector of tensors, one for each dimension of the given tensor, containing the indices of
    /// the non-zero elements in that dimension.
    #[cfg(any(feature = "wasm-sync", not(target_family = "wasm")))]
    fn bool_nonzero<const D: usize>(tensor: BoolTensor<B, D>) -> Vec<IntTensor<B, 1>> {
        let indices = B::bool_argwhere(tensor);
        let dims = B::int_shape(&indices).dims;
        B::int_chunk(indices, dims[1], 1)
            .into_iter()
            .map(|t| B::int_reshape(t, Shape::new([dims[0]])))
            .collect()
    }

    /// Broadcasts the bool `tensor` to the given `shape`.
    fn bool_expand<const D1: usize, const D2: usize>(
        tensor: BoolTensor<B, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<B, D2>;
}
