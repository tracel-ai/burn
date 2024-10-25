use super::{
    cat::cat_with_slice_assign, repeat_dim::repeat_with_slice_assign, BoolTensor, Device,
    FloatTensor, IntTensor,
};
use crate::{
    argwhere_data, backend::Backend, chunk, narrow, tensor::Shape, Bool, ElementConversion,
    TensorData,
};
use alloc::{vec, vec::Vec};
use core::{future::Future, ops::Range};

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
    fn bool_empty(shape: Shape, device: &Device<B>) -> BoolTensor<B>;

    /// Returns the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    fn bool_shape(tensor: &BoolTensor<B>) -> Shape;

    /// Converts the tensor to a data structure.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data structure with the tensor's data.
    fn bool_into_data(tensor: BoolTensor<B>) -> impl Future<Output = TensorData> + Send;

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
    fn bool_from_data(data: TensorData, device: &Device<B>) -> BoolTensor<B>;

    /// Converts bool tensor to int tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The int tensor with the same data as the bool tensor.
    fn bool_into_int(tensor: BoolTensor<B>) -> IntTensor<B>;

    /// Converts bool tensor to float tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The float tensor with the same data as the bool tensor.
    fn bool_into_float(tensor: BoolTensor<B>) -> FloatTensor<B>;

    /// Gets the device of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device of the tensor.
    fn bool_device(tensor: &BoolTensor<B>) -> Device<B>;

    /// Moves the tensor to the device.
    fn bool_to_device(tensor: BoolTensor<B>, device: &Device<B>) -> BoolTensor<B>;

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
    fn bool_reshape(tensor: BoolTensor<B>, shape: Shape) -> BoolTensor<B>;

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
    fn bool_slice(tensor: BoolTensor<B>, ranges: &[Range<usize>]) -> BoolTensor<B>;

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
    fn bool_slice_assign(
        tensor: BoolTensor<B>,
        ranges: &[Range<usize>],
        value: BoolTensor<B>,
    ) -> BoolTensor<B>;

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
    fn bool_repeat_dim(tensor: BoolTensor<B>, dim: usize, times: usize) -> BoolTensor<B> {
        repeat_with_slice_assign::<B, Bool>(tensor, dim, times)
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
    fn bool_cat(tensors: Vec<BoolTensor<B>>, dim: usize) -> BoolTensor<B> {
        cat_with_slice_assign::<B, Bool>(tensors, dim)
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
    fn bool_equal(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B>;

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
    fn bool_not_equal(lhs: BoolTensor<B>, rhs: BoolTensor<B>) -> BoolTensor<B> {
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
    fn bool_not(tensor: BoolTensor<B>) -> BoolTensor<B>;

    /// Transposes a bool tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to transpose.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    fn bool_transpose(tensor: BoolTensor<B>) -> BoolTensor<B> {
        let ndims = Self::bool_shape(&tensor).num_dims();
        Self::bool_swap_dims(tensor, ndims - 2, ndims - 1)
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
    fn bool_swap_dims(tensor: BoolTensor<B>, dim1: usize, dim2: usize) -> BoolTensor<B>;

    /// Permutes the dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to permute the dimensions of.
    /// * `axes` - The new order of the dimensions.
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    fn bool_permute(tensor: BoolTensor<B>, axes: &[usize]) -> BoolTensor<B>;

    /// Reverse the order of elements in a tensor along the given axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to reverse.
    /// * `axes` - The axes to reverse.
    ///
    /// The tensor with the elements reversed.
    fn bool_flip(tensor: BoolTensor<B>, axes: &[usize]) -> BoolTensor<B>;

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
    fn bool_narrow(
        tensor: BoolTensor<B>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> BoolTensor<B> {
        narrow::<B, Bool>(tensor, dim, start, length)
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
    fn bool_chunk(tensor: BoolTensor<B>, chunks: usize, dim: usize) -> Vec<BoolTensor<B>> {
        chunk::<B, Bool>(tensor, chunks, dim)
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
    fn bool_any(tensor: BoolTensor<B>) -> BoolTensor<B> {
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

    fn bool_any_dim(tensor: BoolTensor<B>, dim: usize) -> BoolTensor<B> {
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
    fn bool_all(tensor: BoolTensor<B>) -> BoolTensor<B> {
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

    fn bool_all_dim(tensor: BoolTensor<B>, dim: usize) -> BoolTensor<B> {
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
    /// A 2D tensor containing the indices of all non-zero elements of the given tensor.
    /// Each row contains the indices of a non-zero element.
    fn bool_argwhere(tensor: BoolTensor<B>) -> impl Future<Output = IntTensor<B>> + Send {
        async {
            // Size of each output tensor is variable (= number of nonzero elements in the tensor).
            // Reading the data to count the number of truth values might cause sync but is required.
            let device = B::bool_device(&tensor);
            let data = B::bool_into_data(tensor).await;
            argwhere_data::<B>(data, &device)
        }
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
    /// the non-zero elements in that dimension. If all elements are zero, the vector is empty.
    fn bool_nonzero(tensor: BoolTensor<B>) -> impl Future<Output = Vec<IntTensor<B>>> + Send {
        async {
            let indices = B::bool_argwhere(tensor).await;

            if B::int_shape(&indices).num_elements() == 0 {
                // Return empty vec when all elements are zero
                return vec![];
            }

            let dims = B::int_shape(&indices).dims;
            B::int_chunk(indices, dims[1], 1)
                .into_iter()
                .map(|t| B::int_reshape(t, Shape::new([dims[0]])))
                .collect()
        }
    }

    /// Broadcasts the bool `tensor` to the given `shape`.
    fn bool_expand(tensor: BoolTensor<B>, shape: Shape) -> BoolTensor<B>;
}
