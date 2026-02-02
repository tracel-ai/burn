use alloc::vec::Vec;
use burn_std::{DType, Shape, Slice};

use crate::{
    Backend, ExecutionError, Scalar, TensorData, TensorMetadata,
    element::Element,
    ops::TransactionPrimitive,
    tensor::{IndexingUpdateOp, IntTensor, TensorKind},
};

/// Trait that list all operations that can be applied on all tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by the
#[cfg_attr(doc, doc = crate::doc_tensor!())]
#[cfg_attr(not(doc), doc = "`Tensor`")]
/// struct.
pub trait BasicOps<B: Backend>: TensorKind<B> {
    /// The type of the tensor elements.
    type Elem: Element;

    /// Creates an empty tensor with the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// The empty tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating empty tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("empty"))]
    #[cfg_attr(not(doc), doc = "`Tensor::empty`")]
    /// function, which is more high-level and designed for public use.
    fn empty(shape: Shape, device: &B::Device, dtype: DType) -> Self::Primitive;

    /// Creates a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// The tensor filled with zeros.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor filled with zeros, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("zeros"))]
    #[cfg_attr(not(doc), doc = "`Tensor::zeros`")]
    /// function, which is more high-level and designed for public use.
    fn zeros(shape: Shape, device: &B::Device, dtype: DType) -> Self::Primitive;

    /// Creates a tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// The tensor filled with ones.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor filled with ones, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("ones"))]
    #[cfg_attr(not(doc), doc = "`Tensor::ones`")]
    /// function, which is more high-level and designed for public use.
    fn ones(shape: Shape, device: &B::Device, dtype: DType) -> Self::Primitive;

    /// Creates a tensor of the given shape where each element is equal to the provided value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `fill_value` - The value with which to fill the tensor.
    /// * `device` - The device on which the tensor will be allocated.
    /// * `dtype` - The target data type.
    ///
    /// # Returns
    ///
    /// The tensor filled with the specified value.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating full tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("full"))]
    #[cfg_attr(not(doc), doc = "`Tensor::full`")]
    /// function, which is more high-level and designed for public use.
    fn full(shape: Shape, fill_value: Scalar, device: &B::Device, dtype: DType) -> Self::Primitive;

    /// Reshapes the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `shape` - The new shape of the tensor.
    ///
    /// # Returns
    ///
    /// The reshaped tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For reshaping a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("reshape"))]
    #[cfg_attr(not(doc), doc = "`Tensor::reshape`")]
    /// function, which is more high-level and designed for public use.
    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive;

    /// Transposes a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to transpose.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    fn transpose(tensor: Self::Primitive) -> Self::Primitive;

    /// Swaps two dimensions of a tensor.
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
    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive;

    /// Permutes the dimensions of a tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to permute the dimensions of.
    /// * `axes` - The new order of the dimensions.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive;

    /// Flips the tensor along the given axes.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to flip.
    /// * `axes` - The axes to flip the tensor along.
    ///
    /// # Returns
    ///
    /// The tensor with the axes flipped.
    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive;

    ///  Select tensor elements corresponding to the given slices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `slices` - The slices specifying ranges and steps for each dimension.
    ///
    /// # Returns
    ///
    /// The selected elements.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For selecting elements of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("slice"))]
    #[cfg_attr(not(doc), doc = "`Tensor::slice`")]
    /// function, which is more high-level and designed for public use.
    fn slice(tensor: Self::Primitive, slices: &[Slice]) -> Self::Primitive;

    /// Assigns the given value to the tensor elements corresponding to the given slices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `slices` - The slices specifying which elements to assign, including support for steps.
    /// * `value` - The value to assign.
    ///
    /// # Returns
    ///
    /// The tensor with the assigned values.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For assigning values to elements of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("slice_assign"))]
    #[cfg_attr(not(doc), doc = "`Tensor::slice_assign`")]
    /// function, which is more high-level and designed for public use.
    fn slice_assign(
        tensor: Self::Primitive,
        slices: &[Slice],
        value: Self::Primitive,
    ) -> Self::Primitive;

    /// Select tensor elements along the given dimension corresponding to the given indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select from.
    /// * `dim` - The dimension along which to select.
    /// * `indices` - The indices of the elements to select.
    ///
    /// # Returns
    ///
    /// The selected tensor elements.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For selecting elements from a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("select"))]
    #[cfg_attr(not(doc), doc = "`Tensor::select`")]
    /// function, which is more high-level and designed for public use.
    fn select(tensor: Self::Primitive, dim: usize, indices: IntTensor<B>) -> Self::Primitive;

    /// Assign the selected elements along the given dimension corresponding to the given indices
    /// from the value tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to assign elements to.
    /// * `dim` - The axis along which to assign elements.
    /// * `indices` - The indices of the elements to assign.
    /// * `values` - The values to assign to the tensor.
    /// * `update` - The operation used to update the existing values at the indexed positions (e.g., add).
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is taken from the
    /// corresponding element of the input tensor at the corresponding index along the specified axis,
    /// except for the elements at the specified indices, which are taken from the corresponding
    /// element of the values tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For assigning elements to a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("select_assign"))]
    #[cfg_attr(not(doc), doc = "`Tensor::select_assign`")]
    /// function, which is more high-level and designed for public use.
    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: IntTensor<B>,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive;

    /// Selects elements from a tensor based on a boolean mask.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select elements from if the corresponding element of the mask is true.
    /// * `mask` - The boolean mask to use for selecting elements.
    /// * `source` - The tensor to select elements from when the corresponding element of the mask is false.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensors, where each element is taken from the
    /// corresponding element of the left hand side tensor if the corresponding element of the mask
    /// is true, and from the corresponding element of the right hand side tensor otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For selecting elements from a tensor based on a boolean mask, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("mask_where"))]
    #[cfg_attr(not(doc), doc = "`Tensor::mask_where`")]
    /// function, which is more high-level and designed for public use.
    fn mask_where(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        source: Self::Primitive,
    ) -> Self::Primitive;

    /// Fills elements of a tensor based on a boolean mask.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor where will be overwritten with the value
    ///   when the corresponding element of the mask is true.
    /// * `mask` - The boolean mask to use for filling elements.
    /// * `value` - The value to fill elements with when the corresponding element of the mask is true.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensors, where each element is taken from the
    /// corresponding element unmodified if the corresponding element of the mask is false, and
    /// filled with the value otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For filling elements of a tensor based on a boolean mask, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("mask_fill"))]
    #[cfg_attr(not(doc), doc = "`Tensor::mask_fill`")]
    /// function, which is more high-level and designed for public use.
    fn mask_fill(
        tensor: Self::Primitive,
        mask: B::BoolTensorPrimitive,
        value: Scalar,
    ) -> Self::Primitive;

    /// Gathers elements from a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to gather elements.
    /// * `tensor` - The tensor to gather elements from.
    /// * `indices` - The indices of the elements to gather.
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is taken from the
    /// corresponding element of the input tensor at the corresponding index along the specified axis.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For gathering elements from a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("gather"))]
    #[cfg_attr(not(doc), doc = "`Tensor::gather`")]
    /// function, which is more high-level and designed for public use.
    fn gather(dim: usize, tensor: Self::Primitive, indices: IntTensor<B>) -> Self::Primitive;

    /// Scatters elements into a tensor along an axis.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to scatter elements.
    /// * `tensor` - The tensor to scatter elements into.
    /// * `indices` - The indices of the elements to scatter.
    /// * `values` - The values to scatter into the tensor.
    /// * `update` - The operation used to update the existing values at the indexed positions (e.g., add).
    ///
    /// # Returns
    ///
    /// A tensor with the same shape as the input tensor, where each element is taken from the
    /// corresponding element of the input tensor at the corresponding index along the specified axis,
    /// except for the elements at the specified indices, which are taken from the corresponding
    /// element of the values tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For scattering elements into a tensor along an axis, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("scatter"))]
    #[cfg_attr(not(doc), doc = "`Tensor::scatter`")]
    /// function, which is more high-level and designed for public use.
    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: IntTensor<B>,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive;

    /// Returns the device on which the tensor is allocated.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The device on which the tensor is allocated.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the device of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("device"))]
    #[cfg_attr(not(doc), doc = "`Tensor::device`")]
    /// function, which is more high-level and designed for public use.
    fn device(tensor: &Self::Primitive) -> B::Device;

    /// Moves the tensor to the given device.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `device` - The device on which the tensor will be moved.
    ///
    /// # Returns
    ///
    /// The tensor on the given device.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For moving a tensor to a device, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("to_device"))]
    #[cfg_attr(not(doc), doc = "`Tensor::to_device`")]
    /// function, which is more high-level and designed for public use.
    #[allow(clippy::wrong_self_convention)]
    fn to_device(tensor: Self::Primitive, device: &B::Device) -> Self::Primitive;

    /// Extracts the data from the tensor asynchronously.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The data of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For extracting the data of a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("into_data"))]
    #[cfg_attr(not(doc), doc = "`Tensor::into_data`")]
    /// function, which is more high-level and designed for public use.
    #[allow(clippy::wrong_self_convention)]
    fn into_data_async(
        tensor: Self::Primitive,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send;

    /// Read the data from the tensor using a transaction.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    fn register_transaction(tr: &mut TransactionPrimitive<B>, tensor: Self::Primitive);

    /// Creates a tensor from the given data.
    ///
    /// # Arguments
    ///
    /// * `data` - The data of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// The tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor from data, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("from_data"))]
    #[cfg_attr(not(doc), doc = "`Tensor::from_data`")]
    /// function, which is more high-level and designed for public use.
    fn from_data(data: TensorData, device: &B::Device) -> Self::Primitive;
    /// Creates a tensor from the given data enforcing the given data type.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor from data, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("from_data_dtype"))]
    #[cfg_attr(not(doc), doc = "`Tensor::from_data_dtype`")]
    /// function, which is more high-level and designed for public use.
    fn from_data_dtype(data: TensorData, device: &B::Device, dtype: DType) -> Self::Primitive;

    /// Repeat the tensor along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `dim` - The dimension along which the tensor will be repeated.
    /// * `times` - The number of times the tensor will be repeated.
    ///
    /// # Returns
    ///
    /// The repeated tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For repeating a tensor, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("repeat_dim"))]
    #[cfg_attr(not(doc), doc = "`Tensor::repeat_dim`")]
    /// function, which is more high-level and designed for public use.
    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive;

    /// Concatenates the given tensors along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `vectors` - The tensors to concatenate.
    /// * `dim` - The dimension along which the tensors will be concatenated.
    ///
    /// # Returns
    ///
    /// The concatenated tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For concatenating tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("cat"))]
    #[cfg_attr(not(doc), doc = "`Tensor::cat`")]
    /// function, which is more high-level and designed for public use.
    fn cat(vectors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive;

    /// Equates the given tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The tensor of booleans indicating whether the corresponding elements are equal.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For equating tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("equal"))]
    #[cfg_attr(not(doc), doc = "`Tensor::equal`")]
    /// function, which is more high-level and designed for public use.
    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive;

    /// Element-wise equality between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding elements of the input tensors are equal, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise equality between two tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("equal_elem"))]
    #[cfg_attr(not(doc), doc = "`Tensor::equal_elem`")]
    /// function, which is more high-level and designed for public use.
    fn equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive;

    /// Applies element-wise non-equality comparison between the given tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side tensor.
    ///
    /// # Returns
    ///
    /// The tensor of booleans indicating whether the corresponding elements are equal.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For non-equality comparison of tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("not_equal"))]
    #[cfg_attr(not(doc), doc = "`Tensor::not_equal`")]
    /// function, which is more high-level and designed for public use.
    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> B::BoolTensorPrimitive;

    /// Element-wise non-equality between two tensors.
    ///
    /// # Arguments
    ///
    /// * `lhs` - The left hand side tensor.
    /// * `rhs` - The right hand side scalar.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input tensors, where each element is true if the
    /// corresponding elements of the input tensors are equal, and false otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For element-wise non-equality between two tensors, users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("not_equal_elem"))]
    #[cfg_attr(not(doc), doc = "`Tensor::not_equal_elem`")]
    /// function, which is more high-level and designed for public use.
    fn not_equal_elem(lhs: Self::Primitive, rhs: Scalar) -> B::BoolTensorPrimitive;

    /// Returns the name of the element type.
    fn elem_type_name() -> &'static str {
        core::any::type_name::<Self::Elem>()
    }

    /// Returns the tensor data type.
    fn dtype(tensor: &Self::Primitive) -> DType {
        tensor.dtype()
    }

    /// Tests if any element in the `tensor` evaluates to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, True if any element in the input tensor evaluates to True, False otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly. Users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("any"))]
    #[cfg_attr(not(doc), doc = "`Tensor::any`")]
    /// function, which is more high-level and designed for public use.
    fn any(tensor: Self::Primitive) -> B::BoolTensorPrimitive;

    /// Tests if any element in the tensor evaluates to True along a given dimension dim.
    ///
    /// # Arguments
    ///
    /// * tensor - The tensor to test.
    /// * dim - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same size as input tensor, except in the dim axis where the size is 1.
    /// Returns True if any element in the input tensor along the given dimension evaluates to True, False otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly. Users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("any_dim"))]
    #[cfg_attr(not(doc), doc = "`Tensor::any_dim`")]
    /// function, which is more high-level and designed for public use.
    fn any_dim(tensor: Self::Primitive, dim: usize) -> B::BoolTensorPrimitive;

    /// Tests if all elements in the `tensor` evaluate to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, True if all elements in the input tensor evaluates to True, False otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly. Users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("all"))]
    #[cfg_attr(not(doc), doc = "`Tensor::all`")]
    /// function, which is more high-level and designed for public use.
    fn all(tensor: Self::Primitive) -> B::BoolTensorPrimitive;

    /// Tests if all elements in the `tensor` evaluate to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same size as input `tensor`, except in the `dim` axis where the size is 1.
    /// Returns True if all elements in the input tensor along the given dimension evaluate to True, False otherwise.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly. Users should prefer the
    #[cfg_attr(doc, doc = crate::doc_tensor!("all_dim"))]
    #[cfg_attr(not(doc), doc = "`Tensor::all_dim`")]
    /// function, which is more high-level and designed for public use.
    fn all_dim(tensor: Self::Primitive, dim: usize) -> B::BoolTensorPrimitive;

    /// Broadcasts the given tensor to the specified shape.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to broadcast.
    /// * `shape` - The shape to broadcast to.
    ///
    /// # Returns
    ///
    /// The broadcasted tensor.
    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive;

    /// Unfold windows along a dimension.
    ///
    /// Returns a view of the tensor with all complete windows of size `size` in dimension `dim`;
    /// where windows are advanced by `step` at each index.
    ///
    /// The number of windows is `max(0, (shape[dim] - size).ceil_div(step))`.
    ///
    /// # Warning
    ///
    /// For the `ndarray` and `candle` backends; this is not a view but a full copy.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The input tensor to unfold; of shape ``[pre=..., dim shape, post=...]``
    /// * `dim` - the dimension to unfold.
    /// * `size` - the size of each unfolded window.
    /// * `step` - the step between each window.
    ///
    /// # Returns
    ///
    /// A tensor view with shape ``[pre=..., windows, post=..., size]``.
    fn unfold(tensor: Self::Primitive, dim: usize, size: usize, step: usize) -> Self::Primitive;
}
