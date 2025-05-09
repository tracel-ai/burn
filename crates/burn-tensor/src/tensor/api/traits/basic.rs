use crate::{DType, Shape, TensorData, TensorKind, TensorMetadata, Transaction, backend::Backend};

use super::{AssignOps, ComparisonOps, CreationOps, ReductionOps, ViewOps};

// BaseOps: CreationOps + ViewOps + AssignOps {
//     // also define other base ops here like scatter, gather, select, flip, etc.
// }

/// Trait that list all operations that can be applied on all tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait BasicOps<B: Backend>:
    TensorKind<B> + CreationOps<B> + AssignOps<B> + ComparisonOps<B> + ReductionOps<B> + ViewOps<B>
{
    /// Returns the name of the element type.
    fn elem_type_name() -> &'static str {
        core::any::type_name::<Self::Elem>()
    }

    /// Returns the tensor data type.
    fn dtype(tensor: &Self::Primitive) -> DType {
        tensor.dtype()
    }

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
    /// For getting the device of a tensor, users should prefer the [Tensor::device](Tensor::device) function,
    /// which is more high-level and designed for public use.
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
    /// For moving a tensor to a device, users should prefer the [Tensor::to_device](Tensor::to_device) function,
    /// which is more high-level and designed for public use.
    fn to_device(tensor: Self::Primitive, device: &B::Device) -> Self::Primitive;

    /// Read the data from the tensor using a transaction.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    fn register_transaction(tr: &mut Transaction<B>, tensor: Self::Primitive);

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
    /// For creating a tensor from data, users should prefer the [Tensor::from_data](Tensor::from_data) function,
    /// which is more high-level and designed for public use.
    fn from_data(data: TensorData, device: &B::Device) -> Self::Primitive;

    /// Creates a tensor from the given data enforcing the given data type.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor from data, users should prefer the [Tensor::from_data_dtype](Tensor::from_data_dtype)
    /// function, which is more high-level and designed for public use.
    fn from_data_dtype(data: TensorData, device: &B::Device, dtype: DType) -> Self::Primitive;

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
    /// For extracting the data of a tensor, users should prefer the [Tensor::into_data](Tensor::into_data) function,
    /// which is more high-level and designed for public use.
    fn into_data_async(tensor: Self::Primitive) -> impl Future<Output = TensorData> + Send;

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
    /// For reshaping a tensor, users should prefer the [Tensor::reshape](Tensor::reshape) function,
    /// which is more high-level and designed for public use.
    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive;

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
    /// For repeating a tensor, users should prefer the [Tensor::repeat_dim](Tensor::repeat_dim) function,
    /// which is more high-level and designed for public use.
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
    /// For concatenating tensors, users should prefer the [Tensor::cat](Tensor::cat) function,
    /// which is more high-level and designed for public use.
    fn cat(vectors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive;

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
    /// [Tensor::gather](Tensor::gather) function, which is more high-level and designed for public use.
    fn gather(
        dim: usize,
        tensor: Self::Primitive,
        indices: B::IntTensorPrimitive,
    ) -> Self::Primitive;

    /// Select tensor elements along the given dimension corresponding for the given indices.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to select elements from.
    /// * `dim` - The axis along which to select elements.
    /// * `indices` - The indices of the elements to select.
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
    /// For selecting elements from a tensor along an axis, users should prefer the
    /// [Tensor::select](Tensor::select) function, which is more high-level and designed for public use.
    fn select(
        tensor: Self::Primitive,
        dim: usize,
        indices: B::IntTensorPrimitive,
    ) -> Self::Primitive;
}
