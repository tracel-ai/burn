use crate::{Distribution, ElementConversion, Shape, backend::Backend};

use super::TensorKind;

/// Trait that list common creation operations that can be applied on all tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait CreationOps<B: Backend>: TensorKind<B> {
    /// Creates an empty tensor with the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
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
    /// For creating empty tensors, users should prefer the [Tensor::empty](Tensor::empty) function,
    /// which is more high-level and designed for public use.
    fn empty(shape: Shape, device: &B::Device) -> Self::Primitive;

    /// Creates a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
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
    /// For creating a tensor filled with zeros, users should prefer the [Tensor::zeros](Tensor::zeros) function,
    /// which is more high-level and designed for public use.
    fn zeros(shape: Shape, device: &B::Device) -> Self::Primitive;

    /// Creates a tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `device` - The device on which the tensor will be allocated.
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
    /// For creating a tensor filled with ones, users should prefer the [Tensor::ones](Tensor::ones) function,
    /// which is more high-level and designed for public use.
    fn ones(shape: Shape, device: &B::Device) -> Self::Primitive;

    /// Creates a tensor filled with elements equal to the given value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `fill_value` - The value with which to fill the tensor
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// The tensor filled with elements equal to the given value
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For creating a tensor filled with a specific value, users should prefer the [Tensor::full](Tensor::full) function,
    /// which is more high-level and designed for public use.
    fn full<E: ElementConversion>(
        shape: Shape,
        fill_value: E,
        device: &B::Device,
    ) -> Self::Primitive;
}

/// Trait that list common creation operations that can be applied on numerical tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait NumericCreationOps<B: Backend>: TensorKind<B> {
    /// Create a random tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the output tensor.
    /// * `distribution` - The distribution used to sample.
    /// * `device` - The device to use.
    ///
    /// # Returns
    ///
    /// A new tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// Users should prefer the [Tensor::random](Tensor::random) function,
    /// which is more high-level and designed for public use.
    fn random(shape: Shape, distribution: Distribution, device: &B::Device) -> Self::Primitive;

    // TODO: arange
}
// arange?
