#![allow(clippy::single_range_in_vec_init)]
use crate::backend::ExecutionError;

pub use burn_backend::tensor::BasicOps;

use alloc::vec::Vec;

use alloc::format;
use alloc::string::String;
use alloc::vec;

use burn_std::stub::RwLock;
use core::iter::repeat;
use core::{fmt::Debug, ops::Range};
use serde::{Deserialize, Deserializer};

use serde::{Serialize, Serializer};

use crate::IndexingUpdateOp;
use crate::{AsIndex, Slice, SliceArg, canonicalize_dim, wrap_index};
use crate::{
    Bool, ElementConversion, Float, Int, Shape, TensorData, TensorKind, TensorMetadata,
    backend::Backend, check,
};
use crate::{DType, Element};
use crate::{cast::ToElement, check::TensorCheck};

/// A tensor with a given backend, shape and data type.
///
/// # Indexing
/// Indexing a tensor can be done using [`slice`](Tensor::slice) for all tensor types
/// or [`select`](Tensor::select) for numeric types.
///
/// ## Example
///
/// ```rust
/// use burn_tensor::backend::Backend;
/// use burn_tensor::Tensor;
/// use burn_tensor::Int;
///
/// fn example<B: Backend>() {
///     let device = Default::default();
///
///     let tensor = Tensor::<B, 2>::from_data(
///         [
///             [3.0, 4.9, 2.0],
///             [2.0, 1.9, 3.0],
///             [6.0, 1.5, 7.0],
///             [3.0, 4.9, 9.0],
///         ],
///         &device,
///     );
///
///     // Slice the tensor to get the second and third rows:
///     // [[2.0, 1.9, 3.0], [6.0, 1.5, 7.0]]
///     // The resulting tensor will have dimensions [2, 3].
///     let slice = tensor.clone().slice([1..3]);
///     println!("{slice}");
///
///     // Slice the tensor to get the first two rows and the first 2 columns:
///     // [[3.0, 4.9], [2.0, 1.9]]
///     // The resulting tensor will have dimensions [2, 2].
///     let slice = tensor.clone().slice([0..2, 0..2]);
///     println!("{slice}");
///
///     // Index the tensor along the dimension 1 to get the elements 0 and 2:
///     // [[3.0, 2.0], [2.0, 3.0], [6.0, 7.0], [3.0, 9.0]]
///     // The resulting tensor will have dimensions [4, 2]
///     let indices = Tensor::<B, 1, Int>::from_data([0, 2], &device);
///     let indexed = tensor.select(1, indices);
///     println!("{indexed}");
/// }
/// ```
#[derive(new, Clone, Debug)]
pub struct Tensor<B, const D: usize, K = Float>
where
    B: Backend,
    K: TensorKind<B>,
{
    pub(crate) primitive: K::Primitive,
}

impl<B, const D: usize, K, T> From<T> for Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
    T: Into<TensorData>,
{
    fn from(value: T) -> Self {
        Tensor::from_data(value.into(), &Default::default())
    }
}

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
    K::Elem: Element,
{
    /// Executes an operation on the tensor and modifies its value.
    ///
    /// # Notes
    ///
    /// This won't necessarily reuse the same tensor data/buffer, but it should if there is
    /// no other reference pointing to the same tensor.
    ///
    /// Wrapping operations with inplace is not an optimization, it's mainly there if you
    /// want to mutate a tensor by using owned operations. A plausible usage would be to
    /// update the weights of a mutable model reference.
    pub fn inplace<F: FnOnce(Self) -> Self>(&mut self, func: F) {
        let mut tensor_owned = Tensor::empty([0; D], &self.device());
        core::mem::swap(&mut tensor_owned, self);

        let mut tensor_new = func(tensor_owned);
        core::mem::swap(&mut tensor_new, self);
    }

    /// Converts the tensor into a primitive tensor.
    pub fn into_primitive(self) -> K::Primitive {
        self.primitive
    }

    /// Converts from a primitive tensor into a tensor.
    pub fn from_primitive(tensor: K::Primitive) -> Self {
        Self::new(tensor)
    }

    /// Returns the number of dimensions of the tensor.
    pub fn rank(&self) -> usize {
        self.primitive.rank()
    }

    /// Returns the tensor primitive data type.
    ///
    /// # Note
    /// Some element types are encoded in different primitive types depending on the backend
    /// (e.g., bool could be encoded as `u8` or `u32`).
    pub fn dtype(&self) -> DType {
        self.primitive.dtype()
    }

    /// Create an empty tensor of the given shape.
    ///
    /// # Arguments
    ///
    /// - `shape`: The shape of the tensor.
    /// - `device`: The device where the tensor will be created.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    // Create an empty tensor with dimensions [2, 3, 4].
    ///    let tensor = Tensor::<B, 3>::empty([2, 3, 4], &device);
    /// }
    /// ```
    pub fn empty<S: Into<Shape>>(shape: S, device: &B::Device) -> Self {
        let shape = shape.into();
        check!(TensorCheck::creation_ops::<D>("Empty", &shape.dims));
        Self::new(K::empty(shape, device, K::Elem::dtype()))
    }

    /// Create a tensor of the given shape where each element is zero.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::zeros(Shape::new([2, 3]), &device);
    ///    println!("{tensor}");
    ///    // [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    /// }
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S, device: &B::Device) -> Self {
        let shape = shape.into();
        check!(TensorCheck::creation_ops::<D>("Zeros", &shape.dims));
        Self::new(K::zeros(shape, device, K::Elem::dtype()))
    }

    /// Returns a new tensor with the same shape, dtype, and device as the current tensor filled with zeros.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///   let tensor = tensor.zeros_like();
    ///   println!("{tensor}");
    ///   // [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    /// }
    /// ```
    pub fn zeros_like(&self) -> Self {
        Self::new(K::zeros(self.shape(), &self.device(), self.dtype()))
    }

    /// Create a tensor of the given shape where each element is one.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::ones(Shape::new([2, 3]), &device);
    ///   println!("{tensor}");
    ///   // [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    /// }
    /// ```
    pub fn ones<S: Into<Shape>>(shape: S, device: &B::Device) -> Self {
        let shape = shape.into();
        check!(TensorCheck::creation_ops::<D>("Ones", &shape.dims));
        Self::new(K::ones(shape, device, K::Elem::dtype()))
    }

    /// Returns a new tensor with the same shape, dtype, and device as the current tensor filled with ones.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.ones_like();
    ///    println!("{tensor}");
    ///    // [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    /// }
    /// ```
    pub fn ones_like(&self) -> Self {
        Self::new(K::ones(self.shape(), &self.device(), self.dtype()))
    }

    /// Create a tensor of the given shape where each element is equal to the provided value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::full(Shape::new([2, 3]), 5.0, &device);
    ///   println!("{tensor}");
    ///   // [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    /// }
    /// ```
    pub fn full<S: Into<Shape>, E: ElementConversion>(
        shape: S,
        fill_value: E,
        device: &B::Device,
    ) -> Self {
        let shape = shape.into();
        check!(TensorCheck::creation_ops::<D>("Full", &shape.dims));
        Self::new(K::full(shape, fill_value, device, K::Elem::dtype()))
    }

    /// Returns a new tensor with the same shape, dtype, and device as the current tensor,
    /// filled with the provided value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///    let device = B::Device::default();
    ///    let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///    let tensor = tensor.full_like(5.0);
    ///    println!("{tensor}");
    ///    // [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    /// }
    /// ```
    pub fn full_like<E: ElementConversion>(&self, fill_value: E) -> Self {
        Self::new(K::full(
            self.shape(),
            fill_value,
            &self.device(),
            self.dtype(),
        ))
    }

    /// Returns the dimensions of the current tensor.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///   let device = Default::default();
    ///   let tensor = Tensor::<B, 3>::ones([2, 3, 4], &device);
    ///   let dims = tensor.dims(); // [2, 3, 4]
    ///   println!("{dims:?}");
    /// }
    /// ```
    pub fn dims(&self) -> [usize; D] {
        Self::shape(self).dims()
    }

    /// Returns the shape of the current tensor.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 3>::ones([2, 3, 4], &device);
    ///    // Shape { dims: [2, 3, 4] }
    ///    let shape = tensor.shape();
    /// }
    /// ```
    pub fn shape(&self) -> Shape {
        self.primitive.shape()
    }

    /// Reshape the tensor to have the given shape.
    ///
    /// The tensor has the same data and number of elements as the input.
    ///
    /// A `-1` in the shape is used to infer the remaining dimensions, e.g.: `[2, -1]`
    /// will reshape the tensor with [2, 3, 4] dimensions to [2, 12].
    ///
    /// A `0` in the shape instructs to keep the current dimension from the original tensor,
    /// e.g.: `[2, 0, 4]` will reshape the tensor with [2, 3, 4] dimensions to [2, 3, 4].
    /// This is useful when reshaping tensors with unknown dimensions and combining with `-1`
    /// to infer the remaining dimensions, e.g. `[0, -1]` will reshape the tensor
    /// with [1, 3, 4] dimensions to [1, 12].
    ///
    /// # Arguments
    /// - `shape`: The new shape of the tensor.
    ///
    /// # Panics
    /// - If the tensor contains more than one `-1` in the shape.
    /// - If the tensor contains values that are not positive (other than -1).
    /// - If the shape does not match the number of elements of the original shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    // Create a tensor with dimensions [2, 3, 4]
    ///    let tensor = Tensor::<B, 3>::ones([2, 3, 4], &device);
    ///    // Reshape it to [2, 12], where 12 is inferred from the number of elements.
    ///    let reshaped = tensor.reshape([2, -1]);
    ///    println!("{reshaped}");
    /// }
    /// ```
    pub fn reshape<const D2: usize, S: ReshapeArgs<D2>>(self, shape: S) -> Tensor<B, D2, K> {
        // Convert reshape args to shape
        let shape = shape.into_shape(&self);
        Tensor::new(K::reshape(self.primitive, shape))
    }

    /// Transpose the tensor.
    ///
    /// For a 2D tensor, this is the standard matrix transpose. For `D > 2`, the transpose is
    /// applied on the last two dimensions. For example, the transpose of a tensor with shape
    /// `[1, 2, 3, 4]` will have shape `[1, 2, 4, 3]`.
    ///
    /// See also [`permute`](Tensor::permute).
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to transpose.
    ///
    /// # Returns
    ///
    /// The transposed tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 2D tensor of shape [2, 3]
    ///     let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///
    ///     // Transpose the tensor:
    ///     // [[1.0, 5.0], [-2.0, 9.0], [3.0, 6.0]]
    ///     // The resulting tensor will have dimensions [3, 2].
    ///     let transposed = tensor.transpose();
    ///     println!("{transposed}");
    /// }
    /// ```
    pub fn transpose(self) -> Tensor<B, D, K> {
        Tensor::new(K::transpose(self.primitive))
    }

    /// Alias for `transpose`.
    #[inline(always)]
    pub fn t(self) -> Tensor<B, D, K> {
        self.transpose()
    }

    /// Swaps two dimensions of a tensor.
    ///
    /// This is a no-op when `dim1 == dim2`, assuming both are within bounds.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to swap the dimensions of.
    /// * `dim1` - The first dimension to swap, supports negative indexing.
    /// * `dim2` - The second dimension to swap, supports negative indexing.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions swapped.
    ///
    /// # Panics
    ///
    /// When dimensions are out of bounds.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 2D tensor of shape [2, 3]
    ///     let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]], &device);
    ///
    ///     // Swap the dimensions 0 and -1 (equivalent to `tensor.transpose()`):
    ///     // [[1.0, 5.0], [-2.0, 9.0], [3.0, 6.0]]
    ///     // The resulting tensor will have dimensions [3, 2].
    ///     let swapped = tensor.swap_dims(0, -1);
    ///     println!("{swapped}");
    /// }
    /// ```
    pub fn swap_dims<Dim1, Dim2>(self, dim1: Dim1, dim2: Dim2) -> Tensor<B, D, K>
    where
        Dim1: AsIndex,
        Dim2: AsIndex,
    {
        let dim1 = canonicalize_dim(dim1, D, false);
        let dim2 = canonicalize_dim(dim2, D, false);
        check!(TensorCheck::swap_dims::<D>(dim1, dim2));
        if dim1 == dim2 {
            self
        } else {
            Tensor::new(K::swap_dims(self.primitive, dim1, dim2))
        }
    }

    /// Permute the dimensions of the tensor.
    ///
    /// This is a no-op when the resolved `axes` match the current order.
    ///
    /// # Arguments
    ///
    /// * `axes` - The new order of the dimensions. The length of the axes
    ///   must be equal to the number of dimensions of the tensor.
    ///   The values must be unique and in the range of the number of dimensions.
    ///   The values can be negative, in which case they are used as an offset from the end.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 2D tensor of shape [3, 2]
    ///     let tensor = Tensor::<B, 2>::from_data([[1.0, 5.0], [-2.0, 9.0], [3.0, 6.0]], &device);
    ///
    ///     // Permute the dimensions 1 and 0:
    ///     // [[1.0, -2.0, 3.0], [5.0, 9.0, 6.0]]
    ///     // The resulting tensor will have dimensions [3, 2].
    ///     let permuted = tensor.permute([1, 0]);
    ///     println!("{permuted}");
    /// }
    /// ```
    pub fn permute<Dim>(self, axes: [Dim; D]) -> Tensor<B, D, K>
    where
        Dim: AsIndex,
    {
        let mut no_op = true;
        let mut fixed_axes = [0; D];
        for (i, axis) in axes.into_iter().enumerate() {
            let dim = canonicalize_dim(axis, D, false);
            no_op &= dim == i;
            fixed_axes[i] = dim;
        }

        if no_op {
            self
        } else {
            check!(TensorCheck::permute(fixed_axes));
            Tensor::new(K::permute(self.primitive, &fixed_axes))
        }
    }

    /// Moves the dimension(s) of input at the position(s) in source to the position(s) in destination.
    ///
    /// Other dimensions of input that are not explicitly moved remain in their original order and appear
    /// at the positions not specified in destination.
    ///
    /// # Arguments
    ///
    /// * `src` - The dimension(s) to move. The values must be unique and in the range of the number of dimensions.
    ///   The values can be negative, in which case they are used as an offset from the end.
    ///
    /// * `dst` - Destination positions for each of the original dims. These must also be unique.
    ///
    /// # Panics
    ///
    /// - If the source and destination dimensions are not of the same length.
    /// - If the source and destination vectors contain duplicate values.
    /// - If the source and destination vectors contain values that are out of bounds.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions moved.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 3D tensor of shape [3, 2, 1]
    ///     let tensor = Tensor::<B, 3>::from_data([[[1.0], [5.0]], [[-2.0], [9.0]], [[3.0], [6.0]]], &device);
    ///
    ///     // Move the dimensions 0 and 1:
    ///     // [[[1.0], [-2.0], [3.0]], [[5.0], [9.0], [6.0]]]
    ///     // The resulting tensor will have dimensions [2, 3, 1].
    ///     let moved = tensor.movedim(1, 0);
    ///     println!("{moved}");
    /// }
    /// ```
    ///
    /// # Note
    ///
    /// This is a syntactic sugar for `permute`. It is used widely enough, so we define a separate Op
    /// for it
    pub fn movedim<S1: MovedimArgs, S2: MovedimArgs>(self, src: S1, dst: S2) -> Tensor<B, D, K> {
        let source_dims = src.into_dim_vec::<D>();
        let destination_dims = dst.into_dim_vec::<D>();

        check!(TensorCheck::movedim_args_length(
            &source_dims,
            &destination_dims
        ));

        let mut m = [-1; D];
        for (&d, &s) in destination_dims.iter().zip(source_dims.iter()) {
            m[d] = s as isize;
        }
        let mut axes: [isize; D] = [0; D];
        let mut source_i = 0;
        for (dest_i, item) in axes.iter_mut().enumerate().take(D) {
            *item = if m[dest_i] != -1 {
                m[dest_i]
            } else {
                while source_dims.contains(&source_i) {
                    source_i += 1;
                }
                let result = source_i as isize;
                source_i += 1;
                result
            };
        }

        self.permute(axes)
    }

    /// Reverse the order of elements in the tensor along the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `axes` - The dimensions to reverse. The values must be unique and in the range of the number of dimensions.
    ///   The values can be negative, in which case they are used as an offset from the end.
    ///
    /// # Returns
    ///
    /// The tensor with the axes flipped.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 2D tensor with dimensions [4, 3]
    ///     let tensor = Tensor::<B, 2>::from_data(
    ///         [
    ///             [3.0, 4.9, 2.0],
    ///             [2.0, 1.9, 3.0],
    ///             [4.0, 5.9, 8.0],
    ///             [1.4, 5.8, 6.0],
    ///         ],
    ///         &device,
    ///     );
    ///
    ///     // Flip the elements in dimensions 0 and 1:
    ///     // [[6.0, 5.8, 1.4],
    ///     //  [8.0, 5.9, 4.0],
    ///     //  [3.0, 1.9, 2.0],
    ///     //  [2.0, 4.9, 3.0]]
    ///     // The resulting tensor will have dimensions [4, 3].
    ///     let flipped = tensor.flip([0, 1]);
    ///     println!("{flipped}");
    /// }
    /// ```
    pub fn flip<const N: usize>(self, axes: [isize; N]) -> Tensor<B, D, K> {
        // Convert the axes to usize and handle negative values without using vector
        let mut transformed_axes: [usize; N] = [0; N];
        for (i, &x) in axes.iter().enumerate() {
            transformed_axes[i] = if x < 0 {
                (D as isize + x) as usize
            } else {
                x as usize
            };
        }

        // Check if the axes are valid
        check!(TensorCheck::flip(D, &transformed_axes));

        Tensor::new(K::flip(self.primitive, &transformed_axes))
    }

    /// Flatten the tensor along a given range of dimensions.
    ///
    /// This function collapses the specified range of dimensions into a single dimension,
    /// effectively flattening the tensor in that range.
    ///
    /// # Arguments
    ///
    /// - `start_dim`: The starting dimension of the range to be flattened,
    ///   supports negative indexing.
    /// - `end_dim`: The ending dimension of the range to be flattened (inclusive),
    ///   supports negative indexing.
    ///
    /// # Type Parameters
    ///
    /// - `D2`: The resulting number of dimensions in the flattened tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor<B, D2, K>` instance with the specified range of dimensions flattened.
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 3D tensor with dimensions [2, 3, 4]
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 4]), &device);
    ///
    ///     // Flatten the tensor from dimensions 1 to 2 (inclusive).
    ///     // The resulting tensor will have dimensions [2, 12]
    ///     let flattened: Tensor<B, 2> = tensor.flatten(1, 2);
    ///     println!("{flattened}");
    /// }
    /// ```
    pub fn flatten<const D2: usize>(
        self,
        start_dim: impl AsIndex,
        end_dim: impl AsIndex,
    ) -> Tensor<B, D2, K> {
        let start_dim = canonicalize_dim(start_dim, D, false);
        let end_dim = canonicalize_dim(end_dim, D, false);
        check!(TensorCheck::flatten::<D, D2>(start_dim, end_dim));

        let current_dims = self.shape().dims;
        let mut new_dims: [usize; D2] = [0; D2];
        let mut flatten_dims = 1;

        for i in current_dims[start_dim..=end_dim].iter() {
            flatten_dims *= i;
        }

        new_dims[..start_dim].copy_from_slice(&current_dims[..start_dim]);
        new_dims[start_dim] = flatten_dims;
        new_dims[start_dim + 1..].copy_from_slice(&current_dims[end_dim + 1..]);

        Tensor::new(K::reshape(self.primitive, new_dims.into()))
    }

    /// Squeeze the tensor along all dimensions, removing dimensions
    /// of size one, and effectively reducing the rank of the tensor.
    ///
    /// # Type Parameters
    ///
    ///  - `D2`: The resulting number of dimensions in the squeezed tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor<B, D2, K>` instance with the specified dimension removed.
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 4D tensor with dimensions [1, 3, 1, 3]
    ///     let tensor = Tensor::<B, 4>::from_data(
    ///         [[[[3.0, 4.9, 2.0]], [[2.0, 1.9, 3.0]], [[4.0, 5.9, 8.0]]]],
    ///         &device,
    ///     );
    ///
    ///     // Squeeze the tensor dimensions.
    ///     // The resulting tensor will have dimensions [3, 3].
    ///     let squeezed = tensor.squeeze::<2>();
    ///     println!("{squeezed}");
    /// }
    /// ```
    pub fn squeeze<const D2: usize>(self) -> Tensor<B, D2, K> {
        let new_dims = self
            .shape()
            .dims
            .iter()
            .filter_map(|&dim| if dim == 1 { None } else { Some(dim) })
            .collect::<Vec<_>>();
        check!(TensorCheck::squeeze_dims_len::<D2>(new_dims.len()));

        Tensor::new(K::reshape(self.primitive, new_dims.into()))
    }

    /// Squeeze the tensor along the given dimension, removing the specified dimension
    /// of size one, and effectively reducing the rank of the tensor by one.
    ///
    /// # Arguments
    ///
    /// - `dim`: The dimension to be squeezed.
    ///
    /// # Type Parameters
    ///
    ///  - `D2`: The resulting number of dimensions in the squeezed tensor.
    ///
    /// # Panics
    ///
    /// If the size in the squeezed dimension is not 1.
    ///
    /// # Returns
    ///
    /// A new `Tensor<B, D2, K>` instance with the specified dimension removed.
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 3D tensor with dimensions [3, 1, 3]
    ///     let tensor = Tensor::<B, 3>::from_data(
    ///         [[[3.0, 4.9, 2.0]], [[2.0, 1.9, 3.0]], [[4.0, 5.9, 8.0]]],
    ///         &device,
    ///     );
    ///
    ///     // Squeeze the dimension 1.
    ///     // The resulting tensor will have dimensions [3, 3].
    ///     let squeezed = tensor.squeeze_dim::<2>(1);
    ///     println!("{squeezed}");
    /// }
    /// ```
    pub fn squeeze_dim<const D2: usize>(self, dim: usize) -> Tensor<B, D2, K> {
        check!(TensorCheck::squeeze::<D2>(dim, &self.shape().dims));

        let current_dims = self.shape().dims;
        let mut new_dims: [usize; D2] = [0; D2];

        new_dims[..dim].copy_from_slice(&current_dims[..dim]);
        new_dims[dim..].copy_from_slice(&current_dims[dim + 1..]);

        check!(TensorCheck::squeeze_dims_len::<D2>(new_dims.len()));
        Tensor::new(K::reshape(self.primitive, new_dims.into()))
    }

    /// Removes specified dimensions of size 1 from a tensor's shape. This function takes a tensor and
    /// an array of dimensions (`dims`) to be squeezed. If `dims` is provided, only the dimensions
    /// specified in this array will be removed. Each dimension in `dims` should correspond to a size of 1
    /// in the tensor; otherwise, the dimension will not be squeezed. If `dims` is empty, all single-dimensional entries
    /// in the tensor will be removed. If entries in `dims` are negative, then dimensions will be counted
    /// from the back.
    ///
    /// # Arguments
    ///
    /// - `dims`: The dimension(s) to be squeezed.
    ///
    /// # Type Parameters
    ///
    ///  - `D2`: The resulting number of dimensions in the squeezed tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor<B, D2, K>` instance with the specified dimensions removed.
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 4D tensor with dimensions [2, 1, 4, 1]
    ///     let tensor = Tensor::<B, 4>::ones(Shape::new([2, 1, 4, 1]), &device);
    ///
    ///     // Squeeze the dimensions 1 and 3.
    ///     // The resulting tensor will have dimensions [2, 4].
    ///     let squeezed: Tensor<B, 2> = tensor.squeeze_dims(&[1, 3]);
    ///     println!("{squeezed}");
    /// }
    /// ```
    pub fn squeeze_dims<const D2: usize>(self, dims: &[isize]) -> Tensor<B, D2, K> {
        let current_dims = self.shape().dims;
        let mut dim_indices: Vec<usize>;

        // Check if dims is empty, if yes then assign dim_indices all single-dimensional entries
        if dims.is_empty() {
            dim_indices = current_dims
                .iter()
                .enumerate()
                .filter_map(|(index, &dim)| if dim == 1 { Some(index) } else { None })
                .collect();
        } else {
            // If negative dims, count from the back
            dim_indices = dims
                .iter()
                .map(|&d| {
                    if d < 0 {
                        (current_dims.len() as isize + d) as usize
                    } else {
                        d as usize
                    }
                })
                .collect();
        }

        // Sort indices and remove duplicates
        dim_indices.sort_unstable();
        dim_indices.dedup();

        // Make sure squeeze_dims doesn't result in a tensor with < 1 dimensions
        check!(TensorCheck::squeeze_dims_input::<D2>(
            &dim_indices,
            &current_dims
        ));

        // Calculate new dimensions
        let mut new_dims = Vec::new();
        for (index, &dim_size) in current_dims.iter().enumerate() {
            // Exclude the dimension if it's explicitly marked for squeezing
            if dim_indices.contains(&index) {
                check!(TensorCheck::squeeze::<D2>(index, &current_dims));
                continue;
            }
            new_dims.push(dim_size);
        }

        // Check that after squeezing, we still respect the D2 size
        check!(TensorCheck::squeeze_dims_len::<D2>(new_dims.len()));

        Tensor::new(K::reshape(self.primitive, new_dims.into()))
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit the given size.
    ///
    /// # Type Parameters
    ///
    ///  - `D2`: The resulting number of dimensions in the unsqueezed tensor.
    ///
    /// # Panics
    ///
    /// If the output size `D2` is smaller than the current number of dimensions.
    ///
    /// # Returns
    ///
    /// A new `Tensor<B, D2, K>` instance with the specified dimensions added.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 2D tensor with dimensions [3, 3]
    ///     let tensor = Tensor::<B, 2>::ones(Shape::new([3, 3]), &device);
    ///     // Unsqueeze the tensor up to 4 dimensions.
    ///     // The resulting tensor will have dimensions [1, 1, 3, 3].
    ///     let unsqueezed = tensor.unsqueeze::<4>();
    ///     println!("{unsqueezed}");
    /// }
    /// ```
    pub fn unsqueeze<const D2: usize>(self) -> Tensor<B, D2, K> {
        check!(TensorCheck::unsqueeze::<D, D2>());

        let mut dims = [1; D2];
        let num_ones = D2 - D;
        let shape = self.shape();

        dims[num_ones..(D + num_ones)].copy_from_slice(&shape[..D]);

        let shape = Shape::new(dims);
        self.reshape(shape)
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 2D tensor with dimensions [3, 3]
    ///     let tensor = Tensor::<B, 2>::ones(Shape::new([3, 3]), &device);
    ///     // Unsqueeze the dimension 1.
    ///     // The resulting tensor will have dimensions [3, 1, 3].
    ///     let unsqueezed: Tensor<B, 3> = tensor.unsqueeze_dim(1);
    ///     println!("{unsqueezed}");
    /// }
    /// ```
    pub fn unsqueeze_dim<const D2: usize>(self, dim: usize) -> Tensor<B, D2, K> {
        check!(TensorCheck::unsqueeze_dim::<D, D2>(dim));

        let mut dims = [1; D2];
        let shape = self.shape();

        dims[0..dim].copy_from_slice(&shape[0..dim]);

        if dim < D {
            dims[dim] = 1;
            dims[(dim + 1)..].copy_from_slice(&shape[dim..]);
        } else {
            dims[dim] = 1;
        }

        let shape = Shape::new(dims);
        self.reshape(shape)
    }

    /// Creates a new tensor with added dimensions of size one inserted at the specified indices.
    /// The indices can be negative, in which case they are counted from the last to the first dimension.
    /// the axes can contain duplicates, in which case the number of dimensions inserted at the index
    /// is the number of duplicates.
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 3D tensor with dimensions [3, 4, 5]
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([3, 4, 5]), &device);
    ///     // Unsqueeze the leading dimension (0) once and the trailing dimension (-1) twice.
    ///     // The resulting tensor will have dimensions [1, 3, 4, 5, 1, 1].
    ///     let unsqueezed: Tensor<B, 6> = tensor.unsqueeze_dims(&[0, -1, -1]);
    ///     println!("{unsqueezed}");
    /// }
    /// ```
    pub fn unsqueeze_dims<const D2: usize>(self, axes: &[isize]) -> Tensor<B, D2, K> {
        let mut new_dims = [1; D2];
        let old_dims = self.shape().dims;
        //for checking if the dimension is in the acceptable range

        //part 1: convert the negative indices to positive
        let mut neg_offset = D2;
        let mut dim_indices = axes
            .iter()
            .map(|d| {
                // check if the dimension is in the acceptable range
                check!(TensorCheck::unsqueeze_dims::<{ D2 }>(*d));
                (if *d < 0 {
                    neg_offset -= 1; // handle multiple negative indices (decrease dim value in reverse)
                    d + neg_offset as isize + 1
                } else {
                    *d
                }) as usize
            })
            .collect::<Vec<usize>>();

        //sort the indices
        dim_indices.sort_unstable();

        //Now use this to copy the chunks of the dims
        let mut prev_idx: usize = 0;
        let mut current_left_b: usize = 0;
        let mut current_right_b: usize = 0;
        let mut offset: usize = 0;
        dim_indices.iter().for_each(|d| {
            //check if there is space for at least one dimension
            if prev_idx < *d {
                current_right_b = *d - offset;
                //copy the chunks of the dims
                if current_right_b < D {
                    new_dims[prev_idx..*d]
                        .copy_from_slice(&old_dims[current_left_b..current_right_b]);
                } else {
                    new_dims[prev_idx..*d].copy_from_slice(&old_dims[current_left_b..]);
                }
                prev_idx = *d + 1;
                //offset is equal to the number of extracted elements from the original shape
                offset += current_right_b - current_left_b;
                current_left_b = current_right_b;
            } else {
                //it's sorted so the only reason this would happen
                //is if multiple indices are the same
                prev_idx += 1;
            }
        });
        //copy over anything past the index of the last new dimension
        if current_left_b < D {
            new_dims[prev_idx..].copy_from_slice(&old_dims[current_left_b..]);
        }

        //lastly, create the shape and reshape
        let shape = Shape::new(new_dims);
        self.reshape(shape)
    }

    /// Roll operation along a specific dimension; wrapping around the elements.
    ///
    /// ## Parameters
    ///
    /// - `shift`: The roll extent; supports negative values and wraps around.
    /// - `dim`: The dimension to roll; supports negative indexing.
    ///
    /// ## Returns
    ///
    /// A new tensor with the specified dimension rolled by the given shift amount.
    pub fn roll_dim<Shift, Dim>(self, shift: Shift, dim: Dim) -> Self
    where
        Shift: AsIndex,
        Dim: AsIndex,
    {
        let dim = canonicalize_dim(dim, D, false);
        let size = self.shape().dims[dim];
        if size == 0 {
            // If the dimension is empty, return the tensor as is.
            return self;
        }

        let shift = wrap_index(shift, size);
        if shift == 0 {
            // If the shift is zero, return the tensor as is.
            return self;
        }

        self.unchecked_roll_dim(shift, dim)
    }

    /// Internal implementation of `roll_dim` that does not canonicalize dimensions or shifts.
    ///
    /// ## Parameters
    ///
    /// - `shift`: The number of positions to shift; must be (0 < shift < size).
    /// - `dim`: The dimension to roll; must be a valid index for the tensor's shape.
    ///
    /// ## Returns
    ///
    /// A new tensor with the specified dimension rolled by the given shift amount.
    #[inline(always)]
    fn unchecked_roll_dim(self, shift: usize, dim: usize) -> Self {
        #[cfg(debug_assertions)]
        {
            let size = self.shape().dims[dim];
            assert!(
                0 < shift && shift < size,
                "Expected: 0 < shift < size: found shift={shift}, size={size}",
            );
            assert!(
                dim < self.shape().num_dims(),
                "Expected: dim < num_dims: found dim={dim}, num_dims={size}",
            );
        }

        Tensor::cat(
            vec![
                self.clone().slice_dim(dim, shift..),
                self.slice_dim(dim, ..shift),
            ],
            dim,
        )
    }

    /// Roll operation.
    ///
    /// Note: unlike ``pytorch``, `dims` and `shifts` must have the same length.
    ///
    /// A given `dim` may be rolled multiple times, and the shifts will be applied sequentially.
    ///
    /// ## Parameters
    ///
    /// - `shifts`: A slice of shifts corresponding to each dimension;
    ///   supports negative values and wraps around.
    /// - `dims`: A slice of dimensions to roll; supports negative indexing.
    ///
    /// ## Returns
    ///
    /// A new tensor with the specified dimensions rolled by the given shifts.
    pub fn roll<Shift, Dim>(self, shifts: &[Shift], dims: &[Dim]) -> Self
    where
        Shift: AsIndex,
        Dim: AsIndex,
    {
        assert_eq!(
            dims.len(),
            shifts.len(),
            "Dimensions and shifts must align; found dims={dims:#?}, shifts={shifts:#?}",
        );

        // This is a fair amount of complexity, which could be replaced
        // by a simple canonicalization of `dims` and wrapping of `shifts`.
        // The work is done here to ensure that any roll operation
        // which could be a no-op is a no-op; simplifying the accounting
        // needed by backend-specific implementations of the inner roll op.

        let item_count = dims.len();

        let shape = self.shape().dims;

        // Accumulate the effective shifts for each dimension.
        let mut accumulated_shifts: Vec<isize> = vec![0; shape.len()];
        for i in 0..item_count {
            let dim = canonicalize_dim(dims[i], D, false);
            accumulated_shifts[dim] += shifts[i].index();
        }

        // Do this after we've checked the validity of `dims` and `shifts`.
        if self.shape().num_elements() == 0 {
            // If the tensor is empty, return it as is.
            return self;
        }

        // Wrap the accumulated shifts, and filter out empty dimensions.
        let mut effective_dims: Vec<usize> = Vec::with_capacity(item_count);
        let mut effective_shifts: Vec<usize> = Vec::with_capacity(item_count);
        for dim in 0..shape.len() {
            // `wrap_index` should inline, and has a fast-exit path for zero shifts.
            let shift = wrap_index(accumulated_shifts[dim], shape[dim]);
            if shift == 0 {
                continue;
            }

            effective_dims.push(dim);
            effective_shifts.push(shift);
        }

        // If no shifts are needed, return the original tensor.
        if effective_shifts.is_empty() {
            return self;
        }

        // At this point:
        // - `dims` contains the effective dimensions to roll, in index order,
        // - `shifts` contains the effective usize shifts for each dimension.
        // - Every shift is non-zero, and less than the size of the corresponding dimension.
        self.unchecked_roll(&effective_shifts, &effective_dims)
    }

    /// `roll` internal implementation.
    ///
    /// ## Parameters
    ///
    /// - `shifts`: A slice of shifts corresponding to each dimension;
    ///   must be non-empty, the same length as `dims`, and all ``1..<size>``.
    /// - `dims`: A slice of dimensions to roll; must be non-empty;
    ///   the same length as `shifts`, and must not contain repeats.
    ///
    /// ## Panics
    ///
    /// Panics if the shifts and dimensions do not align, or if dimensions contain repeats.
    ///
    /// ## Returns
    ///
    /// A new tensor with the specified dimensions rolled by the given shifts.
    #[inline(always)]
    fn unchecked_roll(self, shifts: &[usize], dims: &[usize]) -> Self {
        #[cfg(debug_assertions)]
        {
            assert!(!shifts.is_empty());
            assert_eq!(
                shifts.len(),
                dims.len(),
                "Shifts and dimensions must align; found {} shifts and {} dims",
                shifts.len(),
                dims.len()
            );

            let mut unique_dims = dims.to_vec();
            unique_dims.dedup();

            assert_eq!(
                unique_dims.len(),
                dims.len(),
                "Dimensions must not contain repeats; found {} unique dims and {} total dims",
                unique_dims.len(),
                dims.len()
            )
        }

        let x = self.unchecked_roll_dim(shifts[0], dims[0]);

        if dims.len() == 1 {
            x
        } else {
            x.unchecked_roll(&shifts[1..], &dims[1..])
        }
    }

    /// Returns a tensor containing the elements selected from the given slices.
    ///
    /// This method provides flexible tensor slicing with support for various range types,
    /// negative indices, and stepped slicing. The method accepts both single slices and
    /// arrays of slices, with the [`s!`] macro providing convenient syntax for complex patterns.
    ///
    /// # Arguments
    ///
    /// * `slices` - Can be:
    ///   - A single range for 1D slicing (e.g., `0..5`, `..`, `2..`)
    ///   - An array of ranges (e.g., `[0..2, 1..4]`)
    ///   - The [`s!`] macro output for advanced slicing with steps
    ///
    /// # Behavior
    ///
    /// - Supports partial and full slicing in any number of dimensions
    /// - Handles negative indices by wrapping from the end (-1 is the last element)
    /// - Automatically clamps ranges that exceed tensor dimensions
    /// - Supports stepped slicing for selecting every nth element
    /// - Negative steps reverse the selection order
    ///
    /// # Panics
    ///
    /// - If the number of slices exceeds the tensor's dimensions
    /// - If a range is descending (e.g., 2..1) or empty (e.g., 1..1) without negative step
    /// - If a step is zero
    ///
    /// # Examples
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape, s};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///
    ///     // Single dimension slicing - no brackets needed!
    ///     let tensor = Tensor::<B, 1, burn_tensor::Int>::arange(0..10, &device);
    ///     let slice = tensor.clone().slice(2..8);  // Simple range
    ///     assert_eq!(slice.into_data().to_vec::<i32>().unwrap(), vec![2, 3, 4, 5, 6, 7]);
    ///
    ///     // Using s! macro for single dimension with step
    ///     let slice = tensor.clone().slice(s![0..10;2]);  // Every 2nd element
    ///     assert_eq!(slice.into_data().to_vec::<i32>().unwrap(), vec![0, 2, 4, 6, 8]);
    ///
    ///     // Reverse a dimension with negative step
    ///     let slice = tensor.slice(s![..;-1]);  // Reverse entire tensor
    ///     assert_eq!(slice.into_data().to_vec::<i32>().unwrap(), vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    ///
    ///     // Multi-dimensional slicing
    ///     let tensor = Tensor::<B, 2>::ones(Shape::new([4, 6]), &device);
    ///
    ///     // Array syntax for simple ranges
    ///     let slice = tensor.clone().slice([1..3, 2..5]);
    ///     assert_eq!(slice.dims(), [2, 3]);
    ///
    ///     // Advanced multi-dimensional with s! macro
    ///     let slice = tensor.clone().slice(s![0..4;2, ..;-1]);  // Every 2nd row, reverse columns
    ///     assert_eq!(slice.dims(), [2, 6]);
    ///
    ///     // Complex 3D example with mixed slice types
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([4, 6, 8]), &device);
    ///     let slice = tensor.slice(s![1..3, ..;2, -3..]);  // Rows 1-2, every 2nd col, last 3 depth
    ///     assert_eq!(slice.dims(), [2, 3, 3]);
    ///
    ///     // Using negative indices
    ///     let tensor = Tensor::<B, 2>::ones(Shape::new([4, 6]), &device);
    ///     let slice = tensor.slice(s![-2.., ..-1]);  // Last 2 rows, all but last column
    ///     assert_eq!(slice.dims(), [2, 5]);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`s!`] - The recommended macro for creating complex slice specifications
    /// - [`slice_assign`](Self::slice_assign) - Assign values to a slice
    /// - [`slice_fill`](Self::slice_fill) - Fill a slice with a constant value
    /// - [`slice_dim`](Self::slice_dim) - Slice a single dimension
    ///
    /// [`s!`]: crate::s!
    pub fn slice<const D2: usize, S>(self, slices: S) -> Self
    where
        S: SliceArg<D2>,
    {
        let shape = self.shape();
        let slices = slices.into_slices(shape.clone());

        // Validate slices
        check!(TensorCheck::slice::<D, D2>(&shape, &slices));

        // Calculate output shape and check for empty slices
        let mut output_dims = shape.dims.clone();
        for (dim, slice) in slices.iter().enumerate() {
            output_dims[dim] = slice.output_size(shape.dims[dim]);
        }

        // Return empty tensor if any dimension is 0 (empty slice)
        if output_dims.contains(&0) {
            return Self::empty(output_dims, &self.device());
        }

        // Use the slice method that supports steps
        Self::new(K::slice(self.primitive, &slices))
    }

    /// Assigns values to a slice of the tensor and returns the updated tensor.
    ///
    /// This method supports advanced slicing with steps, including negative steps for reverse
    /// assignment. Like `slice`, it accepts both single slices and arrays, with the [`s!`] macro
    /// providing powerful syntax for complex patterns.
    ///
    /// # Arguments
    ///
    /// * `slices` - Slice specification (same format as `slice` method)
    /// * `values` - Tensor with values to assign (must match slice dimensions)
    ///
    /// # Panics
    ///
    /// - If slices exceed tensor dimensions
    /// - If values dimensions don't match the selected slice shape
    /// - If a step is zero
    ///
    /// # Examples
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, s};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///
    ///     // Simple assignment to a sub-region
    ///     let mut tensor = Tensor::<B, 2>::zeros([4, 6], &device);
    ///     let values = Tensor::<B, 2>::ones([2, 3], &device);
    ///     tensor = tensor.slice_assign([1..3, 2..5], values);
    ///     // Now tensor[1..3, 2..5] contains ones
    ///
    ///     // Single dimension assignment with step
    ///     let mut tensor = Tensor::<B, 1>::zeros([10], &device);
    ///     let values = Tensor::<B, 1>::ones([5], &device);
    ///     tensor = tensor.slice_assign(s![0..10;2], values);
    ///     // Now every 2nd element is 1: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    ///
    ///     // Reverse assignment with negative step
    ///     let mut tensor = Tensor::<B, 1>::from_data([0.0, 1.0, 2.0, 3.0, 4.0], &device);
    ///     let values = Tensor::<B, 1>::from_data([10.0, 11.0, 12.0, 13.0, 14.0], &device);
    ///     tensor = tensor.slice_assign(s![..;-1], values);
    ///     // Assigns in reverse: [14, 13, 12, 11, 10]
    ///
    ///     // Complex multi-dimensional assignment
    ///     let mut tensor = Tensor::<B, 3>::zeros([4, 6, 8], &device);
    ///     let values = Tensor::<B, 3>::ones([2, 3, 3], &device);
    ///     tensor = tensor.slice_assign(s![0..4;2, ..;2, -3..], values);
    ///     // Assigns to every 2nd row, every 2nd column, last 3 in depth
    ///
    ///     // Mixed syntax example
    ///     let mut tensor = Tensor::<B, 2>::zeros([8, 8], &device);
    ///     let pattern = Tensor::<B, 2>::ones([4, 4], &device);
    ///     tensor = tensor.slice_assign(s![..;2, ..;2], pattern);
    ///     // Creates a checkerboard pattern with ones
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`s!`] - The recommended macro for creating complex slice specifications
    /// - [`slice`](Self::slice) - Extract a slice from a tensor
    /// - [`slice_fill`](Self::slice_fill) - Fill a slice with a constant value
    ///
    /// [`s!`]: crate::s!
    pub fn slice_assign<const D2: usize, S>(self, slices: S, values: Self) -> Self
    where
        S: SliceArg<D2>,
    {
        let shape = self.shape();
        let slices = slices.into_slices(shape.clone());

        // Check if any slice produces 0 elements (empty assignment).
        // Empty assignments are no-ops and would cause issues in backend implementations.
        let is_empty_assignment = slices
            .iter()
            .enumerate()
            .any(|(i, slice)| slice.output_size(shape.dims[i]) == 0);

        if is_empty_assignment {
            return self;
        }

        check!(TensorCheck::slice_assign::<D, D2>(
            &shape,
            &values.shape(),
            &slices
        ));

        Self::new(K::slice_assign(self.primitive, &slices, values.primitive))
    }

    /// Fills a slice of the tensor with a constant value and returns the updated tensor.
    ///
    /// Like other slice methods, accepts both single slices and arrays. However, this method
    /// currently **does not support stepped slicing** - use [`slice_assign`](Self::slice_assign)
    /// with a constant tensor for stepped patterns.
    ///
    /// # Arguments
    ///
    /// * `slices` - Slice specification (same format as `slice` method, but no steps)
    /// * `value` - The value to fill the slice with
    ///
    /// # Panics
    ///
    /// - If slices exceed tensor dimensions
    /// - If any slice has a step != 1 (not yet supported)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, s};
    ///
    /// fn example<B: Backend>() {
    ///     let device = B::Device::default();
    ///
    ///     // Simple fill for a single dimension
    ///     let mut tensor = Tensor::<B, 1>::zeros([10], &device);
    ///     tensor = tensor.slice_fill(2..5, 1.0);
    ///     // Now tensor is [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    ///
    ///     // Multi-dimensional fill
    ///     let mut tensor = Tensor::<B, 2>::zeros([4, 6], &device);
    ///     tensor = tensor.slice_fill([1..3, 2..5], -1.0);
    ///     // Fills the rectangle at rows 1-2, columns 2-4 with -1
    ///
    ///     // Using negative indices
    ///     let mut tensor = Tensor::<B, 1>::zeros([10], &device);
    ///     tensor = tensor.slice_fill(-3.., 2.0);
    ///     // Fills the last 3 elements with 2.0
    ///
    ///     // Complex multi-dimensional example
    ///     let mut tensor = Tensor::<B, 3>::ones([4, 6, 8], &device);
    ///     tensor = tensor.slice_fill(s![1..3, .., -2..], 0.0);
    ///     // Sets rows 1-2, all columns, last 2 in depth to 0
    ///
    ///     // Stepped slicing is supported
    ///     let mut tensor = Tensor::<B, 1>::zeros([10], &device);
    ///     tensor = tensor.slice_fill(s![0..10;2], 1.0);
    ///     // Now every 2nd element is 1: [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`s!`] - The macro for creating slice specifications with steps
    /// - [`slice`](Self::slice) - Extract a slice from a tensor
    /// - [`slice_assign`](Self::slice_assign) - Assign tensor values to a slice
    ///
    /// [`s!`]: crate::s!
    pub fn slice_fill<const D2: usize, S, E: ElementConversion>(self, slices: S, value: E) -> Self
    where
        S: SliceArg<D2>,
    {
        let shape = self.shape();
        let slices = slices.into_slices(shape.clone());

        check!(TensorCheck::slice::<D, D2>(&shape, &slices));

        Self::new(K::slice_fill(self.primitive, &slices, value.elem()))
    }

    /// Returns a new tensor with the specified dimension sliced.
    ///
    /// # Arguments
    ///
    /// * `dim`: The dimension to slice.
    /// * `slice`: The slice specification for the dimension. Can be a range (e.g., `2..5`),
    ///   slice with step (via `s!` macro, e.g., `s![0..10;2]`), or any type that implements `Into<Slice>`.
    ///
    /// # Returns
    ///
    /// A new tensor with the specified dimension sliced.
    ///
    /// # Panics
    ///
    /// If the slice is out of bounds for the specified dimension.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use burn_tensor::{Tensor, s};
    /// # use burn_tensor::backend::Backend;
    /// #
    /// # fn example<B: Backend>() {
    /// #     let device = B::Device::default();
    ///     let tensor = Tensor::<B, 3>::zeros([3, 4, 5], &device);
    ///
    ///     // Simple range slicing
    ///     let sliced = tensor.clone().slice_dim(1, 1..3);
    ///     assert_eq!(sliced.shape().dims, [3, 2, 5]);
    ///
    ///     // Slicing with step - take every 2nd element
    ///     let sliced = tensor.clone().slice_dim(2, s![0..5;2]);
    ///     assert_eq!(sliced.shape().dims, [3, 4, 3]); // Takes indices 0, 2, 4
    ///
    ///     // Reverse slicing with negative step
    ///     let sliced = tensor.clone().slice_dim(1, s![..;-1]);
    ///     assert_eq!(sliced.shape().dims, [3, 4, 5]); // Reverses dimension 1
    ///
    ///     // Select from index 2 with step 3
    ///     let sliced = tensor.clone().slice_dim(0, s![2..;3]);
    ///     assert_eq!(sliced.shape().dims, [1, 4, 5]); // Takes only index 2
    ///
    ///     // Select single index (reduces dimension to size 1)
    ///     let sliced = tensor.slice_dim(0, 1);
    ///     assert_eq!(sliced.shape().dims, [1, 4, 5]);
    /// # }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`slice`](Self::slice) - Slice multiple dimensions simultaneously
    /// - [`s!`] - The macro for creating complex slice specifications
    ///
    /// [`s!`]: crate::s!
    pub fn slice_dim<S>(self, dim: usize, slice: S) -> Self
    where
        S: Into<Slice>,
    {
        check!(TensorCheck::check_dim::<D>(dim));
        let slice: Slice = slice.into();

        Self::new(K::slice_dim(self.primitive, dim, &slice))
    }

    /// Returns the device of the current tensor.
    pub fn device(&self) -> B::Device {
        K::device(&self.primitive)
    }

    /// Move the tensor to the given device.
    pub fn to_device(self, device: &B::Device) -> Self {
        Self::new(K::to_device(self.primitive, device))
    }

    /// Select tensor elements along the given dimension corresponding to the given indices.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to select from. Supports negative indexing.
    /// * `indices` - The indices of the elements to select.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Int};
    ///
    /// fn example<B: Backend>() {
    ///   let device = B::Device::default();
    ///   let tensor = Tensor::<B, 2>::from_data([[1.0, -2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    ///   let indices = Tensor::<B, 1, Int>::from_data([0], &device);
    ///   let tensor = tensor.select(0, indices);
    ///   println!("{tensor}");
    ///   //  [[1.0, -2.0, 3.0]]
    /// }
    /// ```
    pub fn select(self, dim: impl AsIndex, indices: Tensor<B, 1, Int>) -> Self {
        let dim = canonicalize_dim(dim, D, false);
        check!(TensorCheck::select::<D>(dim));
        Self::new(K::select(self.primitive, dim, indices.primitive))
    }

    /// Assign the selected elements along the given dimension corresponding to the given indices
    /// from the value tensor to the original tensor using sum reduction.
    ///
    /// # Note
    /// For booleans, the sum operator is logical or.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to select. Supports negative indexing.
    /// * `indices` - The indices to select from the tensor.
    /// * `values` - The values to assign to the selected indices.
    /// * `update` - The operation used to update the existing values at the indexed positions (e.g., add).
    ///
    /// # Example
    ///
    /// Example using a 3D tensor:
    ///
    /// `input[indices[i], j, k] += values[i, j, k]; // dim = 0`
    /// `input[i, indices[j], k] += values[i, j, k]; // dim = 1`
    /// `input[i, j, indices[k]] += values[i, j, k]; // dim = 2`
    /// `input[i, j, indices[k]] += values[i, j, k]; // dim = -1 (same as dim = 2)`
    ///
    /// # Warning
    ///
    /// Not all backends have runtime bound checks for the indices, so make sure they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    pub fn select_assign(
        self,
        dim: impl AsIndex,
        indices: Tensor<B, 1, Int>,
        values: Tensor<B, D, K>,
        update: IndexingUpdateOp,
    ) -> Self {
        let dim = canonicalize_dim(dim, D, false);
        check!(TensorCheck::select_assign::<D>(
            dim,
            &indices.shape(),
            &values.shape()
        ));

        Self::new(K::select_assign(
            self.primitive,
            dim,
            indices.primitive,
            values.primitive,
            update,
        ))
    }

    /// Converts the data of the current tensor.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](crate::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub fn into_data(self) -> TensorData {
        self.try_into_data().expect(
            "Error while reading data: use `try_into_data` instead to catch the error at runtime",
        )
    }

    /// Converts the data of the current tensor and returns any error that might have occurred since the
    /// last time the device was synchronized.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](crate::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub fn try_into_data(self) -> Result<TensorData, ExecutionError> {
        crate::try_read_sync(self.into_data_async()).expect(
            "Failed to read tensor data synchronously.
        This can happen on platforms that don't support blocking futures like WASM.
        If possible, try using into_data_async instead.",
        )
    }

    /// Converts the data of the current tensor.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](crate::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub fn to_data(&self) -> TensorData {
        self.clone().into_data()
    }

    /// Returns the data of the current tensor.
    pub async fn into_data_async(self) -> Result<TensorData, ExecutionError> {
        K::into_data_async(self.primitive).await
    }

    /// Returns the data of the current tensor.
    pub async fn to_data_async(&self) -> Result<TensorData, ExecutionError> {
        self.clone().into_data_async().await
    }

    /// Create a tensor from the given data on the given device.
    pub fn from_data<T>(data: T, device: &B::Device) -> Self
    where
        T: Into<TensorData>,
    {
        let data = data.into();
        check!(TensorCheck::creation_ops::<D>(
            "From Data",
            data.shape.as_slice()
        ));
        Self::new(K::from_data(data, device))
    }

    /// Create a tensor from the given data on the given device enforcing the given data type.
    pub fn from_data_dtype<T>(data: T, device: &B::Device, dtype: DType) -> Self
    where
        T: Into<TensorData>,
    {
        let data = data.into();
        check!(TensorCheck::creation_ops::<D>(
            "From Data",
            data.shape.as_slice()
        ));
        Self::new(K::from_data_dtype(data, device, dtype))
    }

    /// Repeat the tensor along the given dimension.
    ///
    /// The output tensor has the same shape, except along the given dimension.
    ///
    /// # Arguments
    /// - `dim`: The dimension to repeat.
    /// - `times`: The number of times to repeat the tensor along the given dimension in the new tensor.
    ///
    /// # Returns
    ///
    /// A new tensor with the given dimension repeated `times` times.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 2D tensor with dimensions [3, 2]
    ///     let tensor = Tensor::<B, 2>::from_data([[3.0, 4.9], [2.0, 1.9], [4.0, 5.9]], &device);
    ///
    ///     // Repeat the tensor along the dimension 0 twice.
    ///     // [[3.0, 4.9], [2.0, 1.9], [4.0, 5.9], [3.0, 4.9], [2.0, 1.9], [4.0, 5.9]]
    ///     // The resulting tensor will have dimensions [6, 2].
    ///     let repeated = tensor.repeat_dim(0, 2);
    ///     println!("{repeated}");
    /// }
    /// ```
    pub fn repeat_dim(self, dim: usize, times: usize) -> Self {
        Self::new(K::repeat_dim(self.primitive, dim, times))
    }

    /// Repeat the tensor along the given dimensions.
    /// # Arguments
    /// - `sizes`: Borrowed slice of the number of times to repeat each dimension.
    ///
    /// # Returns
    ///
    /// A new tensor with the given dimensions repeated `times` times.
    ///
    /// # Panics
    ///
    /// If `sizes` contains more elements than the number of dimensions.
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 2D tensor with dimensions [3, 2]
    ///     let tensor = Tensor::<B, 2>::from_data([[3.0, 4.9], [2.0, 1.9], [4.0, 5.9]], &device);
    ///
    ///     // Repeat the tensor along the dimension 0 twice and the dimension 0 once.
    ///     // [[3.0, 4.9], [2.0, 1.9], [4.0, 5.9], [3.0, 4.9], [2.0, 1.9], [4.0, 5.9]]
    ///     // The resulting tensor will have dimensions [6, 2].
    ///     let repeated = tensor.repeat(&[2, 1]);
    /// }
    /// ```
    pub fn repeat(self, sizes: &[usize]) -> Self {
        let mut tensor = self;
        for (dim, &times) in sizes.iter().enumerate() {
            if times > 1 {
                tensor = tensor.repeat_dim(dim, times);
            }
        }
        tensor
    }

    /// Applies element-wise equal comparison.
    ///
    /// # Returns
    /// A boolean tensor that is `true` where input is equal to `other` and `false` elsewhere.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let t1 = Tensor::<B, 2>::from_data([[2.0, 4.9], [2.0, 1.9], [4.0, 5.9]], &device);
    ///     let t2 = Tensor::<B, 2>::from_data([[3.0, 4.9], [2.0, 1.9], [4.0, 5.9]], &device);
    ///     // Compare the elements of the two 2D tensors with dimensions [3, 2].
    ///     // [[false, true], [true, true], [true, true]]
    ///     let equal = t1.equal(t2);
    ///     println!("{equal}");
    /// }
    /// ```
    pub fn equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Equal", &self, &other));
        Tensor::new(K::equal(self.primitive, other.primitive))
    }

    /// Applies element-wise non-equality comparison.
    ///
    /// # Returns
    /// A boolean tensor that is `true` where input is not equal to `other` and `false` elsewhere.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let t1 = Tensor::<B, 2>::from_data([[2.0, 4.9], [2.0, 1.9], [4.0, 5.9]], &device);
    ///     let t2 = Tensor::<B, 2>::from_data([[3.0, 4.9], [2.0, 1.9], [4.0, 5.9]], &device);
    ///     // Compare the elements of the two 2D tensors for inequality.
    ///     // [[true, false], [false, false], [false, false]]
    ///     let not_equal = t1.not_equal(t2);
    ///     println!("{not_equal}");
    /// }
    /// ```
    pub fn not_equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("NotEqual", &self, &other));
        Tensor::new(K::not_equal(self.primitive, other.primitive))
    }

    /// Concatenates all tensors into a new one along the given dimension.
    ///
    /// # Panics
    ///
    /// - If `dim` is higher than the rank.
    /// - If `tensors` is an empty vector.
    /// - If all tensors don't have the same shape (the dimension `dim` is ignored).
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let t1 = Tensor::<B, 2>::from_data([[3.0, 4.9, 2.0, 1.0], [2.0, 1.9, 3.0, 1.0]], &device);
    ///     let t2 = Tensor::<B, 2>::from_data([[4.0, 5.9, 8.0], [1.4, 5.8, 6.0]], &device);
    ///
    ///     // Concatenate the two tensors with shapes [2, 4] and [2, 3] along the dimension 1.
    ///     // [[3.0, 4.9, 2.0, 1.0, 4.0, 5.9, 8.0], [2.0, 1.9, 3.0, 1.0, 1.4, 5.8, 6.0]]
    ///     // The resulting tensor will have shape [2, 7].
    ///     let concat = Tensor::cat(vec![t1, t2], 1);
    ///     println!("{concat}");
    /// }
    /// ```
    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
        check!(TensorCheck::cat(&tensors, dim));

        // Filter out tensors with size 0 along the concatenation dimension.
        // Empty tensors don't contribute to the output and would cause issues
        // in backend implementations (e.g., division by zero in slice_assign).
        // Safety: TensorCheck::cat ensures tensors is non-empty
        let first_tensor = tensors.first().unwrap();
        let device = first_tensor.device();
        let mut shape = first_tensor.shape();

        let non_empty_primitives: Vec<_> = tensors
            .into_iter()
            .filter(|t| t.shape().dims[dim] > 0)
            .map(|t| t.primitive)
            .collect();

        // If all tensors were empty, return an empty tensor with size 0 on concat dim
        if non_empty_primitives.is_empty() {
            shape.dims[dim] = 0;
            return Self::empty(shape, &device);
        }

        Self::new(K::cat(non_empty_primitives, dim))
    }

    /// Concatenates all tensors into a new one along a new dimension.
    ///
    /// # Panics
    ///
    /// - If all tensors don't have the same shape.
    /// - If given dimension is not with range of 0..D2
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let t1 = Tensor::<B, 2>::from_data([[3.0, 4.9, 2.0], [2.0, 1.9, 3.0]], &device);
    ///     let t2 = Tensor::<B, 2>::from_data([[4.0, 5.9, 8.0], [1.4, 5.8, 6.0]], &device);
    ///     let t3 = Tensor::<B, 2>::from_data([[4.0, 5.9, 8.0], [1.4, 5.8, 6.0]], &device);
    ///
    ///     // Concatenate the three tensors with shape [2, 3] along a new dimension, 0.
    ///     // [[[3.0, 4.9, 2.0], [2.0, 1.9, 3.0]],
    ///     //  [[4.0, 5.9, 8.0], [1.4, 5.8, 6.0]],
    ///     //  [[4.0, 5.9, 8.0], [1.4, 5.8, 6.0]]]
    ///     // The resulting tensor will have shape [3, 2, 3].
    ///     let stacked= Tensor::stack::<3>(vec![t1, t2, t3], 0);
    ///     println!("{stacked}");
    /// }
    /// ```
    pub fn stack<const D2: usize>(tensors: Vec<Tensor<B, D, K>>, dim: usize) -> Tensor<B, D2, K> {
        check!(TensorCheck::stack::<B, D, K, D2>(&tensors, dim));
        let tensors = tensors.into_iter().map(|t| t.unsqueeze_dim(dim)).collect();
        Tensor::<B, D2, K>::cat(tensors, dim)
    }

    /// Iterate over slices of tensors alongside a given dimension.
    ///
    /// # Panics
    ///
    /// If given dimension is greater than or equal to tensor rank.
    ///
    /// # Returns
    ///
    /// A tensor iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    /// fn example<B: Backend>() {
    ///   let device = Default::default();
    ///   let tensor = Tensor::<B,2>::from_data([[3.0, 4.9, 2.0], [2.0, 1.9, 3.0]], &device);
    ///   // Given a 2D tensor with dimensions [2, 3], iterate over slices of tensors along the dimension 0.
    ///   let iter = tensor.iter_dim(0);
    ///   for (i,tensor) in iter.enumerate() {
    ///     println!("Tensor {}: {}", i, tensor);
    ///     // Tensor 0: Tensor { data: [[3.0, 4.9, 2.0]], ... }
    ///     // Tensor 1: Tensor { data: [[2.0, 1.9, 3.0]], ... }
    ///  }
    /// }
    /// ```
    pub fn iter_dim(self, dim: usize) -> DimIter<B, D, K> {
        check!(TensorCheck::dim_ops::<D>("iter_dim", dim));
        DimIter::new(self, dim)
    }

    /// Returns a new tensor with the given dimension narrowed to the given range.
    ///
    /// # Panics
    ///
    /// - If the dimension is greater than the number of dimensions of the tensor.
    /// - If the given range exceeds the number of elements on the given dimension.
    ///
    /// # Returns
    ///
    /// A new tensor with the given dimension narrowed to the given range.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 2D tensor with dimensions [4, 3]
    ///     let tensor = Tensor::<B, 2>::from_data(
    ///         [
    ///             [3.0, 4.9, 2.0],
    ///             [2.0, 1.9, 3.0],
    ///             [6.0, 1.5, 7.0],
    ///             [3.0, 4.9, 9.0],
    ///         ],
    ///         &device,
    ///     );
    ///     // Narrow the tensor along the dimension 0, keeping 3 elements starting from index 1.
    ///     // [[2.0, 1.9, 3.0], [6.0, 1.5, 7.0], [3.0, 4.9, 9.0]]
    ///     // The resulting tensor will have dimensions [3, 3].
    ///     let narrowed = tensor.narrow(0, 1, 3);
    ///     println!("{narrowed}");
    /// }
    /// ```
    pub fn narrow(self, dim: usize, start: usize, length: usize) -> Self {
        check!(TensorCheck::dim_ops::<D>("narrow", dim));
        check!(TensorCheck::narrow(&self, dim, start, length));
        let dims = self.dims();

        let ranges: [Range<usize>; D] = dims
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if i == dim {
                    start..(start + length)
                } else {
                    0..*d
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        Self::slice(self, ranges)
    }

    /// Attempts to split the tensor into a specified number of chunks along a given dimension.
    /// May return less chunks than requested if the tensor size is not divisible by the number of chunks.
    ///
    /// When the given dimension is evenly divisible by the number of chunks, the chunks will be of equal size.
    /// Otherwise all chunks will be of equal size except for the last one.
    ///
    /// # Panics
    ///
    /// If the dimension is greater than the number of dimensions of the tensor.
    ///
    /// # Returns
    /// A vector of tensors.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 2D tensor with dimensions [4, 3]
    ///     let tensor = Tensor::<B, 2>::from_data(
    ///         [
    ///             [3.0, 4.9, 2.0],
    ///             [2.0, 1.9, 3.0],
    ///             [6.0, 1.5, 7.0],
    ///             [3.0, 4.9, 9.0],
    ///         ],
    ///         &device,
    ///     );
    ///     // Split the tensor along the dimension 1 into 2 chunks.
    ///     // The first chuck will have shape [4, 2]:
    ///     // [[3.0, 4.9], [2.0, 1.9], [6.0, 1.5], [3.0, 4.9]]
    ///     // The second chunk will have shape [4, 1]:
    ///     // [[2.0], [3.0], [7.0], [9.0]]
    ///     let chunks = tensor.chunk(2, 1);
    ///     println!("{chunks:?}");
    /// }
    /// ```
    pub fn chunk(self, chunks: usize, dim: usize) -> Vec<Self> {
        check!(TensorCheck::dim_ops::<D>("chunk", dim));
        let size = self.shape().dims[dim];
        if size < chunks {
            return (0..size)
                .map(|i| Self::narrow(self.clone(), dim, i, 1))
                .collect();
        }

        let mut tensors = Vec::with_capacity(chunks);
        let mut sum_chunk_size = 0;
        if size.is_multiple_of(chunks) {
            let chunk_size = size / chunks;
            for _ in 0..chunks {
                tensors.push(Self::narrow(self.clone(), dim, sum_chunk_size, chunk_size));
                sum_chunk_size += chunk_size;
            }
        } else {
            let chunk_size = (size / chunks) + 1; // assumes not divisible
            for _ in 0..chunks - 1 {
                tensors.push(Self::narrow(self.clone(), dim, sum_chunk_size, chunk_size));
                sum_chunk_size += chunk_size;
            }
            let remainder = size % chunk_size;
            tensors.push(Self::narrow(self.clone(), dim, sum_chunk_size, remainder));
        }

        tensors
    }

    /// Splits the tensor into chunks of a specified size along a given dimension.
    /// Each chunk is a view of the original tensor.
    ///
    /// If the tensor size along the given dimension is not divisible by `split_size`,
    /// then the last chunk will be smaller.
    ///
    /// # Panics
    ///
    /// If the specified dimension to split along is greater than the number of dimensions of the tensor.
    ///
    /// # Returns
    ///
    /// A vector of tensors.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 1D tensor with 5 elements
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, 1.0, 2.0, 3.0, 4.0], &device);
    ///     // Split the tensor into chunks of size 2 along dimension 0
    ///     let chunks = tensor.split(2, 0);
    ///     // The result is a vector of tensors:
    ///     // [Tensor([0.0, 1.0]), Tensor([2.0, 3.0]), Tensor([4.0])]
    ///     println!("{:?}", chunks);
    /// }
    /// ```
    pub fn split(self, split_size: usize, dim: usize) -> Vec<Self> {
        check!(TensorCheck::split::<D>(&self.shape(), split_size, dim));
        let size = self.shape().dims[dim];
        let mut tensors = Vec::new();

        let mut start = 0;
        while start < size {
            let length = usize::min(split_size, size - start);
            tensors.push(Self::narrow(self.clone(), dim, start, length));
            start += length;
        }

        tensors
    }

    /// Splits the tensor into chunks with the specified sizes along a given dimension.
    /// Each chunk is a view of the original tensor.
    ///
    /// The sizes of the chunks are specified in the `split_sizes` vector. The sum of the sizes
    /// in `split_sizes` must equal the size of the tensor along the specified dimension.
    ///
    /// # Panics
    ///
    /// If the specified dimension to split along is greater than the number of dimensions of the tensor or
    /// if the sum of `dim_sizes` does not equal the size of the tensor along `dim`.
    ///
    /// # Returns
    ///
    /// A vector of tensors.
    ///
    /// # Example
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 1D tensor with 5 elements
    ///     let tensor = Tensor::<B, 1>::from_data([0.0, 1.0, 2.0, 3.0, 4.0], &device);
    ///     // Split the tensor into chunks with sizes [2, 3] along dimension 0
    ///     let chunks = tensor.split_with_sizes(vec![2, 3], 0);
    ///     // The result is a vector of tensors:
    ///     // [Tensor([0.0, 1.0]), Tensor([2.0, 3.0, 4.0])]
    ///     println!("{:?}", chunks);
    /// }
    /// ```
    pub fn split_with_sizes(self, split_sizes: Vec<usize>, dim: usize) -> Vec<Self> {
        check!(TensorCheck::split_with_sizes::<D>(
            &self.shape(),
            &split_sizes,
            dim
        ));
        let mut tensors = Vec::new();

        let mut start = 0;
        for length in split_sizes {
            if length == 0 {
                continue;
            }
            tensors.push(Self::narrow(self.clone(), dim, start, length));
            start += length;
        }

        tensors
    }

    /// Tests if any element in the `tensor` evaluates to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test. All input tensor types (Float, Int, Bool) are supported.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, 1, Bool>` containing a single element, True if any element in the input tensor
    /// evaluates to True, False otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example<B: Backend>() {
    ///   let device = Default::default();
    ///   let tensor = Tensor::<B,2, Bool>::from_data([[true,false,true],[false,true,false]], &device);
    ///   let tensor_two = Tensor::<B,2, Bool>::from_data([[false,false,false],[false,false,false]], &device);
    ///
    ///   // Given a 2D tensor with dimensions [2, 3], test if any element in the tensor evaluates to True.
    ///   let any_tensor = tensor.any();
    ///   println!("{}", any_tensor);
    ///   // Tensor { data: [true], ... }
    ///
    ///   // Given a 2D tensor with dimensions [2, 3], test if any element in the tensor evaluates to True.
    ///   let any_tensor_two = tensor_two.any();
    ///   println!("{}", any_tensor_two);
    ///   // Tensor { data: [false], ... }
    /// }
    /// ```
    pub fn any(self) -> Tensor<B, 1, Bool> {
        Tensor::new(K::any(self.primitive))
    }

    /// Tests if any element in the `tensor` evaluates to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test. All input tensor types (Float, Int, Bool) are supported.
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, D, Bool>` with the same shape as input `tensor`, except in the `dim` axis
    /// where the size is 1. The elem in the `dim` axis is True if any element along this dim in the input
    /// evaluates to True, False otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let tensor =
    ///         Tensor::<B, 2, Bool>::from_data([[true, false, false], [false, true, false]], &device);
    ///     // Check if any element in the tensor evaluates to True along the dimension 1.
    ///     // [[true], [true]],
    ///     let any_dim = tensor.clone().any_dim(1);
    ///     println!("{any_dim}");
    /// }
    /// ```
    pub fn any_dim(self, dim: usize) -> Tensor<B, D, Bool> {
        Tensor::new(K::any_dim(self.primitive, dim))
    }

    /// Tests if all elements in the `tensor` evaluate to True.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test. All input tensor types (Float, Int, Bool) are supported.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, 1, Bool>` with a single element, True if all elements in the input tensor
    /// evaluate to True, False otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let tensor =
    ///         Tensor::<B, 2, Bool>::from_data([[true, false, true], [true, true, true]], &device);
    ///     // Check if all elements in the tensor evaluate to True (which is not the case).
    ///     // [false]
    ///     let all = tensor.all();
    ///     println!("{all}");
    /// }
    /// ```
    pub fn all(self) -> Tensor<B, 1, Bool> {
        Tensor::new(K::all(self.primitive))
    }

    /// Tests if all elements in the `tensor` evaluate to True along a given dimension `dim`.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to test. All input tensor types (Float, Int, Bool) are supported.
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor `Tensor<B, D, Bool>` with the same shape as input `tensor`, except in the `dim` axis
    /// where the size is 1. The elem in the `dim` axis is True if all elements along this dim in the input
    /// evaluates to True, False otherwise.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Bool};
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let tensor =
    ///         Tensor::<B, 2, Bool>::from_data([[true, true, false], [true, true, true]], &device);
    ///     // Check if all elements in the tensor evaluate to True along the dimension 1.
    ///     // [[true, true, false]]
    ///     let all_dim = tensor.clone().all_dim(0);
    ///     println!("{all_dim}");
    /// }
    /// ```
    pub fn all_dim(self, dim: usize) -> Tensor<B, D, Bool> {
        Tensor::new(K::all_dim(self.primitive, dim))
    }

    /// Convert the tensor into a scalar.
    ///
    /// # Panics
    ///
    /// - If the tensor doesn't have one element.
    /// - If the backend fails to read the tensor data synchronously.
    ///
    /// # Returns
    ///
    /// The scalar value of the tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     let tensor = Tensor::<B, 2>::from_data([[3.0]], &device);
    ///     // Convert the tensor with a single element into a scalar.
    ///     let scalar = tensor.into_scalar();
    ///     println!("{scalar}");
    /// }
    /// ```
    pub fn into_scalar(self) -> K::Elem {
        crate::try_read_sync(self.into_scalar_async())
            .expect(
            "Failed to read tensor data synchronously. This can happen on platforms
            that don't support blocking futures like WASM. Try into_scalar_async instead.",
            )
            .expect("Error while reading data: use `try_into_scalar` instead to catch the error at runtime")
    }

    /// Convert the tensor into a scalar and returns any error that might have occurred since the
    /// last time the device was synchronized.
    ///
    /// # Panics
    ///
    /// - If the tensor doesn't have one element.
    /// - If the backend fails to read the tensor data synchronously.
    ///
    /// # Returns
    ///
    /// The scalar value of the tensor.
    pub fn try_into_scalar(self) -> Result<K::Elem, ExecutionError> {
        crate::try_read_sync(self.into_scalar_async()).expect(
            "Failed to read tensor data synchronously. This can happen on platforms
            that don't support blocking futures like WASM. Try into_scalar_async instead.",
        )
    }

    /// Convert the tensor into a scalar.
    ///
    /// # Panics
    ///
    /// If the tensor doesn't have one element.
    pub async fn into_scalar_async(self) -> Result<K::Elem, ExecutionError> {
        check!(TensorCheck::into_scalar::<D>(&self.shape()));

        Ok(self.into_data_async().await?.iter().next().unwrap())
    }

    /// Broadcast the tensor to the given shape.
    ///
    /// Only singleton dimensions can be expanded to a larger size. Other dimensions must have the same size
    /// (which can be inferred with `-1`).
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape to broadcast the tensor to.
    ///   Can contain -1 for dimensions that should be inferred.
    ///   The number of elements in the shape must be greater or equal as
    ///   the number of dimensions of the tensor.
    ///
    /// # Panics
    ///
    /// If the tensor cannot be broadcasted to the given shape.
    ///
    /// # Returns
    ///
    /// A new tensor with the given shape.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let device = Default::default();
    ///     // Create a 2D tensor with dimensions [3, 1]
    ///     let tensor = Tensor::<B, 2>::from_data([[1.], [2.], [3.]], &device);
    ///     // Expand the tensor to a new shape [3, 4]
    ///     // [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]]
    ///     let expanded = tensor.expand([3, 4]);
    ///     println!("{}", expanded);
    /// }
    /// ```
    pub fn expand<const D2: usize, S: BroadcastArgs<D, D2>>(self, shape: S) -> Tensor<B, D2, K> {
        let shape = shape.into_shape(&self.shape());
        check!(TensorCheck::expand::<D, D2>(
            "expand",
            &self.shape(),
            &shape,
        ));

        Tensor::<B, D2, K>::new(K::expand(self.primitive, shape))
    }

    /// Unfold windows along a dimension.
    ///
    /// Returns a view of the tensor with all complete windows of size `size` in dimension `dim`;
    /// where windows are advanced by `step` at each index.
    ///
    /// The number of windows is `max(0, (shape[dim] - size).ceil_div(step))`.
    ///
    /// The new view will have the unfolded dimension replaced by two dimensions;
    /// one in the position of the original dimension, with size equal to the number of windows,
    /// and one appended to the right-most position, with size equal to `size`.
    ///
    /// # Warning
    ///
    /// For the `ndarray` and `candle` backends; this is not a view but a copy
    /// with duplicated data.
    ///
    /// # Arguments
    ///
    /// * `dim` - the dimension to unfold.
    /// * `size` - the size of each unfolded window.
    /// * `step` - the step between each window.
    ///
    /// # Returns
    ///
    /// A tensor view with the shape ``[pre=..., windows, post=..., size]``.
    pub fn unfold<const D2: usize, I: AsIndex>(
        self,
        dim: I,
        size: usize,
        step: usize,
    ) -> Tensor<B, D2, K> {
        let dim = canonicalize_dim(dim, D, false);
        check!(TensorCheck::unfold::<D, D2>(
            "unfold",
            &self.shape(),
            dim,
            size,
            step,
        ));
        Tensor::<B, D2, K>::new(K::unfold(self.primitive, dim, size, step))
    }
}

/// Iterator given by (Tensor::iter_dim).
pub struct DimIter<B, const D: usize, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    start: usize,
    end: usize,
    dim: usize,
    ranges: [Range<usize>; D],
    tensor: Tensor<B, D, K>,
}

impl<B: Backend, const D: usize, K: BasicOps<B>> Iterator for DimIter<B, D, K> {
    type Item = Tensor<B, D, K>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            return None;
        }

        let mut ranges = self.ranges.clone();
        ranges[self.dim] = self.start..(self.start + 1);

        let slice = self.tensor.clone().slice(ranges);
        self.start += 1;

        Some(slice)
    }
}

impl<B: Backend, const D: usize, K: BasicOps<B>> DoubleEndedIterator for DimIter<B, D, K> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            return None;
        }

        let mut ranges = self.ranges.clone();
        ranges[self.dim] = (self.end - 1)..self.end;

        let slice = self.tensor.clone().slice(ranges);
        self.end = self.end.saturating_sub(1);

        Some(slice)
    }
}

impl<B: Backend, const D: usize, K: BasicOps<B>> DimIter<B, D, K> {
    fn new(tensor: Tensor<B, D, K>, dim: usize) -> Self {
        let dims = tensor.dims();
        let ranges = dims
            .iter()
            .map(|&dim| 0..dim)
            .collect::<Vec<Range<usize>>>();
        let ranges: [Range<usize>; D] = ranges.try_into().unwrap();
        Self {
            end: dims[dim],
            ranges,
            start: 0,
            dim,
            tensor,
        }
    }
}

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
    <K as BasicOps<B>>::Elem: Debug,
{
    #[inline]
    fn push_newline_indent(acc: &mut String, indent: usize) {
        acc.push('\n');
        for _ in 0..indent {
            acc.push(' ');
        }
    }
    fn fmt_inner_tensor(
        &self,
        acc: &mut String,
        depth: usize,
        multi_index: &mut [usize],
        range: (usize, usize),
        precision: Option<usize>,
    ) {
        let (start, end) = range;
        for i in start..end {
            if i > 0 {
                acc.push_str(", ");
            }
            multi_index[depth] = i;
            let range: [Range<usize>; D] =
                core::array::from_fn(|i| multi_index[i]..multi_index[i] + 1);

            let data = burn_std::reader::try_read_sync(self.clone().slice(range).into_data_async());

            if let Some(Ok(data)) = data {
                let elem = data.iter::<<K as BasicOps<B>>::Elem>().next().unwrap();
                match (precision, K::name()) {
                    (Some(p), "Float") => acc.push_str(&format!("{elem:.p$}")),
                    (_, "Bool") => acc.push_str(&format!("{}", elem.to_bool())),
                    _ => acc.push_str(&format!("{elem:?}")),
                }
            } else {
                acc.push_str("<Tensor data not available>");
            }
        }
    }

    fn fmt_outer_tensor(
        &self,
        acc: &mut String,
        depth: usize,
        multi_index: &mut [usize],
        print_options: &PrintOptions,
        summarize: bool,
        range: (usize, usize),
    ) {
        let (start, end) = range;
        for i in start..end {
            if i > start {
                acc.push(',');
                Self::push_newline_indent(acc, depth + 1);
            }
            acc.push('[');
            multi_index[depth] = i;
            self.display_recursive(acc, depth + 1, multi_index, print_options, summarize);
            acc.push(']');
        }
    }

    /// Recursively formats the tensor data for display and appends it to the provided accumulator string.
    ///
    /// This function is designed to work with tensors of any dimensionality.
    /// It traverses the tensor dimensions recursively, converting the elements
    /// to strings and appending them to the accumulator string with the
    /// appropriate formatting.
    ///
    /// # Arguments
    ///
    /// * `acc` - A mutable reference to a `String` used as an accumulator for the formatted output.
    /// * `depth` - The current depth of the tensor dimensions being processed.
    /// * `multi_index` - A mutable slice of `usize` representing the current indices in each dimension.
    fn display_recursive(
        &self,
        acc: &mut String,
        depth: usize,
        multi_index: &mut [usize],
        print_options: &PrintOptions,
        summarize: bool,
    ) {
        let edge_items = print_options.edge_items;

        if depth == 0 {
            acc.push('[');
        }

        if depth == self.dims().len() - 1 {
            // if we are at the innermost dimension, just push its elements into the accumulator
            if summarize && self.dims()[depth] > 2 * edge_items {
                // print the starting `edge_items` elements
                self.fmt_inner_tensor(
                    acc,
                    depth,
                    multi_index,
                    (0, edge_items),
                    print_options.precision,
                );
                acc.push_str(", ...");
                // print the last `edge_items` elements
                self.fmt_inner_tensor(
                    acc,
                    depth,
                    multi_index,
                    (self.dims()[depth] - edge_items, self.dims()[depth]),
                    print_options.precision,
                );
            } else {
                // print all the elements
                self.fmt_inner_tensor(
                    acc,
                    depth,
                    multi_index,
                    (0, self.dims()[depth]),
                    print_options.precision,
                );
            }
        } else {
            // otherwise, iterate through the current dimension and recursively display the inner tensors
            if summarize && self.dims()[depth] > 2 * edge_items {
                self.fmt_outer_tensor(
                    acc,
                    depth,
                    multi_index,
                    print_options,
                    summarize,
                    (0, edge_items),
                );

                acc.push(',');
                Self::push_newline_indent(acc, depth + 1);
                acc.push_str("...");
                Self::push_newline_indent(acc, depth + 1);

                self.fmt_outer_tensor(
                    acc,
                    depth,
                    multi_index,
                    print_options,
                    summarize,
                    (self.dims()[depth] - edge_items, self.dims()[depth]),
                );
            } else {
                self.fmt_outer_tensor(
                    acc,
                    depth,
                    multi_index,
                    print_options,
                    summarize,
                    (0, self.dims()[depth]),
                );
            }
        }

        if depth == 0 {
            acc.push(']');
        }
    }
}

#[derive(Clone, Debug)]
/// Options for Tensor pretty printing
pub struct PrintOptions {
    /// number of elements to start summarizing tensor
    pub threshold: usize,

    /// number of starting elements and ending elements to display
    pub edge_items: usize,

    /// Precision for floating point numbers
    pub precision: Option<usize>,
}

static PRINT_OPTS: RwLock<PrintOptions> = RwLock::new(PrintOptions::const_default());

impl PrintOptions {
    /// Print options with default values
    pub const fn const_default() -> Self {
        Self {
            threshold: 1000,
            edge_items: 3,
            precision: None,
        }
    }
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self::const_default()
    }
}

/// Set print options
pub fn set_print_options(options: PrintOptions) {
    let mut print_opts = PRINT_OPTS.write().unwrap();
    *print_opts = options;
}

/// Pretty print tensors
impl<B, const D: usize, K> core::fmt::Display for Tensor<B, D, K>
where
    B: Backend,
    B::IntElem: core::fmt::Display,
    K: BasicOps<B>,
    <K as BasicOps<B>>::Elem: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Tensor {{")?;

        {
            // Do not lock the mutex for the whole function
            let mut po = { PRINT_OPTS.read().unwrap().clone() };

            // Override the precision if it is set from the formatter
            // This will be possible when the tensor is printed using the `{:.*}` syntax
            if let Some(precision) = f.precision() {
                po.precision = Some(precision);
            }

            let mut acc = String::new();
            let mut multi_index = vec![0; D];
            let summarize = self.shape().num_elements() > po.threshold;

            self.display_recursive(&mut acc, 0, &mut multi_index, &po, summarize);

            writeln!(f, "  data:")?;
            write!(f, "{acc}")?;
            writeln!(f, ",")?;
        }

        writeln!(f, "  shape:  {:?},", self.dims())?;
        writeln!(f, "  device:  {:?},", self.device())?;
        writeln!(f, "  backend:  {:?},", B::name(&self.device()))?;
        writeln!(f, "  kind:  {:?},", K::name())?;

        let dtype = self.primitive.dtype();

        writeln!(f, "  dtype:  {:?},", dtype.name())?;
        write!(f, "}}")
    }
}

/// Trait used for movedim arguments
pub trait MovedimArgs {
    /// Converts into a set of dimensions `Vec<usize>` for the `tensor.movedim()` function
    fn into_dim_vec<const D: usize>(self) -> Vec<usize>;
}

impl MovedimArgs for Vec<i32> {
    fn into_dim_vec<const D: usize>(self) -> Vec<usize> {
        let set = self
            .iter()
            .map(|&dim| {
                if dim < 0 {
                    (D as i32 + dim) as usize
                } else {
                    dim as usize
                }
            })
            .collect::<Vec<usize>>();
        check!(TensorCheck::movedim_args_vec::<D>(&set));

        set
    }
}

impl MovedimArgs for Vec<usize> {
    fn into_dim_vec<const D: usize>(self) -> Vec<usize> {
        check!(TensorCheck::movedim_args_vec::<D>(&self));
        self
    }
}

impl MovedimArgs for usize {
    #[allow(clippy::vec_init_then_push)]
    fn into_dim_vec<const D: usize>(self) -> Vec<usize> {
        check!(TensorCheck::movedim_args_usize::<D>(self));

        let mut set = Vec::with_capacity(1);
        set.push(self);

        set
    }
}

impl MovedimArgs for i32 {
    #[allow(clippy::vec_init_then_push)]
    fn into_dim_vec<const D: usize>(self) -> Vec<usize> {
        check!(TensorCheck::movedim_args_i32::<D>(self));

        let dim = if self < 0 {
            (D as i32 + self) as usize
        } else {
            self as usize
        };

        let mut set = Vec::with_capacity(1);
        set.push(dim);

        set
    }
}

/// Trait used for reshape arguments.
pub trait ReshapeArgs<const D2: usize> {
    /// Converts to a shape.
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape;
}

impl<const D2: usize> ReshapeArgs<D2> for Shape {
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape {
        check!(TensorCheck::reshape_args_usize::<D, D2>(
            &tensor.shape(),
            &self
        ));

        self
    }
}
impl<const D2: usize> ReshapeArgs<D2> for [usize; D2] {
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape {
        let shape = Shape::from(self);

        check!(TensorCheck::reshape_args_usize::<D, D2>(
            &tensor.shape(),
            &shape
        ));

        shape
    }
}

impl<const D2: usize> ReshapeArgs<D2> for [i64; D2] {
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape {
        // Validate the reshape arguments
        check!(TensorCheck::reshape_args_i64(&self));

        // Temporary shape
        let mut new_shape: [i64; D2] = [1; D2];

        // We need to find the index of the 0 dimension and
        // replace it with the actual dimension value.
        for (i, &s) in self.iter().enumerate() {
            if s != 0 {
                new_shape[i] = s;
            } else {
                new_shape[i] = tensor.dims()[i] as i64;
            }
        }

        // Find the index of the inferred dimension (-1)
        let infer_index = new_shape.iter().position(|x| x == &-1);

        // Handle the case where the dimension is inferred (via -1)
        if let Some(index) = infer_index {
            // Handle the case where the dimension is inferred
            let mut product = 1;
            for (i, &s) in new_shape.iter().enumerate() {
                if i != index {
                    product *= s;
                }
            }
            let product_current = tensor.shape().num_elements() as i64;

            new_shape[index] = product_current / product;

            // Check if the reshape is valid
            if product_current % product != 0 {
                panic!(
                    "Cannot reshape tensor of shape {:?} to shape {:?}",
                    tensor.shape(),
                    new_shape
                );
            }
        };

        // Convert each element to usize
        let new_shape: [usize; D2] = new_shape.map(|x| x as usize);

        Shape::from(new_shape)
    }
}

impl<const D2: usize> ReshapeArgs<D2> for [i32; D2] {
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape {
        // Convert i32 array to i64 array and use existing implementation
        let i64_array: [i64; D2] = self.map(|x| x as i64);
        ReshapeArgs::into_shape(i64_array, tensor)
    }
}

/// Trait used for broadcast arguments.
pub trait BroadcastArgs<const D1: usize, const D2: usize> {
    /// Converts to a shape.
    fn into_shape(self, shape: &Shape) -> Shape;
}

impl<const D1: usize, const D2: usize> BroadcastArgs<D1, D2> for Shape {
    fn into_shape(self, _shape: &Shape) -> Shape {
        self
    }
}

impl<const D1: usize, const D2: usize, E: AsIndex> BroadcastArgs<D1, D2> for [E; D2] {
    // Passing -1 as the size for a dimension means not changing the size of that dimension.
    fn into_shape(self, shape: &Shape) -> Shape {
        if self.len() < shape.num_dims() {
            panic!("Broadcast arguments must be greater than the number of dimensions");
        }

        // Zip the two shapes in reverse order and replace -1 with the actual dimension value.
        let new_shape: Vec<_> = self
            .iter()
            .rev()
            .map(|x| {
                let primitive = x.index();
                if primitive < -1 || primitive == 0 {
                    panic!("Broadcast arguments must be positive or -1");
                }
                primitive
            })
            .zip(shape.iter().rev().chain(repeat(&0)).take(self.len())) // Pad the original shape with 0s
            .map(|(x, &y)| if x == -1 { y } else { x as usize })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        if new_shape.contains(&0) {
            panic!("Cannot substitute -1 for a non-existing dimension");
        }

        let new_shape: [usize; D2] = new_shape.try_into().unwrap();

        Shape::from(new_shape)
    }
}

impl<B, const D: usize, K> Serialize for Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
    K::Elem: Debug + Copy + Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let data = self.to_data();
        data.serialize(serializer)
    }
}

impl<'de, B, const D: usize, K> Deserialize<'de> for Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
    K::Elem: Debug + Copy + Deserialize<'de>,
{
    fn deserialize<De: Deserializer<'de>>(deserializer: De) -> Result<Self, De::Error> {
        let tensor = Tensor::from_data(
            TensorData::deserialize(deserializer)?,
            &<B::Device as Default>::default(),
        );
        Ok(tensor)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Shape, s};

    #[test]
    fn slice_range_single_dim_leading() {
        let shape = Shape::new([8, 4]);

        // Half-open range
        let slices = shape.clone().into_slices([0..5]);
        assert_eq!(slices[0].to_range(8), 0..5);
        let slices = shape.clone().into_slices([-3..-1]);
        assert_eq!(slices[0].to_range(8), 5..7);

        // Inclusive range
        let slices = shape.clone().into_slices([0..=4]);
        assert_eq!(slices[0].to_range(8), 0..5);
        let slices = shape.clone().into_slices([-2..=-1]);
        assert_eq!(slices[0].to_range(8), 6..8);

        // Unbounded start
        let slices = shape.clone().into_slices([..3]);
        assert_eq!(slices[0].to_range(8), 0..3);
        let slices = shape.clone().into_slices([..-5]);
        assert_eq!(slices[0].to_range(8), 0..3);

        // Unbounded end
        let slices = shape.clone().into_slices([5..]);
        assert_eq!(slices[0].to_range(8), 5..8);
        let slices = shape.clone().into_slices([-3..]);
        assert_eq!(slices[0].to_range(8), 5..8);

        // Full range
        let slices = shape.into_slices([..]);
        assert_eq!(slices[0].to_range(8), 0..8);
    }

    #[test]
    fn test_negative_slice_indices() {
        use crate::Slice;

        // Test negative indices conversion
        let slice: Slice = (-3..-1).into();
        assert_eq!(slice.start, -3);
        assert_eq!(slice.end, Some(-1));

        // Test to_range conversion with size 8
        let range = slice.to_range(8);
        assert_eq!(range, 5..7);

        // Test with shape slice
        let shape = Shape::new([8, 4]);
        let result = shape.clone().into_slices([-3..-1]);
        assert_eq!(result[0].to_range(8), 5..7);

        // Test more negative index cases
        let slice2: Slice = (-5..).into();
        assert_eq!(slice2.to_range(10), 5..10);

        let slice3: Slice = (..-2).into();
        assert_eq!(slice3.to_range(10), 0..8);

        // Test with s! macro - single dimension returns Slice directly
        let slice4 = s![-3..-1];
        assert_eq!(slice4.start, -3);
        assert_eq!(slice4.end, Some(-1));
    }

    #[test]
    fn slice_range_multi_dim() {
        let shape = Shape::new([8, 4]);

        // Multiple ways to provide ranges
        let slices = shape.clone().into_slices([0..5, 0..4]);
        assert_eq!(slices[0].to_range(8), 0..5);
        assert_eq!(slices[1].to_range(4), 0..4);

        let slices = shape.clone().into_slices([0.., 0..]);
        assert_eq!(slices[0].to_range(8), 0..8);
        assert_eq!(slices[1].to_range(4), 0..4);

        let slices = shape.clone().into_slices([0..=7, 0..=3]);
        assert_eq!(slices[0].to_range(8), 0..8);
        assert_eq!(slices[1].to_range(4), 0..4);

        let slices = shape.clone().into_slices([0..5, 0..3]);
        assert_eq!(slices[0].to_range(8), 0..5);
        assert_eq!(slices[1].to_range(4), 0..3);

        let slices = shape.into_slices([0.., 0..]);
        assert_eq!(slices[0].to_range(8), 0..8);
        assert_eq!(slices[1].to_range(4), 0..4);
    }

    #[test]
    fn slice_range_multi_dim_index() {
        let shape = Shape::new([8, 4]);

        // Indices (single integer) should also convert to correct range
        let slices = shape.clone().into_slices([0, 2]);
        assert_eq!(slices[0].to_range(8), 0..1);
        assert_eq!(slices[1].to_range(4), 2..3);

        let slices = shape.into_slices([-1, -1]);
        assert_eq!(slices[0].to_range(8), 7..8);
        assert_eq!(slices[1].to_range(4), 3..4);
    }

    #[test]
    fn slice_range_multi_dim_heterogeneous() {
        // Slice macro `s![]` can be used to provide different range types
        let shape = Shape::new([8, 4, 2]);
        let slice = s![0..5, .., -1];
        let slices = shape.into_slices(slice);
        assert_eq!(slices[0].to_range(8), 0..5);
        assert_eq!(slices[1].to_range(4), 0..4);
        assert_eq!(slices[2].to_range(2), 1..2);

        let shape = Shape::new([8, 4, 2, 3]);
        let slice = s![..=4, 0..=3, .., -2..];
        let slices = shape.into_slices(slice);
        assert_eq!(slices[0].to_range(8), 0..5);
        assert_eq!(slices[1].to_range(4), 0..4);
        assert_eq!(slices[2].to_range(2), 0..2);
        assert_eq!(slices[3].to_range(3), 1..3);

        let shape = Shape::new([3, 4]);
        let slice = s![1..-1, ..];
        let slices = shape.into_slices(slice);
        assert_eq!(slices[0].to_range(3), 1..2);
        assert_eq!(slices[1].to_range(4), 0..4);
    }
}
