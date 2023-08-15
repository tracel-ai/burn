use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::{fmt::Debug, ops::Range};

use crate::{
    backend::Backend, check, check::TensorCheck, Bool, Data, Float, Int, Shape, TensorKind,
};

/// A tensor with a given backend, shape and data type.
#[derive(new, Clone, Debug)]
pub struct Tensor<B, const D: usize, K = Float>
where
    B: Backend,
    K: TensorKind<B>,
{
    pub(crate) primitive: K::Primitive<D>,
}

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    /// Converts the tensor into a primitive tensor.
    pub fn into_primitive(self) -> K::Primitive<D> {
        self.primitive
    }

    /// Converts from a primitive tensor into a tensor.
    pub fn from_primitive(tensor: K::Primitive<D>) -> Self {
        Self::new(tensor)
    }

    /// Create an empty tensor of the given shape.
    pub fn empty<S: Into<Shape<D>>>(shape: S) -> Self {
        Self::empty_device(shape, &B::Device::default())
    }

    /// Create an empty tensor of the given shape.
    pub fn empty_device<S: Into<Shape<D>>>(shape: S, device: &B::Device) -> Self {
        Self::new(K::empty(shape.into(), device))
    }

    /// Returns the dimensions of the current tensor.
    ///
    /// Equivalent to `tensor.shape().dims`.
    pub fn dims(&self) -> [usize; D] {
        Self::shape(self).dims
    }

    /// Returns the shape of the current tensor.
    pub fn shape(&self) -> Shape<D> {
        K::shape(&self.primitive)
    }

    /// Reshape the tensor to have the given shape.
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
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///    let tensor = Tensor::<B, 3>::ones([2, 3, 4]);
    ///    // Given a 3D tensor with dimensions (2, 3, 4), reshape it to (2, 12)
    ///    let reshaped_tensor: Tensor::<B, 2> = tensor.reshape([2, -1]);
    ///    // The resulting tensor will have dimensions (2, 12).
    ///    println!("{:?}", reshaped_tensor.shape());
    /// }
    /// ```
    pub fn reshape<const D2: usize, S: ReshapeArgs<D2>>(self, shape: S) -> Tensor<B, D2, K> {
        // Convert reshape args to shape
        let shape = shape.into_shape(&self);
        Tensor::new(K::reshape::<D, D2>(self.primitive, shape))
    }

    /// Flatten the tensor along a given range of dimensions.
    ///
    /// This function collapses the specified range of dimensions into a single dimension,
    /// effectively flattening the tensor in that range.
    ///
    /// # Arguments
    ///
    /// - `start_dim`: The starting dimension of the range to be flattened.
    /// - `end_dim`: The ending dimension of the range to be flattened (inclusive).
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
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 4]));
    ///
    ///     // Given a 3D tensor with dimensions (2, 3, 4), flatten the dimensions between indices 1 and 2:
    ///     let flattened_tensor: Tensor::<B, 2> = tensor.flatten(1, 2);
    ///
    ///     // The resulting tensor will have dimensions (2, 12).
    ///    println!("{:?}", flattened_tensor.shape());
    /// }
    ///
    /// ```
    pub fn flatten<const D2: usize>(self, start_dim: usize, end_dim: usize) -> Tensor<B, D2, K> {
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

        Tensor::new(K::reshape::<D, D2>(self.primitive, new_dims.into()))
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
    ///  - 'D2': The resulting number of dimensions in the squeezed tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor<B, D2, K>` instance with the specified dimenension removed.
    ///
    /// # Example
    ///
    /// ```rust
    ///
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 1, 4]));
    ///
    ///     // Given a 3D tensor with dimensions (2, 1, 4), squeeze the dimension 1
    ///     let squeezed_tensor: Tensor::<B, 2> = tensor.squeeze(1);
    ///
    ///     // Resulting tensor will have dimensions (2, 4)
    ///     println!("{:?}", squeezed_tensor.shape());
    /// }
    /// ```
    pub fn squeeze<const D2: usize>(self, dim: usize) -> Tensor<B, D2, K> {
        check!(TensorCheck::squeeze::<D2>(dim, &self.shape().dims));

        let current_dims = self.shape().dims;
        let mut new_dims: [usize; D2] = [0; D2];

        new_dims[..dim].copy_from_slice(&current_dims[..dim]);
        new_dims[dim..].copy_from_slice(&current_dims[dim + 1..]);

        Tensor::new(K::reshape::<D, D2>(self.primitive, new_dims.into()))
    }

    /// Unsqueeze the current tensor. Create new dimensions to fit the given size.
    ///
    /// # Panics
    ///
    /// If the output size is higher than the current tensor.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let tensor = Tensor::<B, 2>::ones(Shape::new([3, 3]));
    ///     let tensor = tensor.unsqueeze::<4>();
    ///     println!("{:?}", tensor.shape());
    ///     // Shape { dims: [1, 1, 3, 3] }
    /// }
    /// ```
    pub fn unsqueeze<const D2: usize>(self) -> Tensor<B, D2, K> {
        check!(TensorCheck::unsqueeze::<D, D2>());

        let mut dims = [1; D2];
        let num_ones = D2 - D;
        let shape = self.shape();

        dims[num_ones..(D + num_ones)].copy_from_slice(&shape.dims[..D]);

        let shape = Shape::new(dims);
        self.reshape(shape)
    }

    /// Returns a tensor containing the elements selected from the given ranges.
    ///
    /// # Panics
    ///
    /// If a range exceeds the number of elements on a dimension.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape};
    ///
    /// fn example<B: Backend>() {
    ///     let tensor = Tensor::<B, 3>::ones(Shape::new([2, 3, 3]));
    ///     let tensor_slices = tensor.slice([0..1, 0..3, 1..2]);
    ///     println!("{:?}", tensor_slices.dims()); // [1, 3, 2]
    ///     
    /// }
    /// ```
    pub fn slice<const D2: usize>(self, ranges: [core::ops::Range<usize>; D2]) -> Self {
        check!(TensorCheck::slice(&self.shape(), &ranges));
        Self::new(K::slice(self.primitive, ranges))
    }

    /// Returns a copy of the current tensor with the selected elements changed to the new ones at
    /// the selected indices.
    ///
    /// # Panics
    ///
    /// - If a range exceeds the number of elements on a dimension.
    /// - If the given values don't match the given ranges.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///     let tensor = Tensor::<B, 3>::ones([2, 3, 3]);
    ///     let values = Tensor::<B, 3>::zeros([1, 1, 1]);
    ///     let tensor_sliced = tensor.slice_assign([0..1, 0..1, 0..1], values);
    ///     println!("{:?}", tensor_sliced.dims()); // [2, 3, 3]
    /// }
    /// ```
    pub fn slice_assign<const D2: usize>(
        self,
        ranges: [core::ops::Range<usize>; D2],
        values: Self,
    ) -> Self {
        check!(TensorCheck::slice_assign(
            &self.shape(),
            &values.shape(),
            &ranges
        ));
        Self::new(K::slice_assign(self.primitive, ranges, values.primitive))
    }

    /// Returns the device of the current tensor.
    pub fn device(&self) -> B::Device {
        K::device(&self.primitive)
    }

    /// Returns a new tensor on the given device.
    pub fn to_device(self, device: &B::Device) -> Self {
        Self::new(K::to_device(self.primitive, device))
    }

    /// Returns the data of the current tensor.
    pub fn into_data(self) -> Data<K::Elem, D> {
        K::into_data(self.primitive)
    }

    /// Returns the data of the current tensor without taking ownership.
    pub fn to_data(&self) -> Data<K::Elem, D> {
        Self::into_data(self.clone())
    }

    /// Create a tensor from the given data.
    pub fn from_data<T>(data: T) -> Self
    where
        T: Into<Data<K::Elem, D>>,
    {
        Self::from_data_device(data, &B::Device::default())
    }

    /// Create a tensor from the given data on the given device.
    pub fn from_data_device<T>(data: T, device: &B::Device) -> Self
    where
        T: Into<Data<K::Elem, D>>,
    {
        Self::new(K::from_data(data.into(), device))
    }

    /// Repeat the tensor along the given dimension.
    ///
    /// # Panics
    ///
    /// If the selected dimension more than one item.
    pub fn repeat(self, dim: usize, times: usize) -> Self {
        Self::new(K::repeat(self.primitive, dim, times))
    }

    /// Applies element wise equal comparison and returns a boolean tensor.
    ///
    /// # Panics
    ///
    /// If the two tensors don't have the same shape.
    pub fn equal(self, other: Self) -> Tensor<B, D, Bool> {
        check!(TensorCheck::binary_ops_ew("Equal", &self, &other));
        K::equal(self.primitive, other.primitive)
    }

    /// Concatenates all tensors into a new one along the given dimension.
    ///
    /// # Panics
    ///
    /// If all tensors don't have the same shape.
    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
        check!(TensorCheck::cat(&tensors, dim));

        Self::new(K::cat(
            tensors.into_iter().map(|vector| vector.primitive).collect(),
            dim,
        ))
    }
}

impl<B, const D: usize, K> Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
    <K as BasicOps<B>>::Elem: Debug,
{
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
    fn display_recursive(&self, acc: &mut String, depth: usize, multi_index: &mut [usize]) {
        if depth == 0 {
            acc.push('[');
        }

        if depth == self.dims().len() - 1 {
            // if we are at the innermost dimension, just push its elements into the accumulator
            for i in 0..self.dims()[depth] {
                if i > 0 {
                    acc.push_str(", ");
                }
                multi_index[depth] = i;
                let range: [core::ops::Range<usize>; D] =
                    core::array::from_fn(|i| multi_index[i]..multi_index[i] + 1);
                let elem = &self.clone().slice(range).to_data().value[0];
                acc.push_str(&format!("{elem:?}"));
            }
        } else {
            // otherwise, iterate through the current dimension and recursively display the inner tensors
            for i in 0..self.dims()[depth] {
                if i > 0 {
                    acc.push_str(", ");
                }
                acc.push('[');
                multi_index[depth] = i;
                self.display_recursive(acc, depth + 1, multi_index);
                acc.push(']');
            }
        }

        if depth == 0 {
            acc.push(']');
        }
    }
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
        write!(f, "  data: ")?;

        let mut acc = String::new();
        let mut multi_index = vec![0; D];
        self.display_recursive(&mut acc, 0, &mut multi_index);
        write!(f, "{acc}")?;
        writeln!(f, ",")?;
        writeln!(f, "  shape:  {:?},", self.dims())?;
        writeln!(f, "  device:  {:?},", self.device())?;
        writeln!(f, "  backend:  {:?},", B::name())?;
        writeln!(f, "  kind:  {:?},", K::name())?;
        writeln!(f, "  dtype:  {:?},", K::elem_type_name())?;
        write!(f, "}}")
    }
}

/// Transpose marker (zero-size type). Used to sugar the transpose of a tensor, e.g.
/// ```rust
/// use burn_tensor::backend::Backend;
/// use burn_tensor::{Tensor, T};
///
/// fn example<B: Backend>() {
///     let tensor = Tensor::<B, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]]);
///     let transposed = tensor^T;
/// }
/// ```
pub struct T;

impl<B: Backend, const D: usize> core::ops::BitXor<T> for Tensor<B, D> {
    type Output = Self;
    fn bitxor(self, _: T) -> Self::Output {
        self.transpose()
    }
}

/// Trait that list all operations that can be applied on all tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait BasicOps<B: Backend>: TensorKind<B> {
    /// The type of the tensor elements.
    type Elem: 'static;

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
    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D>;

    /// Returns the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    ///
    /// # Returns
    ///
    /// The shape of the tensor.
    ///
    /// # Remarks
    ///
    /// This is a low-level function used internally by the library to call different backend functions
    /// with static dispatch. It is not designed for direct usage by users, and not recommended to import
    /// or use this function directly.
    ///
    /// For getting the shape of a tensor, users should prefer the [Tensor::shape](Tensor::shape) function,
    /// which is more high-level and designed for public use.
    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D>;

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
    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2>;

    ///  Select tensor elements corresponding for the given ranges.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `ranges` - The ranges of the elements to select.
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
    /// For selecting elements of a tensor, users should prefer the [Tensor::slice](Tensor::slice) function,
    /// which is more high-level and designed for public use.
    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        range: [Range<usize>; D2],
    ) -> Self::Primitive<D1>;

    ///  Assigns the given value to the tensor elements corresponding for the given ranges.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor.
    /// * `ranges` - The ranges of the elements to select.
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
    /// For assigning values to elements of a tensor, users should prefer the [Tensor::slice_assign](Tensor::slice_assign) function,
    /// which is more high-level and designed for public use.
    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1>;

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
    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> B::Device;

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
    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &B::Device,
    ) -> Self::Primitive<D>;

    /// Extracts the data from the tensor.
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
    fn into_data<const D: usize>(tensor: Self::Primitive<D>) -> Data<Self::Elem, D>;

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
    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: &B::Device,
    ) -> Self::Primitive<D>;

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
    /// For repeating a tensor, users should prefer the [Tensor::repeat](Tensor::repeat) function,
    /// which is more high-level and designed for public use.
    fn repeat<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D>;

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
    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D>;

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
    /// For equating tensors, users should prefer the [Tensor::equal](Tensor::equal) function,
    /// which is more high-level and designed for public use.
    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool>;

    /// Returns the name of the element type.
    fn elem_type_name() -> &'static str {
        core::any::type_name::<Self::Elem>()
    }
}

impl<B: Backend> BasicOps<B> for Float {
    type Elem = B::FloatElem;

    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::empty(shape, device)
    }
    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        B::shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        B::reshape(tensor, shape)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::slice(tensor, ranges)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        B::slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        B::device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::to_device(tensor, device)
    }

    fn into_data<const D: usize>(tensor: Self::Primitive<D>) -> Data<Self::Elem, D> {
        B::into_data(tensor)
    }

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: &B::Device,
    ) -> Self::Primitive<D> {
        B::from_data(data, device)
    }

    fn repeat<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D> {
        B::repeat(tensor, dim, times)
    }

    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D> {
        B::cat(vectors, dim)
    }

    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::equal(lhs, rhs))
    }
}

impl<B: Backend> BasicOps<B> for Int {
    type Elem = B::IntElem;

    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::int_empty(shape, device)
    }
    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        B::int_shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        B::int_reshape(tensor, shape)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::int_slice(tensor, ranges)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        B::int_slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        B::int_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::int_to_device(tensor, device)
    }

    fn into_data<const D: usize>(tensor: Self::Primitive<D>) -> Data<Self::Elem, D> {
        B::int_into_data(tensor)
    }

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: &B::Device,
    ) -> Self::Primitive<D> {
        B::int_from_data(data, device)
    }

    fn repeat<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D> {
        B::int_repeat(tensor, dim, times)
    }

    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_equal(lhs, rhs))
    }

    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D> {
        B::int_cat(vectors, dim)
    }
}

impl<B: Backend> BasicOps<B> for Bool {
    type Elem = bool;

    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D> {
        B::bool_empty(shape, device)
    }
    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        B::bool_shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        B::bool_reshape(tensor, shape)
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::bool_slice(tensor, ranges)
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        ranges: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        B::bool_slice_assign(tensor, ranges, value)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        B::bool_device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::bool_to_device(tensor, device)
    }

    fn into_data<const D: usize>(tensor: Self::Primitive<D>) -> Data<Self::Elem, D> {
        B::bool_into_data(tensor)
    }

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: &B::Device,
    ) -> Self::Primitive<D> {
        B::bool_from_data(data, device)
    }

    fn repeat<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D> {
        B::bool_repeat(tensor, dim, times)
    }

    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool> {
        Tensor::new(B::bool_equal(lhs, rhs))
    }

    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D> {
        B::bool_cat(vectors, dim)
    }
}

/// Trait used for reshape arguments.
pub trait ReshapeArgs<const D2: usize> {
    /// Converts to a shape.
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape<D2>;
}

impl<const D2: usize> ReshapeArgs<D2> for Shape<D2> {
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape<D2> {
        check!(TensorCheck::reshape_args_usize(&self, &tensor.shape()));

        self
    }
}
impl<const D2: usize> ReshapeArgs<D2> for [usize; D2] {
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape<D2> {
        let shape = Shape::from(self);

        check!(TensorCheck::reshape_args_usize(&shape, &tensor.shape()));

        shape
    }
}

impl<const D2: usize> ReshapeArgs<D2> for [i32; D2] {
    fn into_shape<B: Backend, const D: usize, K: BasicOps<B>>(
        self,
        tensor: &Tensor<B, D, K>,
    ) -> Shape<D2> {
        // Validate the reshape arguments
        check!(TensorCheck::reshape_args_i32(&self));

        // Temporary shape
        let mut new_shape: [i32; D2] = [1; D2];

        // We need to find the index of the 0 dimension and
        // replace it with the actual dimension value.
        for (i, &s) in self.iter().enumerate() {
            if s != 0 {
                new_shape[i] = s;
            } else {
                new_shape[i] = tensor.dims()[i] as i32;
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
            let product_current = tensor.shape().num_elements() as i32;

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
