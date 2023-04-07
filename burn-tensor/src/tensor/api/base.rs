use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::{fmt::Debug, ops::Range};

use crate::{backend::Backend, Bool, Data, Float, Int, Shape, TensorKind};

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
    /// # Panics
    ///
    /// If the tensor can not be reshape to the given shape.
    pub fn reshape<const D2: usize, S: Into<Shape<D2>>>(self, shape: S) -> Tensor<B, D2, K> {
        Tensor::new(K::reshape::<D, D2>(self.primitive, shape.into()))
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
        if start_dim > end_dim {
            panic!("The start dim ({start_dim}) must be smaller than the end dim ({end_dim})")
        }

        if D2 > D {
            panic!("Result dim ({D2}) must be smaller than ({D})")
        }

        if D < end_dim + 1 {
            panic!("The end dim ({end_dim}) must be greater than the tensor dim ({D2})")
        }

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
        if D2 < D {
            panic!("Can't unsqueeze smaller tensor, got dim {D2}, expected > {D}")
        }

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
    ///     let tensor_indexed = tensor.index([0..1, 0..3, 1..2]);
    ///     println!("{:?}", tensor_indexed.shape());
    ///     // Shape { dims: [1, 3, 2] }
    /// }
    /// ```
    pub fn index<const D2: usize>(self, indexes: [core::ops::Range<usize>; D2]) -> Self {
        Self::new(K::index(self.primitive, indexes))
    }

    /// Returns a copy of the current tensor with the selected elements changed to the new ones at
    /// the selected indexes.
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
    ///     let tensor_indexed = tensor.index_assign([0..1, 0..1, 0..1], values);
    ///     println!("{:?}", tensor_indexed.shape());
    ///     // Shape { dims: [2, 3, 3] }
    /// }
    /// ```
    pub fn index_assign<const D2: usize>(
        self,
        indexes: [core::ops::Range<usize>; D2],
        values: Self,
    ) -> Self {
        Self::new(K::index_assign(self.primitive, indexes, values.primitive))
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
    pub fn from_data(data: Data<K::Elem, D>) -> Self {
        Self::from_data_device(data, &B::Device::default())
    }

    /// Create a tensor from the given data on the given device.
    pub fn from_data_device(data: Data<K::Elem, D>, device: &B::Device) -> Self {
        Self::new(K::from_data(data, device))
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
        K::equal(self.primitive, other.primitive)
    }

    /// Applies element wise equal comparison and returns a boolean tensor.
    pub fn equal_elem<E: Into<K::Elem>>(self, other: E) -> Tensor<B, D, Bool> {
        let elem: K::Elem = other.into();
        K::equal_elem::<D>(self.primitive, elem)
    }

    /// Concatenates all tensors into a new one along the given dimension.
    ///
    /// # Panics
    ///
    /// If all tensors don't have the same shape.
    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
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
                let elem = &self.clone().index(range).to_data().value[0];
                acc.push_str(&format!("{:?}", elem));
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
        write!(f, "{}", acc)?;
        writeln!(f, ",")?;
        writeln!(f, "  shape:  {:?},", self.dims())?;
        writeln!(f, "  device:  {:?},", self.device())?;
        writeln!(f, "  backend:  {:?},", B::name())?;
        writeln!(f, "  kind:  {:?},", K::name())?;
        writeln!(f, "  dtype:  {:?},", K::elem_type_name())?;
        write!(f, "}}")
    }
}

/// Trait that list all operations that can be applied on all tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait BasicOps<B: Backend>: TensorKind<B> {
    type Elem: 'static;

    fn empty<const D: usize>(shape: Shape<D>, device: &B::Device) -> Self::Primitive<D>;
    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D>;
    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2>;
    fn index<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        indexes: [Range<usize>; D2],
    ) -> Self::Primitive<D1>;
    fn index_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        indexes: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1>;
    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> B::Device;
    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &B::Device,
    ) -> Self::Primitive<D>;
    fn into_data<const D: usize>(tensor: Self::Primitive<D>) -> Data<Self::Elem, D>;
    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: &B::Device,
    ) -> Self::Primitive<D>;
    fn repeat<const D: usize>(
        tensor: Self::Primitive<D>,
        dim: usize,
        times: usize,
    ) -> Self::Primitive<D>;
    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D>;
    fn equal<const D: usize>(
        lhs: Self::Primitive<D>,
        rhs: Self::Primitive<D>,
    ) -> Tensor<B, D, Bool>;
    fn equal_elem<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Elem) -> Tensor<B, D, Bool>;
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

    fn index<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        indexes: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::index(tensor, indexes)
    }

    fn index_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        indexes: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        B::index_assign(tensor, indexes, value)
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

    fn equal_elem<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Elem) -> Tensor<B, D, Bool> {
        Tensor::new(B::equal_elem(lhs, rhs))
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

    fn index<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        indexes: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::int_index(tensor, indexes)
    }

    fn index_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        indexes: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        B::int_index_assign(tensor, indexes, value)
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

    fn equal_elem<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Elem) -> Tensor<B, D, Bool> {
        Tensor::new(B::int_equal_elem(lhs, rhs))
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

    fn index<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        indexes: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::bool_index(tensor, indexes)
    }

    fn index_assign<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        indexes: [Range<usize>; D2],
        value: Self::Primitive<D1>,
    ) -> Self::Primitive<D1> {
        B::bool_index_assign(tensor, indexes, value)
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

    fn equal_elem<const D: usize>(lhs: Self::Primitive<D>, rhs: Self::Elem) -> Tensor<B, D, Bool> {
        Tensor::new(B::bool_equal_elem(lhs, rhs))
    }

    fn cat<const D: usize>(vectors: Vec<Self::Primitive<D>>, dim: usize) -> Self::Primitive<D> {
        B::bool_cat(vectors, dim)
    }
}
