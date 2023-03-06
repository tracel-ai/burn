use alloc::vec::Vec;
use core::ops::Range;

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

/// Trait that list all operations that can be applied on all tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](Tensor).
pub trait BasicOps<B: Backend>: TensorKind<B> {
    type Elem;

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
