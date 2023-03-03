use core::ops::Range;

use crate::{backend::Backend, ops::TensorOps, Bool, Data, Float, Int, Shape, TensorKind};

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
}

/// Trait that list all operations that can be applied on all tensors.
///
/// # Warnings
///
/// This is an internal trait, use the public API provided by [tensor struct](TensorNew).
pub trait BasicOps<B: Backend>: TensorKind<B> {
    type Elem;

    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D>;
    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2>;
    fn index<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        indexes: [Range<usize>; D2],
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
}

impl<B: Backend> BasicOps<B> for Float {
    type Elem = B::FloatElem;

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
}

impl<B: Backend> BasicOps<B> for Int {
    type Elem = <B::IntegerBackend as Backend>::FloatElem;

    fn shape<const D: usize>(tensor: &Self::Primitive<D>) -> Shape<D> {
        B::IntegerBackend::shape(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        shape: Shape<D2>,
    ) -> Self::Primitive<D2> {
        B::IntegerBackend::reshape(tensor, shape)
    }

    fn index<const D1: usize, const D2: usize>(
        tensor: Self::Primitive<D1>,
        indexes: [Range<usize>; D2],
    ) -> Self::Primitive<D1> {
        B::IntegerBackend::index(tensor, indexes)
    }

    fn device<const D: usize>(tensor: &Self::Primitive<D>) -> <B as Backend>::Device {
        B::IntegerBackend::device(tensor)
    }

    fn to_device<const D: usize>(
        tensor: Self::Primitive<D>,
        device: &<B as Backend>::Device,
    ) -> Self::Primitive<D> {
        B::IntegerBackend::to_device(tensor, device)
    }

    fn into_data<const D: usize>(tensor: Self::Primitive<D>) -> Data<Self::Elem, D> {
        B::IntegerBackend::into_data(tensor)
    }

    fn from_data<const D: usize>(
        data: Data<Self::Elem, D>,
        device: &B::Device,
    ) -> Self::Primitive<D> {
        B::IntegerBackend::from_data(data, device)
    }
}

impl<B: Backend> BasicOps<B> for Bool {
    type Elem = bool;

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
        B::from_data_bool(data, device)
    }
}
