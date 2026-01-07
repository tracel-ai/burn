use burn_backend::{Backend, Element, tensor::Device};
use burn_std::DType;

/// Options for tensor creation.
///
/// This struct allows specifying the `device` and/or data type (`dtype`) when creating a tensor.
#[derive(Debug, Clone)]
pub struct TensorOptions<B: Backend> {
    /// Device where the tensor will be created.
    pub device: Device<B>,
    /// Optional data type.
    /// If `None`, the dtype will be inferred on creation from the backend's default dtype for the tensor kind.
    pub dtype: Option<DType>,
}

impl<B: Backend> Default for TensorOptions<B> {
    /// Returns [default float options](TensorOptions::float): default device + backend float dtype.
    fn default() -> Self {
        Self::float()
    }
}

impl<B: Backend> TensorOptions<B> {
    /// Create new options with a specific device and dtype.
    pub fn new(device: Device<B>, dtype: DType) -> Self {
        Self {
            device,
            dtype: Some(dtype),
        }
    }

    /// Create options with a specific device.
    ///
    /// Data type will be inferred on creation from the backend's default dtype for the tensor kind.
    pub fn device(device: Device<B>) -> Self {
        Self {
            device,
            dtype: None,
        }
    }

    /// Create options with a specific dtype.
    ///
    /// Device will default to the backend's default device.
    pub fn dtype(dtype: DType) -> Self {
        Self {
            dtype: Some(dtype),
            ..Default::default()
        }
    }

    /// Create options with backend's default device and float dtype.
    pub fn float() -> Self {
        Self::new(Default::default(), <B::FloatElem as Element>::dtype())
    }

    /// Create options with backend's default device and int dtype.
    pub fn int() -> Self {
        Self::new(Default::default(), <B::IntElem as Element>::dtype())
    }

    /// Create options with backend's default device and bool dtype.
    pub fn bool() -> Self {
        Self::new(Default::default(), <B::BoolElem as Element>::dtype())
    }

    /// Returns the tensor data type, or a provided default if not set.
    ///
    /// This is useful for cases where [`TensorOptions`] may not have an explicit `dtype`.
    pub fn dtype_or(&self, dtype: DType) -> DType {
        self.dtype.unwrap_or(dtype)
    }
}

impl<B: Backend> From<&Device<B>> for TensorOptions<B> {
    /// Convenience conversion from a reference to a device.
    ///
    /// Example:
    /// ```rust
    /// let options: TensorOptions<B> = (&my_device).into();
    /// ```
    fn from(device: &Device<B>) -> Self {
        TensorOptions::device(device.clone())
    }
}
