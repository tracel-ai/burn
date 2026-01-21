use burn_backend::{Backend, Element, tensor::Device};
use burn_std::DType;

/// Options for tensor creation.
///
/// This struct allows specifying the `device` and/or data type (`dtype`) when creating a tensor.
#[derive(Debug, Clone)]
pub struct TensorCreationOptions<B: Backend> {
    /// Device where the tensor will be created.
    pub device: Device<B>,
    /// Optional data type.
    /// If `None`, the dtype will be inferred on creation from the backend's default dtype for the tensor kind.
    pub dtype: Option<DType>,
}

impl<B: Backend> Default for TensorCreationOptions<B> {
    /// Returns new options with the backend's default device.
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl<B: Backend> TensorCreationOptions<B> {
    /// Create new options with a specific device.
    ///
    /// Data type will be inferred on creation from the backend's default dtype for the tensor kind.
    pub fn new(device: Device<B>) -> Self {
        Self {
            device,
            dtype: None,
        }
    }

    /// Set the tensor creation data type.
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);

        self
    }

    /// Set the tensor creation device.
    pub fn with_device(mut self, device: Device<B>) -> Self {
        self.device = device;

        self
    }

    /// Create options with backend's default device and float dtype.
    pub fn float() -> Self {
        Self::default().with_dtype(<B::FloatElem as Element>::dtype())
    }

    /// Create options with backend's default device and int dtype.
    pub fn int() -> Self {
        Self::default().with_dtype(<B::IntElem as Element>::dtype())
    }

    /// Create options with backend's default device and bool dtype.
    pub fn bool() -> Self {
        Self::default().with_dtype(<B::BoolElem as Element>::dtype())
    }

    /// Returns the tensor data type, or a provided default if not set.
    ///
    /// This is useful for cases where [`TensorCreationOptions`] may not have an explicit `dtype`.
    pub fn dtype_or(&self, dtype: DType) -> DType {
        self.dtype.unwrap_or(dtype)
    }
}

impl<B: Backend> From<&Device<B>> for TensorCreationOptions<B> {
    /// Convenience conversion from a reference to a device.
    ///
    /// Example:
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::TensorCreationOptions;
    ///
    /// fn example<B: Backend>(device: B::Device) {
    ///     let options: TensorCreationOptions<B> = (&device).into();
    /// }
    /// ```
    fn from(device: &Device<B>) -> Self {
        TensorCreationOptions::new(device.clone())
    }
}

impl<B: Backend> From<(&Device<B>, DType)> for TensorCreationOptions<B> {
    /// Convenience conversion for a specified `(&device, dtype)` tuple.
    fn from(args: (&Device<B>, DType)) -> Self {
        TensorCreationOptions::new(args.0.clone()).with_dtype(args.1)
    }
}
