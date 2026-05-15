use burn_std::DType;

use crate::{Device, bridge::BasicOps};

/// Options for tensor creation.
///
/// This struct allows specifying the `device` and overriding the data type when creating a tensor.
/// When the `dtype` is not specified, the [device's default settings](crate::DeviceSettings) is used.
#[derive(Debug, Clone)]
pub struct TensorCreationOptions {
    /// Device where the tensor will be created.
    pub device: Device,
    /// Optional data type.
    /// If `None`, the dtype will be inferred on creation from the [device settings](crate::DeviceSettings).
    pub dtype: Option<DType>,
}

impl Default for TensorCreationOptions {
    /// Returns new options with the backend's default device.
    fn default() -> Self {
        Self::new(Default::default())
    }
}

impl TensorCreationOptions {
    /// Create new options with a specific device.
    ///
    /// Data type will follow the [device settings](crate::DeviceSettings) on tensor creation.
    pub fn new(device: Device) -> Self {
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
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;

        self
    }

    /// Returns the tensor data type, or a provided default if not set.
    ///
    /// This is useful for cases where [`TensorCreationOptions`] may not have an explicit `dtype`.
    pub fn dtype_or(&self, dtype: DType) -> DType {
        self.dtype.unwrap_or(dtype)
    }

    /// Returns the tensor data type, or the default from the [device settings](crate::DeviceSettings).
    pub(crate) fn resolve_dtype<K: BasicOps>(&self) -> DType {
        let kind_name = K::name();
        // TODO: tensor kind enum?
        self.dtype.unwrap_or_else(|| {
            let settings = self.device.settings();
            if kind_name == "Float" {
                settings.float_dtype.into()
            } else if kind_name == "Int" {
                settings.int_dtype.into()
            } else {
                settings.bool_dtype.into()
            }
        })
    }
}

impl From<&Device> for TensorCreationOptions {
    /// Convenience conversion from a reference to a device.
    ///
    /// Example:
    /// ```rust
    /// use burn_tensor::TensorCreationOptions;
    /// use burn_tensor::Device;
    ///
    /// fn example(device: Device) {
    ///     let options: TensorCreationOptions = (&device).into();
    /// }
    /// ```
    fn from(device: &Device) -> Self {
        TensorCreationOptions::new(device.clone())
    }
}

impl From<(&crate::Device, DType)> for TensorCreationOptions {
    /// Convenience conversion for a specified `(&device, dtype)` tuple.
    fn from(args: (&crate::Device, DType)) -> Self {
        TensorCreationOptions::new(args.0.clone()).with_dtype(args.1)
    }
}
