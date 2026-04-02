pub use burn_backend::DeviceError;
pub use burn_backend::DeviceSettings;

use burn_backend::Backend;
use burn_dispatch::{Dispatch, DispatchDevice};
use burn_std::FloatDType;
use burn_std::IntDType;
use burn_std::QuantScheme;

// TODO: docs + book

/// A high-level device handle for tensor operations.
///
/// [`Device`] provides a unified interface to interact with the underlying compute backend
///
/// Autodiff support is a property of the device rather than a separate type parameter.
/// Wrap a device with [`.autodiff()`](Device::autodiff) or
/// [`.autodiff_checkpointed()`](Device::autodiff_checkpointed) to enable gradient automatic
/// differentiation with the device.
///
/// # Backend selection
///
/// Enable the desired backend via Cargo feature flags, then construct the corresponding
/// backend device. You can use [`Device::new`], the [`From`]/[`Into`] trait,
/// or [`Device::default()`] for the active backend's default device:
///
/// ```rust,ignore
/// let device = Device::new(CudaDevice::default());
///
/// let device: Device = CudaDevice::default().into();
///
/// // Default device for whichever backend is enabled
/// let device = Device::default();
/// ```
///
/// # Autodiff
///
/// Requires `autodiff` feature.
///
/// Gradient computation is opt-in for a device:
///
/// ```rust,ignore
/// let device = Device::default().autodiff();
///
/// // Tensors created on this device will track gradients
/// let x = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);
/// ```
#[derive(Clone, Default, PartialEq)]
pub struct Device {
    pub(crate) dispatch: DispatchDevice,
}

// TODO: Device::default() priority for DispatchDevice should be detailed

impl core::fmt::Debug for Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Device<{:?}>", self.dispatch)
    }
}

impl<D: Into<DispatchDevice>> From<D> for Device {
    fn from(device: D) -> Self {
        Self::new(device)
    }
}

impl Device {
    /// Creates a new [`Device`] from a supported backend device.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = Device::new(CudaDevice::default());
    /// ```
    pub fn new(device: impl Into<DispatchDevice>) -> Self {
        Self {
            dispatch: device.into(),
        }
    }

    /// Enables autodiff on this device.
    ///
    /// Autodiff is a property of the device: tensors created on the returned device
    /// will participate in the autodiff graph.
    ///
    /// Only first-order autodiff is supported. Calling this method on a device that
    /// already has autodiff enabled will panic.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = Device::default().autodiff();
    /// let x = Tensor::<1>::from_floats([1.0, 2.0, 3.0], &device);
    /// // x.backward() is now available
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if autodiff is already enabled on this device.
    #[cfg(feature = "autodiff")]
    pub fn autodiff(mut self) -> Self {
        match self.dispatch {
            DispatchDevice::Autodiff(_) => unimplemented!("Only first-order autodiff is supported"),
            other => self.dispatch = DispatchDevice::autodiff(other),
        }

        self
    }

    /// Enables autodiff with gradient checkpointing on this device.
    ///
    /// Gradient checkpointing recomputes activations during backpropagation for operations
    /// marked as memory-bound, while compute-bound operations still cache their
    /// output. This reduces peak memory usage at the cost of additional computation
    /// for memory-bound ops.
    ///
    /// Like [`autodiff`](Device::autodiff), autodiff is a property of the device:
    /// all tensors created on the returned device participate in the autodiff graph.
    ///
    /// Only first-order autodiff is supported. Calling this method on a device that
    /// already has autodiff enabled will panic.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = Device::default().autodiff_checkpointed();
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if autodiff is already enabled on this device.
    #[cfg(feature = "autodiff")]
    pub fn autodiff_checkpointed(mut self) -> Self {
        match self.dispatch {
            DispatchDevice::Autodiff(_) => unimplemented!("Only first-order autodiff is supported"),
            other => {
                self.dispatch = DispatchDevice::autodiff_checkpointed(other, Default::default())
            }
        }

        self
    }
    /// Synchronize the device, waiting for all pending operations to complete.
    ///
    /// # Errors
    ///
    /// Returns an [`ExecutionError`](burn_backend::ExecutionError) if an operation failed to execute.
    pub fn sync(&self) -> Result<(), burn_backend::ExecutionError> {
        Dispatch::sync(&self.dispatch)
    }

    /// Seeds the random number generator for this device.
    ///
    /// Seeding before tensor operations that involve randomness (e.g. [`Tensor::random`](crate::Tensor::random))
    /// makes those operations reproducible in a single-threaded program.
    ///
    /// # Note
    ///
    /// Depending on the backend, the seed may be applied globally rather than scoped
    /// to this specific device. It is guaranteed that at least this device will be seeded.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = Device::default();
    /// device.seed(42);
    /// let t = Tensor::<1>::random([8], Distribution::Default, &device);
    /// ```
    pub fn seed(&self, seed: u64) {
        Dispatch::seed(&self.dispatch, seed)
    }

    /// Returns the default [quantization scheme](QuantScheme) for this device.
    pub fn default_quant_scheme(&self) -> QuantScheme {
        // TODO: maybe in device settings?
        Dispatch::default_quant_scheme(&self.dispatch)
    }

    /// Returns the [`DeviceSettings`] for this device.
    ///
    /// Settings include the default float and integer data types used when creating
    /// tensors on this device.
    ///
    /// See [`set_default_dtypes`](Device::set_default_dtypes) to configure them.
    pub fn settings(&self) -> DeviceSettings {
        burn_backend::get_device_settings::<Dispatch>(&self.dispatch)
    }

    /// Sets the default float and integer data types for tensors created on this device.
    ///
    /// This configures the dtype used when no explicit type is specified at tensor
    /// creation time.
    ///
    /// Settings can only be initialized once per device, and must happen before any
    /// tensor is created on the device. The first tensor operation will lock the device
    /// to its defaults, causing subsequent initializations attempt to return
    /// [`DeviceError::AlreadyInitialized`].
    ///
    /// # Errors
    ///
    /// Returns [`DeviceError::AlreadyInitialized`] if settings have already been set
    /// for this device (either by a prior call or because a tensor operation has
    /// already occurred).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = Device::default();
    ///
    /// device.set_default_dtypes(DType::F16, DType::I32)?;
    ///
    /// // Float tensors will now use F16
    /// let floats = Tensor::<2>::zeros([2, 3], &device);
    /// // Int tensors will now use I32
    /// let ints = Tensor::<2, Int>::zeros([2, 3], &device);
    /// ```
    pub fn set_default_dtypes(
        &self,
        float_dtype: impl Into<FloatDType>,
        int_dtype: impl Into<IntDType>,
    ) -> Result<(), DeviceError> {
        burn_backend::set_default_dtypes::<Dispatch>(&self.dispatch, float_dtype, int_dtype)
    }
}
