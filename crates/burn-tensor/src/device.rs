pub use burn_backend::{DeviceError, DeviceId, DeviceSettings};
pub use burn_dispatch::devices::*;

use burn_backend::Backend;
#[allow(unused)]
use burn_dispatch::DispatchDeviceId;
use burn_dispatch::{Dispatch, DispatchDevice};
use burn_std::FloatDType;
use burn_std::IntDType;
use burn_std::QuantScheme;

use alloc::vec::Vec;
use enumset::EnumSet;
use enumset::EnumSetType;

/// A high-level device handle for tensor operations.
///
/// [`Device`] provides a unified interface to interact with the underlying compute backend.
///
/// Autodiff support is a property of the device rather than a separate type parameter.
/// Wrap a device with [`.autodiff()`](Device::autodiff) to enable automatic  differentiation
/// with the device.
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
/// let device = Default::default();
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
#[derive(Clone, Default)]
pub struct Device {
    pub(crate) dispatch: DispatchDevice,
}

impl core::fmt::Debug for Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Device<{:?}>", self.dispatch)
    }
}

// Manually implement both `eq` and `ne` to add documentation on equality.
#[allow(clippy::partialeq_ne_impl)]
impl PartialEq for Device {
    /// Compares devices based on hardware identity.
    ///
    /// Returns `true` if both devices represent the same compute resource.
    /// Note that this comparison ignores autodiff and checkpointing settings.
    /// To check if two devices have identical capabilities, check [`Device::is_autodiff`].
    fn eq(&self, other: &Self) -> bool {
        self.dispatch == other.dispatch
    }

    /// Compares devices based on hardware identity.
    ///
    /// Returns `false` if both devices represent the same compute resource,
    /// even if one has autodiff enabled and the other does not.
    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl Eq for Device {}

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

    /// Enables gradient checkpointing on the autodiff device.
    ///
    /// Gradient checkpointing recomputes activations during backpropagation for operations
    /// marked as memory-bound, while compute-bound operations still cache their
    /// output. This reduces peak memory usage at the cost of additional computation
    /// for memory-bound ops.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = Device::default().autodiff().gradient_checkpointing();
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if autodiff is not enabled on this device.
    #[cfg(feature = "autodiff")]
    pub fn gradient_checkpointing(mut self) -> Self {
        match self.dispatch {
            DispatchDevice::Autodiff(device) => {
                use burn_dispatch::CheckpointingStrategy;

                self.dispatch = DispatchDevice::autodiff_checkpointed(
                    device.inner(),
                    CheckpointingStrategy::Balanced,
                )
            }
            _ => panic!("Autodiff is not enabled on this device"),
        }

        self
    }

    /// Returns the underlying device, removing the autodiff capability if present.
    ///
    /// If autodiff is not enabled, this method returns the device as-is.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = Device::default().autodiff();
    /// let inner_device = device.inner();
    ///
    /// assert!(!inner_device.is_autodiff());
    /// ```
    pub fn inner(mut self) -> Self {
        if self.is_autodiff() {
            self.dispatch = self.dispatch.inner();
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
    /// let device = Default::default();
    /// device.seed(42);
    /// let t = Tensor::<1>::random([8], Distribution::Default, &device);
    /// ```
    pub fn seed(&self, seed: u64) {
        Dispatch::seed(&self.dispatch, seed)
    }

    /// Returns `true` if autodiff (gradient tracking) is enabled on this device.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let device = Default::default();
    /// assert!(!device.is_autodiff());
    ///
    /// let ad_device = device.autodiff();
    /// assert!(ad_device.is_autodiff());
    /// ```
    pub fn is_autodiff(&self) -> bool {
        Dispatch::ad_enabled(&self.dispatch)
    }

    /// Returns the default [quantization scheme](QuantScheme) for this device.
    pub fn default_quant_scheme(&self) -> QuantScheme {
        // TODO: maybe in device settings?
        Dispatch::default_quant_scheme(&self.dispatch)
    }

    /// Sets the current allocation mode to persistent.
    pub fn memory_persistent_allocations<
        Output: Send,
        Input: Send,
        Func: Fn(Input) -> Output + Send,
    >(
        &self,
        input: Input,
        func: Func,
    ) -> Output {
        Dispatch::memory_persistent_allocations(&self.dispatch, input, func)
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
    /// let device = Default::default();
    ///
    /// device.set_default_dtypes(DType::F16, DType::I32)?;
    ///
    /// // Float tensors will now use F16
    /// let floats = Tensor::<2>::zeros([2, 3], &device);
    /// // Int tensors will now use I32
    /// let ints = Tensor::<2, Int>::zeros([2, 3], &device);
    /// ```
    pub fn set_default_dtypes(
        &mut self,
        float_dtype: impl Into<FloatDType>,
        int_dtype: impl Into<IntDType>,
    ) -> Result<(), DeviceError> {
        burn_backend::set_default_dtypes::<Dispatch>(&self.dispatch, float_dtype, int_dtype)
    }

    /// Retrieves all available [`Device`]s that match the given [`DeviceType`] filter.
    pub fn enumerate(filter: impl Into<EnumSet<DeviceType>>) -> Vec<Device> {
        #[allow(unused)]
        let mut devices = Vec::new();

        #[allow(clippy::never_loop)] // at least one backend is expected to be enabled.
        for device_type in filter.into() {
            #[allow(unused)]
            let type_id = match device_type {
                #[cfg(feature = "cpu")]
                DeviceType::Cpu => DispatchDeviceId::Cpu,
                #[cfg(feature = "cuda")]
                DeviceType::Cuda => DispatchDeviceId::Cuda,
                #[cfg(feature = "rocm")]
                DeviceType::Rocm => DispatchDeviceId::Rocm,
                #[cfg(any(
                    feature = "wgpu",
                    feature = "metal",
                    feature = "vulkan",
                    feature = "webgpu"
                ))]
                DeviceType::Wgpu => DispatchDeviceId::Wgpu,
                #[cfg(feature = "flex")]
                DeviceType::Flex => DispatchDeviceId::Flex,
                #[cfg(feature = "ndarray")]
                DeviceType::NdArray => DispatchDeviceId::NdArray,
                #[cfg(feature = "tch")]
                DeviceType::LibTorch => DispatchDeviceId::LibTorch,
            };

            #[allow(unreachable_code)] // need to have one backend enabled, so it is reachable
            for device in Dispatch::enumerate(type_id) {
                devices.push(Device::new(device))
            }
        }

        devices
    }
}

// TODO: this is essentially per-backend filter, we could have higher level filters e.g. Cpu (CpuDevice, Ndarray, Flex, LibTorchDevice::Cpu)

/// Represents the devices that can be used.
///
/// `DeviceType` is used to filter the available device types for [`Device::enumerate`].
#[allow(missing_docs)]
#[derive(Debug, EnumSetType)]
pub enum DeviceType {
    #[cfg(feature = "cpu")]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "rocm")]
    Rocm,
    #[cfg(any(
        feature = "wgpu",
        feature = "metal",
        feature = "vulkan",
        feature = "webgpu"
    ))]
    Wgpu,
    #[cfg(feature = "flex")]
    Flex,
    #[cfg(feature = "ndarray")]
    NdArray,
    #[cfg(feature = "tch")]
    LibTorch,
}
