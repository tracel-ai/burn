pub use burn_std::{
    DeviceError, DeviceSettings, ExecutionError, backtrace::BackTrace, device::DeviceId,
};

use burn_backend::Backend;
#[allow(unused)]
use burn_dispatch::DispatchDeviceId;
use burn_dispatch::{Dispatch, DispatchDevice};
use burn_std::FloatDType;
use burn_std::IntDType;
use burn_std::QuantScheme;

use alloc::vec::Vec;
use core::mem::MaybeUninit;
use enumset::EnumSet;
use enumset::EnumSetType;

/// A high-level device handle for tensor operations.
///
/// [`Device`] provides a unified interface to interact with the underlying compute backend.
///
/// Autodiff support is a property of the device rather than a separate type parameter.
#[cfg_attr(
    feature = "autodiff",
    doc = "Wrap a device with [`.autodiff()`](Device::autodiff) to enable automatic differentiation with the device."
)]
#[cfg_attr(
    not(feature = "autodiff"),
    doc = "Enable the `autodiff` feature to add automatic differentiation support to devices."
)]
///
/// # Backend selection
///
/// Enable the desired backend via Cargo feature flags, then call the
/// corresponding factory method:
///
/// ```rust,ignore
/// // Requires the `cuda` feature.
/// let device = Device::cuda();
///
/// // Pick a specific CUDA hardware index.
/// let device = Device::cuda_at(0);
///
/// // Default device for whichever backend is enabled.
/// let device = Default::default();
/// ```
///
/// Available factory methods (each gated by its matching Cargo feature):
/// `Device::cpu`, `Device::cuda`, `Device::cuda_at`, `Device::rocm`,
/// `Device::wgpu`, `Device::flex`, `Device::ndarray`, `Device::libtorch`,
/// `Device::libtorch_cuda`.
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
pub struct Device {
    blob: DeviceBlob,
}

type DeviceInner = MaybeUninit<DispatchDevice>;

/// Storage for [`Device`]. Holds the raw bytes of a [`DispatchDevice`] while
/// preserving the alignment requirement via the zero-sized `_align` field, so
/// the backing memory can be safely reinterpreted as a `DispatchDevice`
/// reference. This obfuscates the dispatch type at the field level so it
/// doesn't appear in the public type signature.
#[repr(C)]
struct DeviceBlob {
    bytes: [u8; size_of::<DeviceInner>()],
    _align: [DeviceInner; 0],
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let inner: &mut DeviceInner =
                &mut *(self.blob.bytes.as_mut_ptr() as *mut DeviceInner);
            inner.assume_init_drop();
        }
    }
}

impl Clone for Device {
    fn clone(&self) -> Self {
        Self::from_dispatch(self.as_dispatch().clone())
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::from_dispatch(DispatchDevice::default())
    }
}

impl core::fmt::Debug for Device {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Device<{:?}>", self.as_dispatch())
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
        self.as_dispatch() == other.as_dispatch()
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

impl Device {
    /// Crate-internal helper for wrapping a [`DispatchDevice`] without going
    /// through the public factory methods. Used by burn-tensor's bridge ops
    /// when they receive a dispatch-level device from the dispatch layer.
    pub(crate) fn from_dispatch(dispatch: DispatchDevice) -> Self {
        let mut blob = DeviceBlob {
            bytes: [0u8; size_of::<DeviceInner>()],
            _align: [],
        };
        unsafe {
            let dst = blob.bytes.as_mut_ptr() as *mut DeviceInner;
            dst.write(MaybeUninit::new(dispatch));
        }
        Self { blob }
    }

    /// Crate-internal borrow of the underlying dispatch device.
    pub(crate) fn as_dispatch(&self) -> &DispatchDevice {
        unsafe {
            let inner: &DeviceInner = &*(self.blob.bytes.as_ptr() as *const DeviceInner);
            inner.assume_init_ref()
        }
    }

    /// Crate-internal owning extraction of the underlying dispatch device.
    pub(crate) fn into_dispatch(self) -> DispatchDevice {
        unsafe {
            let inner: DeviceInner =
                core::ptr::read(self.blob.bytes.as_ptr() as *const DeviceInner);
            core::mem::forget(self);
            inner.assume_init()
        }
    }
}

impl Device {
    /// Internal constructor used by the backend-specific factory methods below.
    /// Kept private so backend-specific device types never appear in the public
    /// API surface: callers use [`Device::cpu`], [`Device::cuda`] etc.
    fn new(device: impl Into<DispatchDevice>) -> Self {
        Self::from_dispatch(device.into())
    }

    /// Default CPU device backed by CubeCL's CPU backend.
    #[cfg(feature = "cpu")]
    pub fn cpu() -> Self {
        Self::new(burn_dispatch::devices::CpuDevice::default())
    }

    /// Default CUDA device (device index `0`).
    #[cfg(feature = "cuda")]
    pub fn cuda() -> Self {
        Self::new(burn_dispatch::devices::CudaDevice::default())
    }

    /// CUDA device at the given hardware index.
    #[cfg(feature = "cuda")]
    pub fn cuda_at(index: usize) -> Self {
        Self::new(burn_dispatch::devices::CudaDevice::new(index))
    }

    /// Default ROCm/HIP device.
    #[cfg(feature = "rocm")]
    pub fn rocm() -> Self {
        Self::new(burn_dispatch::devices::RocmDevice::default())
    }

    /// Flex backend device.
    #[cfg(feature = "flex")]
    pub fn flex() -> Self {
        Self::new(burn_dispatch::devices::FlexDevice)
    }

    /// Default NdArray (CPU) device.
    #[cfg(feature = "ndarray")]
    pub fn ndarray() -> Self {
        Self::new(burn_dispatch::devices::NdArrayDevice::default())
    }

    /// LibTorch CPU device.
    #[cfg(feature = "tch")]
    pub fn libtorch() -> Self {
        Self::new(burn_dispatch::devices::LibTorchDevice::Cpu)
    }

    /// LibTorch CUDA device at the given hardware index.
    #[cfg(feature = "tch")]
    pub fn libtorch_cuda(index: usize) -> Self {
        Self::new(burn_dispatch::devices::LibTorchDevice::Cuda(index))
    }

    /// WGPU device — lets `wgpu` auto-select an available adapter (Vulkan,
    /// Metal, or WebGPU depending on which features are enabled).
    #[cfg(any(
        feature = "wgpu",
        feature = "vulkan",
        feature = "metal",
        feature = "webgpu"
    ))]
    pub fn wgpu() -> Self {
        Self::new(burn_dispatch::devices::WgpuDevice::DefaultDevice)
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
    pub fn autodiff(self) -> Self {
        match self.into_dispatch() {
            DispatchDevice::Autodiff(_) => unimplemented!("Only first-order autodiff is supported"),
            other => Self::from_dispatch(DispatchDevice::autodiff(other)),
        }
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
    pub fn gradient_checkpointing(self) -> Self {
        match self.into_dispatch() {
            DispatchDevice::Autodiff(device) => {
                use burn_dispatch::CheckpointingStrategy;

                Self::from_dispatch(DispatchDevice::autodiff_checkpointed(
                    device.inner(),
                    CheckpointingStrategy::Balanced,
                ))
            }
            _ => panic!("Autodiff is not enabled on this device"),
        }
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
    pub fn inner(self) -> Self {
        if self.is_autodiff() {
            Self::from_dispatch(self.into_dispatch().inner())
        } else {
            self
        }
    }

    /// Synchronize the device, waiting for all pending operations to complete.
    ///
    /// # Errors
    ///
    /// Returns an [`ExecutionError`] if an operation failed to execute.
    pub fn sync(&self) -> Result<(), ExecutionError> {
        Dispatch::sync(self.as_dispatch())
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
        Dispatch::seed(self.as_dispatch(), seed)
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
        Dispatch::ad_enabled(self.as_dispatch())
    }

    /// Returns the default [quantization scheme](QuantScheme) for this device.
    pub fn default_quant_scheme(&self) -> QuantScheme {
        // TODO: maybe in device settings?
        Dispatch::default_quant_scheme(self.as_dispatch())
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
        Dispatch::memory_persistent_allocations(self.as_dispatch(), input, func)
    }

    /// Returns the [`DeviceSettings`] for this device.
    ///
    /// Settings include the default float and integer data types used when creating
    /// tensors on this device.
    ///
    /// See [`set_default_dtypes`](Device::set_default_dtypes) to configure them.
    pub fn settings(&self) -> DeviceSettings {
        burn_backend::get_device_settings::<Dispatch>(self.as_dispatch())
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
        burn_backend::set_default_dtypes::<Dispatch>(self.as_dispatch(), float_dtype, int_dtype)
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
