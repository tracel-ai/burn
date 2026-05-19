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
/// // Default CUDA device (requires the `cuda` feature).
/// let device = Device::cuda(DeviceIndex::Default);
///
/// // CUDA device at hardware index 1.
/// let device = Device::cuda(DeviceIndex::new(1));
///
/// // WGPU with explicit selector (requires `wgpu`/`vulkan`/`metal`/`webgpu`).
/// let device = Device::wgpu(DeviceKind::DiscreteGpu(0));
///
/// // Default device for whichever backend is enabled.
/// let device = Default::default();
/// ```
///
/// Available factory methods (each gated by its matching Cargo feature):
/// `Device::cpu`, `Device::cuda` (takes a [`DeviceIndex`]), `Device::rocm`
/// (takes a [`DeviceIndex`]), `Device::wgpu` (takes a [`DeviceKind`]),
/// `Device::vulkan`, `Device::metal`, `Device::webgpu`, `Device::flex`,
/// `Device::ndarray`, `Device::libtorch`, `Device::libtorch_cuda`.
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

/// Storage for [`Device`]. Holds the raw bytes of a [`DispatchDevice`].
///
/// Intentionally has no type-level alignment marker (e.g. `[DeviceInner; 0]`),
/// since that would re-introduce a `burn_dispatch` dependency in the type
/// itself and undermine the compile-time goal of this obfuscation. Alignment
/// must therefore be handled at access sites
/// (TODO: bring back proper alignment without leaking the type).
struct DeviceBlob {
    bytes: [u8; size_of::<DeviceInner>()],
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let inner: &mut DeviceInner = &mut *(self.blob.bytes.as_mut_ptr() as *mut DeviceInner);
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

/// Selector for the hardware index of a backend whose devices are simply
/// indexed (e.g. CUDA, ROCm).
///
/// Use [`DeviceIndex::Default`] to let the backend pick its default device, or
/// [`DeviceIndex::Specified`] / [`DeviceIndex::new`] to target a particular
/// hardware index.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub enum DeviceIndex {
    /// Target a specific device by its hardware index.
    Specified(usize),
    /// Use whatever device the backend considers its default
    /// (typically index `0`).
    #[default]
    Default,
}

impl DeviceIndex {
    /// Convenience constructor for [`DeviceIndex::Specified`].
    pub fn new(index: impl Into<usize>) -> Self {
        Self::Specified(index.into())
    }

    /// Resolve to a concrete hardware index, defaulting to `0` for
    /// [`DeviceIndex::Default`]. Used by backend factory methods that need a
    /// plain `usize`.
    fn resolve(self) -> usize {
        match self {
            DeviceIndex::Specified(i) => i,
            DeviceIndex::Default => 0,
        }
    }
}

/// Selector for the more flexible backends whose device handle is a tagged
/// enum (e.g. WGPU, which can target a discrete/integrated/virtual GPU, a CPU
/// adapter, an externally-created wgpu setup, or just "best available").
///
/// The variants mirror `WgpuDevice` from cubecl so the mapping is direct, but
/// it is kept as a burn-owned enum so callers don't have to depend on cubecl.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Default)]
pub enum DeviceKind {
    /// Discrete GPU with the given index. The index is the index of the discrete GPU in the list
    /// of all discrete GPUs found on the system.
    DiscreteGpu(usize),

    /// Integrated GPU with the given index. The index is the index of the integrated GPU in the
    /// list of all integrated GPUs found on the system.
    IntegratedGpu(usize),

    /// Virtual GPU with the given index. The index is the index of the virtual GPU in the list of
    /// all virtual GPUs found on the system.
    VirtualGpu(usize),

    /// CPU.
    Cpu,

    /// The best available device found with the current [graphics API](crate::GraphicsApi).
    ///
    /// This will prioritize GPUs wgpu recognizes as "high power". Additionally, you can override this using
    /// the `CUBECL_WGPU_DEFAULT_DEVICE` environment variable. This variable is spelled as if i was a `WgpuDevice`,
    /// so for example `CUBECL_WGPU_DEFAULT_DEVICE=IntegratedGpu(1)` or `CUBECL_WGPU_DEFAULT_DEVICE=Cpu`
    #[default]
    DefaultDevice,

    /// Use an externally created, existing, wgpu setup. This is helpful when using `CubeCL` in conjunction
    /// with some existing wgpu setup (eg. egui or bevy), as resources can be transferred in & out of `CubeCL`.
    ///
    /// # Notes
    ///
    /// This can be initialized with [`init_device`](crate::runtime::init_device).
    Existing(u32),
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

    /// CUDA device.
    ///
    /// Pass [`DeviceIndex::Default`] (or `DeviceIndex::Default` via
    /// `Default::default()`) for the default CUDA device, or
    /// [`DeviceIndex::Specified(i)`](DeviceIndex::Specified) / [`DeviceIndex::new(i)`](DeviceIndex::new)
    /// to target hardware index `i`.
    #[cfg(feature = "cuda")]
    pub fn cuda(index: DeviceIndex) -> Self {
        // `CudaDevice` is a plain `struct { index: usize }` with `Default`
        // returning index `0`, so we just resolve our enum to a usize.
        Self::new(burn_dispatch::devices::CudaDevice::new(index.resolve()))
    }

    /// ROCm/HIP device.
    ///
    /// Same selector semantics as [`Device::cuda`]: pass [`DeviceIndex::Default`]
    /// for the default device, or a [`DeviceIndex::Specified`] / [`DeviceIndex::new`]
    /// hardware index.
    #[cfg(feature = "rocm")]
    pub fn rocm(index: DeviceIndex) -> Self {
        // `RocmDevice` (cubecl's `AmdDevice`) is also a plain
        // `struct { index: usize }`.
        Self::new(burn_dispatch::devices::RocmDevice::new(index.resolve()))
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

    /// WGPU device, selected via [`DeviceKind`].
    ///
    /// The actual wgpu adapter (Vulkan / Metal / WebGPU) is picked by the
    /// enabled Cargo features and, for [`DeviceKind::DefaultDevice`], by
    /// `wgpu`'s adapter-selection heuristics (high-power GPU preferred, or
    /// whatever `CUBECL_WGPU_DEFAULT_DEVICE` overrides it to).
    ///
    /// For the typical "give me a reasonable device" case, pass
    /// [`DeviceKind::DefaultDevice`] (or use [`DeviceKind::default()`]) — the
    /// dedicated [`Device::vulkan`] / [`Device::metal`] / [`Device::webgpu`]
    /// helpers are thin wrappers around exactly that.
    #[cfg(any(
        feature = "wgpu",
        feature = "vulkan",
        feature = "metal",
        feature = "webgpu"
    ))]
    pub fn wgpu(device_kind: DeviceKind) -> Self {
        // Direct one-to-one mapping onto cubecl's `WgpuDevice` enum.
        use burn_dispatch::devices::WgpuDevice;
        let device = match device_kind {
            DeviceKind::DiscreteGpu(i) => WgpuDevice::DiscreteGpu(i),
            DeviceKind::IntegratedGpu(i) => WgpuDevice::IntegratedGpu(i),
            DeviceKind::VirtualGpu(i) => WgpuDevice::VirtualGpu(i),
            DeviceKind::Cpu => WgpuDevice::Cpu,
            DeviceKind::DefaultDevice => WgpuDevice::DefaultDevice,
            DeviceKind::Existing(id) => WgpuDevice::Existing(id),
        };
        Self::new(device)
    }

    /// Vulkan-backed WGPU device, selecting [`DeviceKind::DefaultDevice`].
    ///
    /// Use [`Device::wgpu`] directly for finer control (e.g. picking a
    /// specific discrete or integrated GPU). The chosen graphics API
    /// (Vulkan / Metal / WebGPU) is still ultimately determined by enabled
    /// Cargo features.
    #[cfg(feature = "vulkan")]
    pub fn vulkan() -> Self {
        Self::wgpu(DeviceKind::DefaultDevice)
    }

    /// Metal-backed WGPU device, selecting [`DeviceKind::DefaultDevice`].
    ///
    /// See [`Device::vulkan`] for caveats. Use [`Device::wgpu`] directly for
    /// finer control.
    #[cfg(feature = "metal")]
    pub fn metal() -> Self {
        Self::wgpu(DeviceKind::DefaultDevice)
    }

    /// WebGPU-backed device, selecting [`DeviceKind::DefaultDevice`].
    ///
    /// See [`Device::vulkan`] for caveats. Use [`Device::wgpu`] directly for
    /// finer control.
    #[cfg(feature = "webgpu")]
    pub fn webgpu() -> Self {
        Self::wgpu(DeviceKind::DefaultDevice)
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
