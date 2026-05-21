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
use enumset::EnumSet;
use enumset::EnumSetType;

use crate::macros::obfuscate_type;

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
/// let device = Device::cuda(1);
///
/// // WGPU with explicit selector (requires `wgpu`/`vulkan`/`metal`/`webgpu`).
/// let device = Device::wgpu(DeviceKind::DiscreteGpu(0));
///
/// // Default device for whichever backend is enabled.
/// let device = Default::default();
/// ```
///
/// Available factory methods (each gated by its matching Cargo feature):
/// `Device::cpu`, `Device::cuda` / `Device::rocm` / `Device::libtorch_cuda`
/// (take an integer index or a [`DeviceIndex`]), `Device::wgpu` /
/// `Device::vulkan` / `Device::metal` / `Device::webgpu` (take a
/// [`DeviceKind`]), `Device::flex`, `Device::ndarray`, `Device::libtorch`,
/// `Device::libtorch_mps`, `Device::libtorch_vulkan`.
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
    blob: device_blob::Blob,
}

// Aligned, type-erased storage for `DispatchDevice`. See `crate::macros` for
// why this indirection exists (it keeps the dispatch type tree out of
// downstream MIR).
obfuscate_type!(device_blob, DispatchDevice, Send, Sync);

impl Clone for Device {
    fn clone(&self) -> Self {
        Self::new(self.as_dispatch().clone())
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::new(DispatchDevice::default())
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
    /// Wrap a backend-specific device in a unified [`Device`].
    ///
    /// Used by:
    /// - the backend-specific factory methods below (`Device::cuda`, etc.)
    ///   — these are the recommended entry points for downstream code;
    /// - burn-tensor's bridge ops, which already hold a [`DispatchDevice`]
    ///   and just need to wrap it;
    /// - direct callers (tests, type-erased helpers) that have a concrete
    ///   backend device type at hand.
    ///
    /// Anything convertible into [`DispatchDevice`] is accepted, including
    /// `DispatchDevice` itself.
    pub fn new(device: impl Into<DispatchDevice>) -> Self {
        Self {
            blob: device_blob::Blob::new(device.into()),
        }
    }

    /// Crate-internal borrow of the underlying dispatch device.
    pub(crate) fn as_dispatch(&self) -> &DispatchDevice {
        self.blob.as_ref()
    }

    /// Crate-internal owning extraction of the underlying dispatch device.
    pub(crate) fn into_dispatch(self) -> DispatchDevice {
        self.blob.into_inner()
    }
}

impl<D: Into<DispatchDevice>> From<D> for Device {
    fn from(device: D) -> Self {
        Self::new(device)
    }
}

/// Selector for the hardware index of a backend whose devices are simply
/// indexed (e.g. CUDA, ROCm).
///
/// Backend factory methods that take an index (`Device::cuda`, `Device::rocm`,
/// `Device::libtorch_cuda`) accept `impl Into<DeviceIndex>`, so the common
/// shorthand is to pass a plain integer literal:
///
/// ```rust,ignore
/// Device::cuda(0);                    // hardware index 0
/// Device::cuda(DeviceIndex::Default); // backend-chosen default
/// ```
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub enum DeviceIndex {
    /// Target a specific hardware device by its index.
    Specified(usize),
    /// Let the backend pick its default device (typically index `0`).
    #[default]
    Default,
}

impl DeviceIndex {
    /// Construct a [`DeviceIndex::Specified`] from anything convertible into
    /// a `usize`.
    pub fn new(index: impl Into<usize>) -> Self {
        Self::Specified(index.into())
    }

    /// Resolve to a concrete hardware index, defaulting to `0` for
    /// [`DeviceIndex::Default`]. Backend factory methods are each gated by a
    /// Cargo feature, so this looks dead when none of them are enabled.
    #[allow(dead_code)]
    fn resolve(self) -> usize {
        match self {
            DeviceIndex::Specified(i) => i,
            DeviceIndex::Default => 0,
        }
    }
}

impl From<usize> for DeviceIndex {
    fn from(i: usize) -> Self {
        Self::Specified(i)
    }
}

impl From<u32> for DeviceIndex {
    fn from(i: u32) -> Self {
        Self::Specified(i as usize)
    }
}

impl From<u64> for DeviceIndex {
    fn from(i: u64) -> Self {
        Self::Specified(i as usize)
    }
}

impl From<i32> for DeviceIndex {
    fn from(i: i32) -> Self {
        Self::Specified(usize::try_from(i).expect("device index must be non-negative"))
    }
}

impl From<i64> for DeviceIndex {
    fn from(i: i64) -> Self {
        Self::Specified(usize::try_from(i).expect("device index must be non-negative"))
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

    /// The best available device found with the current graphics API.
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
    /// This can be initialized with `init_device` from the wgpu runtime.
    Existing(u32),
}

impl Device {
    /// Default CPU device backed by CubeCL's CPU backend.
    #[cfg(feature = "cpu")]
    pub fn cpu() -> Self {
        Self::new(burn_dispatch::devices::CpuDevice::default())
    }

    /// CUDA device at the given hardware index.
    ///
    /// Accepts a plain integer (e.g. `Device::cuda(0)`) or a
    /// [`DeviceIndex`] — use [`DeviceIndex::Default`] to let the backend
    /// pick.
    #[cfg(feature = "cuda")]
    pub fn cuda(index: impl Into<DeviceIndex>) -> Self {
        Self::new(burn_dispatch::devices::CudaDevice::new(
            index.into().resolve(),
        ))
    }

    /// ROCm/HIP device at the given hardware index.
    ///
    /// Same selector semantics as [`Device::cuda`].
    #[cfg(feature = "rocm")]
    pub fn rocm(index: impl Into<DeviceIndex>) -> Self {
        Self::new(burn_dispatch::devices::RocmDevice::new(
            index.into().resolve(),
        ))
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
    ///
    /// Same selector semantics as [`Device::cuda`].
    #[cfg(feature = "tch")]
    pub fn libtorch_cuda(index: impl Into<DeviceIndex>) -> Self {
        Self::new(burn_dispatch::devices::LibTorchDevice::Cuda(
            index.into().resolve(),
        ))
    }

    /// LibTorch Metal Performance Shaders (MPS) device.
    #[cfg(feature = "tch")]
    pub fn libtorch_mps() -> Self {
        Self::new(burn_dispatch::devices::LibTorchDevice::Mps)
    }

    /// LibTorch Vulkan device.
    #[cfg(feature = "tch")]
    pub fn libtorch_vulkan() -> Self {
        Self::new(burn_dispatch::devices::LibTorchDevice::Vulkan)
    }

    /// WGPU device, selected via [`DeviceKind`].
    ///
    /// The actual wgpu adapter (Vulkan / Metal / WebGPU) is picked by the
    /// enabled Cargo features and, for [`DeviceKind::DefaultDevice`], by
    /// `wgpu`'s adapter-selection heuristics (high-power GPU preferred, or
    /// whatever `CUBECL_WGPU_DEFAULT_DEVICE` overrides it to).
    ///
    /// `Device::vulkan` / `Device::metal` / `Device::webgpu` are
    /// equivalent calls (they exist only to make the intended backend
    /// explicit at the call site — the underlying adapter is still picked by
    /// enabled Cargo features).
    #[cfg(any(
        feature = "wgpu",
        feature = "vulkan",
        feature = "metal",
        feature = "webgpu"
    ))]
    pub fn wgpu(device_kind: DeviceKind) -> Self {
        Self::new(wgpu_device(device_kind))
    }

    /// Vulkan-backed WGPU device, selected via [`DeviceKind`].
    ///
    /// Equivalent to [`Device::wgpu`]; provided so the caller's choice of
    /// graphics API reads clearly at the call site. The actual adapter is
    /// still determined by enabled Cargo features.
    #[cfg(feature = "vulkan")]
    pub fn vulkan(device_kind: DeviceKind) -> Self {
        Self::new(wgpu_device(device_kind))
    }

    /// Metal-backed WGPU device, selected via [`DeviceKind`].
    ///
    /// See [`Device::vulkan`] — same shape, different feature gate.
    #[cfg(feature = "metal")]
    pub fn metal(device_kind: DeviceKind) -> Self {
        Self::new(wgpu_device(device_kind))
    }

    /// WebGPU-backed device, selected via [`DeviceKind`].
    ///
    /// See [`Device::vulkan`] — same shape, different feature gate.
    #[cfg(feature = "webgpu")]
    pub fn webgpu(device_kind: DeviceKind) -> Self {
        Self::new(wgpu_device(device_kind))
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
            other => Self::new(DispatchDevice::autodiff(other)),
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

                Self::new(DispatchDevice::autodiff_checkpointed(
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
            Self::new(self.into_dispatch().inner())
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

/// Map our backend-agnostic [`DeviceKind`] onto cubecl's `WgpuDevice` enum.
///
/// Shared by [`Device::wgpu`], [`Device::vulkan`], [`Device::metal`], and
/// [`Device::webgpu`], which differ only in which Cargo feature gates them.
#[cfg(any(
    feature = "wgpu",
    feature = "vulkan",
    feature = "metal",
    feature = "webgpu"
))]
fn wgpu_device(device_kind: DeviceKind) -> burn_dispatch::devices::WgpuDevice {
    use burn_dispatch::devices::WgpuDevice;
    match device_kind {
        DeviceKind::DiscreteGpu(i) => WgpuDevice::DiscreteGpu(i),
        DeviceKind::IntegratedGpu(i) => WgpuDevice::IntegratedGpu(i),
        DeviceKind::VirtualGpu(i) => WgpuDevice::VirtualGpu(i),
        DeviceKind::Cpu => WgpuDevice::Cpu,
        DeviceKind::DefaultDevice => WgpuDevice::DefaultDevice,
        DeviceKind::Existing(id) => WgpuDevice::Existing(id),
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
