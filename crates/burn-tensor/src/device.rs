pub use burn_std::{
    DeviceError, DeviceSettings, ExecutionError, backtrace::BackTrace, device::DeviceId,
};

use burn_backend::{Backend, DeviceOps};
#[allow(unused)]
use burn_dispatch::DispatchDeviceId;
use burn_dispatch::{Dispatch, DispatchDevice};
use burn_std::{BoolDType, ComplexDType, DType, FloatDType, IntDType, TensorData};

#[cfg(feature = "remote")]
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

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
    blob: device_opaque::Opaque,
}

// Aligned, type-erased storage for `DispatchDevice`. See `crate::macros` for
// why this indirection exists (it keeps the dispatch type tree out of
// downstream MIR).
burn_std::obfuscate!(
    type: DispatchDevice,
    module: device_opaque,
    derives: [Send, Sync]
);

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
            blob: device_opaque::Opaque::new(device.into()),
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

    /// Remote device identified by a network address (e.g. `"ws://127.0.0.1:3000"`) and the
    /// index of the device to select on that server.
    ///
    /// Requires a running [`burn-remote`](burn_dispatch::backends::remote) server at the given
    /// address. Operations on tensors created with this device are shipped to the server and
    /// executed there. The `index` selects which of the server's devices to use (the server
    /// hosts all of its backend's devices); use [`DeviceIndex::Default`] for the server's
    /// default device. Two remote devices with the same address but different indices target
    /// distinct devices on the same host.
    ///
    /// ```rust,ignore
    /// Device::remote("ws://host:3000", 0);                    // first device
    /// Device::remote("ws://host:3000", 1);                    // second device on same host
    /// Device::remote("ws://host:3000", DeviceIndex::Default); // server-chosen default
    /// ```
    #[cfg(feature = "remote")]
    pub fn remote(address: &str, index: impl Into<DeviceIndex>) -> Self {
        let index = index.into().resolve();
        let device = burn_dispatch::devices::RemoteDevice::new(address, index);
        device.connect(); // initializes the connection (required to get the device default settings)
        Self::new(device)
    }

    /// WGPU device, selected via [`DeviceKind`].
    ///
    /// This variant uses the runtime [`AutoCompiler`](burn_dispatch::backends::wgpu::AutoCompiler)
    /// to dispatch to the most appropriate shader language (WGSL, SPIR-V, or MSL) based on the
    /// enabled features.
    ///
    /// For [`DeviceKind::DefaultDevice`], the adapter is picked by `wgpu`'s
    /// selection heuristics (high-power GPU preferred, or overridden by
    /// `CUBECL_WGPU_DEFAULT_DEVICE`).
    ///
    /// `Device::vulkan`, `Device::metal`, and `Device::webgpu` also use the Wgpu runtime,
    /// but bypass runtime dispatch by pinning specific compilers at compile time.
    #[cfg(feature = "wgpu")]
    pub fn wgpu(device_kind: DeviceKind) -> Self {
        Self::new(DispatchDevice::Wgpu(wgpu_device(device_kind)))
    }

    #[cfg(all(feature = "wgpu", target_family = "wasm"))]
    /// Asynchronously creates a WGPU device, initializing the client.
    pub async fn wgpu_async(device_kind: DeviceKind) -> Self {
        Self::new(DispatchDevice::Wgpu(wgpu_init_async(device_kind).await))
    }

    /// Vulkan-backed WGPU device, selected via [`DeviceKind`].
    ///
    /// Pins the wgpu shader compiler to SPIR-V at compile time, avoiding
    /// the runtime [`AutoCompiler`](burn_dispatch::backends::wgpu::AutoCompiler) dispatch.
    #[cfg(feature = "vulkan")]
    pub fn vulkan(device_kind: DeviceKind) -> Self {
        Self::new(DispatchDevice::Vulkan(wgpu_device(device_kind)))
    }

    /// Metal-backed WGPU device, selected via [`DeviceKind`].
    ///
    /// Pins the wgpu shader compiler to MSL at compile time.
    #[cfg(feature = "metal")]
    pub fn metal(device_kind: DeviceKind) -> Self {
        Self::new(DispatchDevice::Metal(wgpu_device(device_kind)))
    }

    /// WebGPU-backed device, selected via [`DeviceKind`].
    ///
    /// Pins the wgpu shader compiler to WGSL at compile time.
    #[cfg(feature = "webgpu")]
    pub fn webgpu(device_kind: DeviceKind) -> Self {
        Self::new(DispatchDevice::WebGpu(wgpu_device(device_kind)))
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

    /// Triggers a memory cleanup on this device.
    ///
    /// The amount of memory reclaimed depends on the allocator implementation.
    /// Calling this method does not guarantee that any memory will be freed.
    pub fn memory_cleanup(&self) {
        Dispatch::memory_cleanup(self.as_dispatch());
    }

    /// Prepares the given data for transfer between the CPU and accelerator devices such as GPUs.
    ///
    /// Depending on the backend, the data may be transferred to pinned memory
    /// or another transfer-optimized format to improve transfer performance.
    pub fn staging<'a, Iter>(&self, data: Iter)
    where
        Iter: Iterator<Item = &'a mut TensorData>,
    {
        Dispatch::staging(data, self.as_dispatch());
    }

    /// Returns the [`DeviceSettings`] for this device.
    ///
    /// Settings include the default float and integer data types used when creating
    /// tensors on this device.
    ///
    /// See [`configure`](Device::configure) to configure them.
    pub fn settings(&self) -> DeviceSettings {
        burn_backend::get_device_settings::<Dispatch>(self.as_dispatch())
    }

    /// Configures the [settings](DeviceSettings) for this device.
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
    /// device.configure((FloatDType::F16, IntDType::I32))?
    ///
    /// // Float tensors will now use F16
    /// let floats = Tensor::<2>::zeros([2, 3], &device);
    /// // Int tensors will now use I32
    /// let ints = Tensor::<2, Int>::zeros([2, 3], &device);
    /// ```
    pub fn configure(&mut self, config: impl Into<DeviceConfig>) -> Result<(), DeviceError> {
        let mut config = config.into();

        let defaults = self.as_dispatch().defaults();

        let float_dtype = config.float_dtype.take().unwrap_or(defaults.float_dtype);
        let int_dtype = config.int_dtype.take().unwrap_or(defaults.int_dtype);
        let bool_dtype = config.bool_dtype.take().unwrap_or(defaults.bool_dtype);
        let complex_dtype = config.complex_dtype.map(DType::from);

        burn_backend::set_default_dtypes::<Dispatch>(
            self.as_dispatch(),
            float_dtype,
            int_dtype,
            bool_dtype,
            complex_dtype,
        )
    }

    /// Retrieves all available [`Device`]s that match the given [`DeviceType`] filter.
    ///
    /// Local backends (CPU, CUDA, WGPU, …) enumerate the hardware found on the host. The
    /// [`Remote`](DeviceType::Remote) variant instead lists every device hosted by the
    /// `burn-remote` server at the given address — it connects to the server to learn how
    /// many devices it exposes:
    ///
    /// ```rust,ignore
    /// // Every CUDA device on this machine.
    /// let local = Device::enumerate(DeviceType::Cuda);
    ///
    /// // Every device hosted by a remote server.
    /// let remote = Device::enumerate(DeviceType::remote("ws://host:3000"));
    ///
    /// // Filters combine with `|`.
    /// let both = Device::enumerate(DeviceType::Cuda | DeviceType::remote("ws://host:3000"));
    /// ```
    pub fn enumerate(filter: impl Into<DeviceFilter>) -> Devices {
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
                #[cfg(feature = "wgpu")]
                DeviceType::Wgpu => DispatchDeviceId::Wgpu,
                #[cfg(feature = "metal")]
                DeviceType::Metal => DispatchDeviceId::Metal,
                #[cfg(feature = "vulkan")]
                DeviceType::Vulkan => DispatchDeviceId::Vulkan,
                #[cfg(feature = "webgpu")]
                DeviceType::WebGpu => DispatchDeviceId::WebGpu,
                #[cfg(feature = "flex")]
                DeviceType::Flex => DispatchDeviceId::Flex,
                #[cfg(feature = "ndarray")]
                DeviceType::NdArray => DispatchDeviceId::NdArray,
                #[cfg(feature = "tch")]
                DeviceType::LibTorch => DispatchDeviceId::LibTorch,
                // Remote devices are keyed by address, not a backend type id, so they take a
                // dedicated enumeration path (connecting to the server for its device count).
                #[cfg(feature = "remote")]
                DeviceType::Remote(address) => {
                    for device in Dispatch::enumerate_remote(&address) {
                        devices.push(Device::new(device));
                    }
                    continue;
                }
            };

            #[allow(unreachable_code)] // need to have one backend enabled, so it is reachable
            for device in Dispatch::enumerate(type_id) {
                devices.push(Device::new(device))
            }
        }

        Devices(devices)
    }
}

/// Map our backend-agnostic [`DeviceKind`] onto cubecl's `WgpuDevice` enum.
///
/// Shared by [`Device::wgpu`], [`Device::vulkan`], [`Device::metal`], and
/// [`Device::webgpu`], which differ only in which Cargo feature gates them.
#[cfg(feature = "wgpu")]
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

#[cfg(all(feature = "wgpu", target_family = "wasm"))]
// TODO: this is only helpful for the default graphics api and runtime options.. we'd have to expose other methods but that leaks the types
// so we might have to introduce some wrapper types.
async fn wgpu_init_async(device_kind: DeviceKind) -> burn_dispatch::devices::WgpuDevice {
    use burn_dispatch::backends::wgpu::{graphics::AutoGraphicsApi, init_setup_async};

    let device = wgpu_device(device_kind);
    init_setup_async::<AutoGraphicsApi>(&device, Default::default()).await;
    device
}

/// Represents the devices that can be used.
///
/// `DeviceType` is used to filter the available device types for [`Device::enumerate`]. Most
/// variants are fieldless and select a backend's local hardware; [`Remote`](Self::Remote)
/// carries the network address of a `burn-remote` server whose devices should be listed.
///
/// Variants combine into a [`DeviceFilter`] with the `|` operator, so a single
/// [`Device::enumerate`] call can span several backends and remote hosts.
#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceType {
    #[cfg(feature = "cpu")]
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    #[cfg(feature = "rocm")]
    Rocm,
    #[cfg(feature = "wgpu")]
    Wgpu,
    #[cfg(feature = "metal")]
    Metal,
    #[cfg(feature = "vulkan")]
    Vulkan,
    #[cfg(feature = "webgpu")]
    WebGpu,
    #[cfg(feature = "flex")]
    Flex,
    #[cfg(feature = "ndarray")]
    NdArray,
    #[cfg(feature = "tch")]
    LibTorch,
    /// Devices hosted by the `burn-remote` server at the given address
    /// (e.g. `"ws://host:3000"`). Unlike the other variants this is resolved at runtime by
    /// connecting to the server, which reports how many devices it exposes.
    #[cfg(feature = "remote")]
    Remote(String),
}

#[cfg(feature = "remote")]
impl DeviceType {
    /// Filter selecting every device hosted by the `burn-remote` server at `address`
    /// (e.g. `"ws://host:3000"`).
    ///
    /// Convenience for [`DeviceType::Remote`] that accepts anything string-like.
    pub fn remote(address: impl Into<String>) -> Self {
        DeviceType::Remote(address.into())
    }
}

/// A set of [`DeviceType`]s passed to [`Device::enumerate`].
///
/// Built from a single [`DeviceType`], a `Vec<DeviceType>`, or by combining variants with the
/// `|` operator (`DeviceType::Cuda | DeviceType::Cpu`). Because [`DeviceType::Remote`] carries
/// an address, this is a plain list rather than a bitset.
#[derive(Debug, Clone, Default)]
pub struct DeviceFilter(Vec<DeviceType>);

impl DeviceFilter {
    /// Create an empty filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a [`DeviceType`] to the filter.
    pub fn with(mut self, device_type: DeviceType) -> Self {
        self.0.push(device_type);
        self
    }
}

impl From<DeviceType> for DeviceFilter {
    fn from(value: DeviceType) -> Self {
        DeviceFilter(vec![value])
    }
}

impl From<Vec<DeviceType>> for DeviceFilter {
    fn from(value: Vec<DeviceType>) -> Self {
        DeviceFilter(value)
    }
}

impl IntoIterator for DeviceFilter {
    type Item = DeviceType;
    type IntoIter = alloc::vec::IntoIter<DeviceType>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl core::ops::BitOr for DeviceType {
    type Output = DeviceFilter;
    fn bitor(self, rhs: Self) -> DeviceFilter {
        DeviceFilter(vec![self, rhs])
    }
}

impl core::ops::BitOr<DeviceType> for DeviceFilter {
    type Output = DeviceFilter;
    fn bitor(mut self, rhs: DeviceType) -> DeviceFilter {
        self.0.push(rhs);
        self
    }
}

/// Configuration options used to initialize a device.
///
/// Unlike [`DeviceSettings`], this type represents partial user-provided
/// configuration and does not require all settings to be specified.
///
/// Any unspecified options will be resolved to device-specific defaults
/// when the device is initialized.
///
/// Use [`Device::configure`] to apply this configuration to a device.
#[derive(new, Debug, Clone, Default)]
pub struct DeviceConfig {
    /// Default floating-point data type.
    pub float_dtype: Option<FloatDType>,

    /// Default integer data type.
    pub int_dtype: Option<IntDType>,

    /// Default boolean data type.
    pub bool_dtype: Option<BoolDType>,
    /// Default complex data type.
    pub complex_dtype: Option<ComplexDType>,
    // TODO: maybe quantization, but for now we keep this as device defaults
}

impl DeviceConfig {
    /// Sets the default floating-point data type for tensors created on the device.
    pub fn float_dtype(mut self, dtype: impl Into<FloatDType>) -> Self {
        self.float_dtype = Some(dtype.into());
        self
    }

    /// Sets the default integer data type for tensors created on the device.
    pub fn int_dtype(mut self, dtype: impl Into<IntDType>) -> Self {
        self.int_dtype = Some(dtype.into());
        self
    }

    /// Sets the default boolean data type storage precision for tensors created on the device.
    pub fn bool_dtype(mut self, dtype: impl Into<BoolDType>) -> Self {
        self.bool_dtype = Some(dtype.into());
        self
    }

    /// Sets the default complex data type for tensors created on the device.
    pub fn complex_dtype(mut self, dtype: impl Into<ComplexDType>) -> Self {
        self.complex_dtype = Some(dtype.into());
        self
    }
}

impl From<FloatDType> for DeviceConfig {
    fn from(value: FloatDType) -> Self {
        DeviceConfig::new(Some(value), None, None, None)
    }
}

impl From<IntDType> for DeviceConfig {
    fn from(value: IntDType) -> Self {
        DeviceConfig::new(None, Some(value), None, None)
    }
}

impl From<BoolDType> for DeviceConfig {
    fn from(value: BoolDType) -> Self {
        DeviceConfig::new(None, None, Some(value), None)
    }
}

impl From<ComplexDType> for DeviceConfig {
    fn from(value: ComplexDType) -> Self {
        DeviceConfig::new(None, None, None, Some(value))
    }
}

impl From<(FloatDType, IntDType)> for DeviceConfig {
    fn from(value: (FloatDType, IntDType)) -> Self {
        DeviceConfig::new(Some(value.0), Some(value.1), None, None)
    }
}

/// A collection of [`Device`]s returned by [`Device::enumerate`].
///
/// This type provides bulk operations and transformations over multiple
/// devices, such as enabling autodiff or configuring the device settings.
///
/// # Example
///
/// ```rust,ignore
/// let mut devices = Device::enumerate(DeviceType::Cuda)
///     .autodiff();
///
/// devices.configure(
///     DeviceConfig::default().float_dtype(FloatDType::F16),
/// )?;
/// ```
///
/// `Devices` dereferences to a slice of [`Device`], so it can be iterated,
/// indexed, and passed anywhere a `&[Device]` is expected.
pub struct Devices(Vec<Device>);

impl Devices {
    /// Enables autodiff across all contained devices.
    ///
    /// Only first-order autodiff is supported. Calling this method on a device that
    /// already has autodiff enabled will panic.
    ///
    /// See [`Device::autodiff`].
    #[cfg(feature = "autodiff")]
    pub fn autodiff(mut self) -> Self {
        for device in &mut self.0 {
            *device = core::mem::take(device).autodiff();
        }

        self
    }

    /// Configures the [settings](DeviceSettings) for all devices.
    ///
    /// This configures the dtype used when no explicit type is specified at tensor
    /// creation time.
    ///
    /// Settings can only be initialized once per device, and must happen before any
    /// tensor is created on the device. The first tensor operation will lock the device
    /// to its defaults, causing subsequent initializations attempt to return
    /// [`DeviceError::AlreadyInitialized`].
    ///
    /// See [`Device::configure`].
    pub fn configure(&mut self, config: impl Into<DeviceConfig>) -> Result<(), DeviceError> {
        let config = config.into();
        for device in &mut self.0 {
            device.configure(config.clone())?;
        }
        Ok(())
    }

    /// Returns the `Vec` of [`Device`]s.
    pub fn into_vec(self) -> Vec<Device> {
        self.0
    }
}

// Loop over `&Devices` or `Devices` seamlessly
impl IntoIterator for Devices {
    type Item = Device;
    type IntoIter = alloc::vec::IntoIter<Device>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl core::ops::Deref for Devices {
    type Target = [Device];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(all(test, feature = "flex", feature = "autodiff"))]
mod autodiff_move_tests {
    use crate::{Device, Tensor};

    // A non-tracked float tensor (e.g. a gradient) can be moved onto an autodiff device; it
    // lands on the underlying hardware and stays non-tracked. Regression test for a panic in
    // `float_to_device` ("Cannot move between autodiff and non-autodiff instances").
    #[test]
    fn move_non_autodiff_float_tensor_to_autodiff_device() {
        let device = Device::default();
        let ad_device = device.clone().autodiff();

        let t = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
        let moved = t.to_device(&ad_device);

        assert_eq!(
            moved.to_data().to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }
}
