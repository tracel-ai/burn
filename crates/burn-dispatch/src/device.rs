use burn_backend::{DeviceId, DeviceOps, DeviceSettings};

use crate::devices::*;

#[cfg(feature = "capture")]
use burn_capture::CaptureDevice;

#[cfg(feature = "autodiff")]
use alloc::boxed::Box;

// Throughput types come from `burn-backend` (which re-exports them from cubecl),
// so `burn-dispatch` needs no direct `cubecl` dependency.
#[cfg(feature = "cubecl")]
use alloc::vec::Vec;
#[cfg(feature = "cubecl")]
use burn_backend::cubecl::{ThroughputError, ThroughputKey, ThroughputValue};
// `cubecl` without a runtime feature gives the throughput *types* but no `Cube` device to
// measure, so the measurement itself follows `cube_backend` rather than the feature.
#[cfg(cube_backend)]
use burn_backend::cubecl::measure_peak_throughput;

/// Represents a device for the [`Dispatch`](crate::Dispatch).
///
/// Each variant corresponds to a backend that the [`Dispatch`](crate::Dispatch) can dispatch operations to.
///
/// # Example
///
/// ```ignore
/// use burn::DispatchDevice;
///
/// // One variant covers every cubecl runtime; the device inside says which.
/// #[cfg(feature = "cuda")]
/// let cuda_device = DispatchDevice::Cube(cubecl::Device::Cuda(Default::default()));
///
/// #[cfg(feature = "ndarray")]
/// let ndarray_device = DispatchDevice::NdArray(Default::default());
/// ```
#[derive(Clone, Eq)]
pub enum DispatchDevice {
    /// A device of the [cubecl backend](crate::backends::Cube): CUDA, ROCm, Metal, Vulkan,
    /// WebGPU, wgpu or the CPU runtime.
    #[cfg(cube_backend)]
    Cube(CubeDevice),

    /// The [Flex backend](crate::backends::Flex) device (CPU-only).
    #[cfg(any(feature = "flex", default_backend))]
    Flex(FlexDevice),

    /// The [NdArray backend](crate::backends::NdArray) device (CPU-only).
    #[cfg(feature = "ndarray")]
    NdArray(NdArrayDevice),

    /// The [LibTorch backend](crate::backends::LibTorch) device.
    #[cfg(feature = "tch")]
    LibTorch(LibTorchDevice),

    /// The [remote backend](crate::backends::Remote) device, identified by a network address.
    #[cfg(feature = "remote")]
    Remote(RemoteDevice),

    /// A non-executing graph capture device.
    #[cfg(feature = "capture")]
    Capture(CaptureDevice),

    /// The [autodiff enabled backend](crate::backends::Autodiff) device.
    #[cfg(feature = "autodiff")]
    Autodiff(AutodiffDevice),
}

#[cfg(feature = "cubecl")]
impl DispatchDevice {
    /// Measure peak throughput for this device against the given `keys`.
    ///
    /// Only cubecl-backed devices can measure throughput; other backends
    /// (ndarray, libtorch, remote, ...) return an empty vector. An autodiff
    /// device reports the peaks of the device it wraps. Each returned result
    /// corresponds positionally to the key at the same index, and carries a
    /// [`ThroughputError`] where the device has no peak for that key.
    // With `cubecl` on but no runtime compiled in, every arm below ignores `keys`.
    #[cfg_attr(not(cube_backend), allow(unused_variables))]
    pub fn performance_stats(
        &self,
        keys: &[ThroughputKey],
    ) -> Vec<Result<ThroughputValue, ThroughputError>> {
        // No catch-all arm: a new backend must fail to compile here rather
        // than silently report no peaks.
        match self {
            #[cfg(cube_backend)]
            DispatchDevice::Cube(device) => {
                let client = device.client();
                keys.iter()
                    .map(|key| measure_peak_throughput(&client, *key))
                    .collect()
            }
            // Autodiff does not change the hardware, so measure the wrapped device.
            #[cfg(feature = "autodiff")]
            DispatchDevice::Autodiff(device) => device.performance_stats(keys),

            // Not cubecl-backed, so there are no kernels to measure.
            #[cfg(any(feature = "flex", default_backend))]
            DispatchDevice::Flex(_) => Vec::new(),
            #[cfg(feature = "ndarray")]
            DispatchDevice::NdArray(_) => Vec::new(),
            #[cfg(feature = "tch")]
            DispatchDevice::LibTorch(_) => Vec::new(),

            // The kernels run on the server, which this local API cannot reach.
            #[cfg(feature = "remote")]
            DispatchDevice::Remote(_) => Vec::new(),
            #[cfg(feature = "capture")]
            DispatchDevice::Capture(_) => Vec::new(),
        }
    }
}

#[cfg(feature = "autodiff")]
// This tuple struct mainly restricts users from creating Autodiff(Autodiff) devices.
/// A wrapper that enables automatic differentiation for a [`DispatchDevice`].
///
/// Use [`DispatchDevice::autodiff`] to construct this type.
#[derive(Debug, Clone)]
pub struct AutodiffDevice {
    pub(crate) inner: Box<DispatchDevice>,
    pub(crate) checkpointing: GradientCheckpointingStrategy,
}

/// Compares on hardware identity only, ignoring the checkpointing strategy, so that this agrees
/// with [`DispatchDevice`]'s own [`PartialEq`] — which has to ignore it, since comparing an
/// `Autodiff` device against a raw one has no strategy to compare against. A derived impl would
/// make `Autodiff(a) == Autodiff(b)` disagree with `DispatchDevice::Autodiff(a) ==
/// DispatchDevice::Autodiff(b)`.
///
/// Use [`gradient_checkpointing_strategy`](Self::gradient_checkpointing_strategy) when the
/// strategy is what you actually need to compare.
#[cfg(feature = "autodiff")]
impl PartialEq for AutodiffDevice {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

#[cfg(feature = "autodiff")]
impl Eq for AutodiffDevice {}

#[cfg(feature = "autodiff")]
impl AutodiffDevice {
    pub(crate) fn new(
        device: DispatchDevice,
        checkpointing: GradientCheckpointingStrategy,
    ) -> Self {
        Self {
            inner: Box::new(device),
            checkpointing,
        }
    }

    /// Returns the underlying device, removing the autodiff capability.
    pub fn inner(self) -> DispatchDevice {
        *self.inner
    }

    /// Returns the gradient checkpointing strategy.
    pub fn gradient_checkpointing_strategy(&self) -> GradientCheckpointingStrategy {
        self.checkpointing
    }
}

#[cfg(feature = "autodiff")]
// Useful for match in dispatch macros
impl core::ops::Deref for AutodiffDevice {
    type Target = DispatchDevice;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Gradient checkpointing strategy for autodiff.
#[repr(u8)]
pub enum GradientCheckpointingStrategy {
    /// Recompute selected activations during backpropagation to reduce peak memory usage.
    Balanced,
    /// Disable gradient checkpointing while retaining autodiff tracking.
    #[default]
    Disabled,
}

impl core::fmt::Debug for DispatchDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            #[cfg(cube_backend)]
            Self::Cube(device) => f.debug_tuple("Cube").field(device).finish(),
            #[cfg(any(feature = "flex", default_backend))]
            Self::Flex(device) => f.debug_tuple("Flex").field(device).finish(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(device) => f.debug_tuple("NdArray").field(device).finish(),
            #[cfg(feature = "tch")]
            Self::LibTorch(device) => f.debug_tuple("LibTorch").field(device).finish(),
            #[cfg(feature = "remote")]
            Self::Remote(device) => f.debug_tuple("Remote").field(device).finish(),
            #[cfg(feature = "capture")]
            Self::Capture(device) => f.debug_tuple("Capture").field(device).finish(),
            #[cfg(feature = "autodiff")]
            // Format without `AutodiffDevice` wrapper
            Self::Autodiff(device) => f
                .debug_struct("Autodiff")
                .field("device", &device.inner)
                .field("checkpointing", &device.checkpointing)
                .finish(),
        }
    }
}

impl Default for DispatchDevice {
    #[allow(unreachable_code)]
    fn default() -> Self {
        // TODO: which priority?
        // Single override e.g. `BURN_DEVICE=vulkan` forces Vulkan or panics if not available.
        // Priority list e.g. `BURN_DEVICE_PRIORITY=cuda,vulkan,cpu` sets the order.
        // Both could be tied into `burn.toml` config
        // For now we just use `BURN_DEVICE` on CI to force a single device

        #[cfg(feature = "std")]
        {
            if let Ok(device_str) = std::env::var("BURN_DEVICE") {
                match device_str.to_lowercase().as_str() {
                    // Every cubecl runtime is the one `Cube` backend; the name here
                    // picks the runtime the device names, and the wgpu spellings all
                    // reach wgpu, whose compiler is chosen for it at runtime.
                    "cuda" => {
                        #[cfg(feature = "cuda")]
                        return Self::Cube(CubeDevice::Cuda(Default::default()));
                        panic!(
                            "BURN_DEVICE=cuda requested, but the 'cuda' feature is not enabled."
                        );
                    }
                    "rocm" => {
                        #[cfg(feature = "rocm")]
                        return Self::Cube(CubeDevice::Hip(Default::default()));
                        panic!(
                            "BURN_DEVICE=rocm requested, but the 'rocm' feature is not enabled."
                        );
                    }
                    "metal" | "vulkan" | "webgpu" | "wgpu" => {
                        #[cfg(any(
                            feature = "metal",
                            feature = "vulkan",
                            feature = "webgpu",
                            feature = "wgpu"
                        ))]
                        return Self::Cube(CubeDevice::Wgpu(Default::default()));
                        panic!(
                            "BURN_DEVICE={device_str} requested, but no wgpu feature is enabled."
                        );
                    }
                    "cpu" => {
                        #[cfg(feature = "cpu")]
                        return Self::Cube(CubeDevice::Cpu(Default::default()));
                        panic!("BURN_DEVICE=cpu requested, but the 'cpu' feature is not enabled.");
                    }
                    "tch" => {
                        #[cfg(feature = "tch")]
                        return Self::LibTorch(LibTorchDevice::default());
                        panic!("BURN_DEVICE=tch requested, but the 'tch' feature is not enabled.");
                    }
                    "remote" => {
                        #[cfg(feature = "remote")]
                        return Self::Remote(RemoteDevice::default());
                        panic!(
                            "BURN_DEVICE=remote requested, but the 'remote' feature is not enabled."
                        );
                    }
                    "flex" => {
                        #[cfg(any(feature = "flex", default_backend))]
                        return Self::Flex(FlexDevice);
                        panic!(
                            "BURN_DEVICE=flex requested, but the 'flex' feature is not enabled."
                        );
                    }
                    "ndarray" => {
                        #[cfg(feature = "ndarray")]
                        return Self::NdArray(NdArrayDevice::default());
                        panic!(
                            "BURN_DEVICE=ndarray requested, but the 'ndarray' feature is not enabled."
                        );
                    }
                    _ => panic!("Unknown BURN_DEVICE override: '{}'.", device_str),
                }
            }
        }

        // Spelled out per feature rather than left to `CubeDevice::default()`: that answers for
        // the runtimes *cubecl* compiled in, and cargo unifies features across a build, so a
        // workspace that also builds `burn-cuda` would hand this crate a CUDA default even when
        // it was built with only `wgpu`. The order is the one a caller who did not choose would
        // want — a discrete accelerator, then the portable path, then the CPU.
        #[cfg(feature = "cuda")]
        return Self::Cube(CubeDevice::Cuda(Default::default()));

        #[cfg(feature = "metal")]
        return Self::Cube(CubeDevice::Wgpu(Default::default()));

        #[cfg(feature = "rocm")]
        return Self::Cube(CubeDevice::Hip(Default::default()));

        #[cfg(feature = "vulkan")]
        return Self::Cube(CubeDevice::Wgpu(Default::default()));

        #[cfg(feature = "webgpu")]
        return Self::Cube(CubeDevice::Wgpu(Default::default()));

        #[cfg(feature = "wgpu")]
        return Self::Cube(CubeDevice::Wgpu(Default::default()));

        #[cfg(feature = "cpu")]
        return Self::Cube(CubeDevice::Cpu(Default::default()));

        #[cfg(feature = "tch")]
        return Self::LibTorch(LibTorchDevice::default());

        // Prefer Flex over NdArray when both are enabled: Flex is the long-term
        // CPU backend replacement and should win the default tie.
        #[cfg(any(feature = "flex", default_backend))]
        return Self::Flex(FlexDevice);

        #[cfg(feature = "remote")]
        return Self::Remote(RemoteDevice::default());

        #[cfg(feature = "ndarray")]
        return Self::NdArray(NdArrayDevice::default());
    }
}

impl PartialEq for DispatchDevice {
    /// Compares devices based on hardware identity.
    ///
    /// Returns `true` if both devices represent the same compute resource.
    /// Note that this comparison ignores autodiff and checkpointing settings.
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // If both are Autodiff, compare the inner devices
            #[cfg(feature = "autodiff")]
            (DispatchDevice::Autodiff(a), DispatchDevice::Autodiff(b)) => {
                a.inner.as_ref() == b.inner.as_ref()
            }
            // If one is Autodiff, compare it to the raw device
            #[cfg(feature = "autodiff")]
            (DispatchDevice::Autodiff(a), b) => a.inner.as_ref() == b,
            #[cfg(feature = "autodiff")]
            (a, DispatchDevice::Autodiff(b)) => a == b.inner.as_ref(),
            #[cfg(cube_backend)]
            (Self::Cube(a), Self::Cube(b)) => a == b,
            #[cfg(any(feature = "flex", default_backend))]
            (Self::Flex(a), Self::Flex(b)) => a == b,
            #[cfg(feature = "ndarray")]
            (Self::NdArray(a), Self::NdArray(b)) => a == b,
            #[cfg(feature = "tch")]
            (Self::LibTorch(a), Self::LibTorch(b)) => a == b,
            #[cfg(feature = "remote")]
            (Self::Remote(a), Self::Remote(b)) => a == b,
            #[cfg(feature = "capture")]
            (Self::Capture(a), Self::Capture(b)) => a == b,
            #[allow(unreachable_patterns)]
            (_, _) => false,
        }
    }
}

const INTERNAL_ID_MASK: u16 = 0x00FF;
const BACKEND_SHIFT: u32 = 8;

impl DispatchDevice {
    /// Create the dispatch representation used by the high-level graph-capture device.
    #[cfg(feature = "capture")]
    #[doc(hidden)]
    pub fn capture() -> Self {
        Self::Capture(CaptureDevice::default())
    }

    #[cfg(feature = "autodiff")]
    /// Creates a new [`DispatchDevice`] with
    /// [automatic differentiation](crate::backends::Autodiff) enabled.
    pub fn autodiff(device: impl Into<DispatchDevice>) -> DispatchDevice {
        Self::autodiff_with_gradient_checkpointing(device, GradientCheckpointingStrategy::Disabled)
    }
    #[cfg(feature = "autodiff")]
    /// Creates a new [`DispatchDevice`] with automatic differentiation and the provided gradient
    /// checkpointing strategy enabled.
    pub fn autodiff_with_gradient_checkpointing(
        device: impl Into<DispatchDevice>,
        checkpointing: GradientCheckpointingStrategy,
    ) -> DispatchDevice {
        let device = device.into();
        DispatchDevice::Autodiff(AutodiffDevice::new(device, checkpointing))
    }

    /// Returns the inner device, without autodiff (when enabled).
    pub fn inner(self) -> Self {
        #[cfg(feature = "autodiff")]
        if let DispatchDevice::Autodiff(device) = self {
            return *device.inner;
        }

        self
    }

    /// Returns a unique number per variant to encode into type_id.
    fn backend_id(&self) -> DispatchDeviceId {
        match self {
            #[cfg(cube_backend)]
            Self::Cube(_) => DispatchDeviceId::Cube,
            #[cfg(any(feature = "flex", default_backend))]
            Self::Flex(_) => DispatchDeviceId::Flex,
            #[cfg(feature = "ndarray")]
            Self::NdArray(_) => DispatchDeviceId::NdArray,
            #[cfg(feature = "tch")]
            Self::LibTorch(_) => DispatchDeviceId::LibTorch,
            #[cfg(feature = "remote")]
            Self::Remote(_) => DispatchDeviceId::Remote,
            #[cfg(feature = "capture")]
            Self::Capture(_) => DispatchDeviceId::Capture,
            #[cfg(feature = "autodiff")]
            Self::Autodiff(device) => device.inner.backend_id(),
        }
    }

    /// Encode variant ID and backend type ID into a unique `type_id`.
    fn encode_type_id(&self, backend_type_id: u16) -> u16 {
        // Use the lower 8 bits for the backend's internal type ID
        let internal_type_id = backend_type_id & INTERNAL_ID_MASK;
        // Use the upper 8 bits for the DispatchDevice/DispatchDeviceId
        let backend = u16::from(self.backend_id()) << BACKEND_SHIFT;
        backend | internal_type_id
    }

    /// Decode an encoded `type_id` into variant ID and backend type ID.
    pub(crate) fn decode_type_id(type_id: u16) -> (DispatchDeviceId, u16) {
        let backend_raw = type_id >> BACKEND_SHIFT;
        let internal_type_id = type_id & INTERNAL_ID_MASK;

        let backend = DispatchDeviceId::try_from(backend_raw).expect("Unknown DispatchDevice ID");

        (backend, internal_type_id)
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum DispatchDeviceId {
    /// Every cubecl runtime: which one is in the device's own id.
    Cube = 0,
    Flex = 4,
    LibTorch = 5,
    NdArray = 6,
    Remote = 10,
    Capture = 11,
}

impl From<DispatchDeviceId> for u16 {
    fn from(variant: DispatchDeviceId) -> Self {
        variant as u16
    }
}

impl TryFrom<u16> for DispatchDeviceId {
    type Error = ();

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            #[cfg(cube_backend)]
            0 => Ok(Self::Cube),
            #[cfg(any(feature = "flex", default_backend))]
            4 => Ok(Self::Flex),
            #[cfg(feature = "tch")]
            5 => Ok(Self::LibTorch),
            #[cfg(feature = "ndarray")]
            6 => Ok(Self::NdArray),
            #[cfg(feature = "remote")]
            10 => Ok(Self::Remote),
            #[cfg(feature = "capture")]
            11 => Ok(Self::Capture),
            _ => Err(()),
        }
    }
}

impl DeviceOps for DispatchDevice {
    fn defaults(&self) -> DeviceSettings {
        match self {
            #[cfg(cube_backend)]
            Self::Cube(device) => device.defaults(),
            #[cfg(any(feature = "flex", default_backend))]
            Self::Flex(device) => device.defaults(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(device) => device.defaults(),
            #[cfg(feature = "tch")]
            Self::LibTorch(device) => device.defaults(),
            #[cfg(feature = "remote")]
            Self::Remote(device) => device.defaults(),
            #[cfg(feature = "capture")]
            Self::Capture(device) => device.defaults(),
            #[cfg(feature = "autodiff")]
            Self::Autodiff(device) => device.inner.defaults(),
        }
    }
}

impl burn_backend::Device for DispatchDevice {
    fn from_id(mut device_id: DeviceId) -> Self {
        let (dispatch_id, backend_type_id) = Self::decode_type_id(device_id.type_id);
        device_id.type_id = backend_type_id;

        match dispatch_id {
            #[cfg(cube_backend)]
            DispatchDeviceId::Cube => Self::Cube(burn_backend::Device::from_id(device_id)),
            #[cfg(any(feature = "flex", default_backend))]
            DispatchDeviceId::Flex => Self::Flex(FlexDevice::from_id(device_id)),
            #[cfg(feature = "ndarray")]
            DispatchDeviceId::NdArray => Self::NdArray(NdArrayDevice::from_id(device_id)),
            #[cfg(feature = "tch")]
            DispatchDeviceId::LibTorch => Self::LibTorch(LibTorchDevice::from_id(device_id)),
            #[cfg(feature = "remote")]
            DispatchDeviceId::Remote => Self::Remote(RemoteDevice::from_id(device_id)),
            #[cfg(feature = "capture")]
            DispatchDeviceId::Capture => Self::Capture(CaptureDevice::from_id(device_id)),
            _ => unreachable!("No backend feature enabled."),
        }
    }

    fn to_id(&self) -> DeviceId {
        let mut device_id = match self {
            #[cfg(cube_backend)]
            Self::Cube(device) => device.to_id(),
            #[cfg(any(feature = "flex", default_backend))]
            Self::Flex(device) => device.to_id(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(device) => device.to_id(),
            #[cfg(feature = "tch")]
            Self::LibTorch(device) => device.to_id(),
            #[cfg(feature = "remote")]
            Self::Remote(device) => device.to_id(),
            #[cfg(feature = "capture")]
            Self::Capture(device) => device.to_id(),
            #[cfg(feature = "autodiff")]
            Self::Autodiff(device) => device.inner.to_id(),
        };
        device_id.type_id = self.encode_type_id(device_id.type_id);
        device_id
    }
}

/// Every cubecl device reaches the one cubecl variant.
#[cfg(cube_backend)]
impl From<CubeDevice> for DispatchDevice {
    fn from(device: CubeDevice) -> Self {
        DispatchDevice::Cube(device)
    }
}

// A runtime's own device type converts too, since that is what its crate hands
// out. There is one variant to reach now, so a wgpu device no longer needs a
// priority chain of gates to decide which of four it lands in.
#[cfg(feature = "cpu")]
impl From<CpuDevice> for DispatchDevice {
    fn from(device: CpuDevice) -> Self {
        DispatchDevice::Cube(CubeDevice::Cpu(device))
    }
}

#[cfg(feature = "cuda")]
impl From<CudaDevice> for DispatchDevice {
    fn from(device: CudaDevice) -> Self {
        DispatchDevice::Cube(CubeDevice::Cuda(device))
    }
}

#[cfg(feature = "rocm")]
impl From<RocmDevice> for DispatchDevice {
    fn from(device: RocmDevice) -> Self {
        DispatchDevice::Cube(CubeDevice::Hip(device))
    }
}

#[cfg(any(
    feature = "wgpu",
    feature = "metal",
    feature = "vulkan",
    feature = "webgpu"
))]
impl From<WgpuDevice> for DispatchDevice {
    fn from(device: WgpuDevice) -> Self {
        DispatchDevice::Cube(CubeDevice::Wgpu(device))
    }
}

#[cfg(any(feature = "flex", default_backend))]
impl From<FlexDevice> for DispatchDevice {
    fn from(device: FlexDevice) -> Self {
        DispatchDevice::Flex(device)
    }
}

#[cfg(feature = "ndarray")]
impl From<NdArrayDevice> for DispatchDevice {
    fn from(device: NdArrayDevice) -> Self {
        DispatchDevice::NdArray(device)
    }
}

#[cfg(feature = "tch")]
impl From<LibTorchDevice> for DispatchDevice {
    fn from(device: LibTorchDevice) -> Self {
        DispatchDevice::LibTorch(device)
    }
}

#[cfg(feature = "remote")]
impl From<RemoteDevice> for DispatchDevice {
    fn from(device: RemoteDevice) -> Self {
        DispatchDevice::Remote(device)
    }
}

#[cfg(all(test, feature = "capture"))]
mod tests {
    use super::*;
    use burn_backend::Device;

    #[test]
    fn capture_device_id_round_trips_through_dispatch() {
        let device = DispatchDevice::capture();
        let restored = DispatchDevice::from_id(device.to_id());

        assert_eq!(restored, device);
    }
}
