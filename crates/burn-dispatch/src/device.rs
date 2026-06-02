use burn_backend::{DeviceId, DeviceOps, DeviceSettings};

use crate::backends::*;

#[cfg(feature = "autodiff")]
use alloc::boxed::Box;

/// Represents a device for the [`Dispatch`](crate::Dispatch).
///
/// Each variant corresponds to a backend that the [`Dispatch`](crate::Dispatch) can dispatch operations to.
///
/// # Example
///
/// ```ignore
/// use burn::DispatchDevice;
///
/// #[cfg(feature = "cpu")]
/// let cpu_device = DispatchDevice::Cpu(Default::default());
///
/// #[cfg(feature = "cuda")]
/// let cuda_device = DispatchDevice::Cuda(Default::default());
/// ```
#[derive(Clone, Eq)]
pub enum DispatchDevice {
    /// The [CPU backend](Cpu) device.
    #[cfg(feature = "cpu")]
    Cpu(CpuDevice),

    /// The [CUDA backend](Cuda) device.
    #[cfg(feature = "cuda")]
    Cuda(CudaDevice),

    /// The [Metal backend](Metal) device (via WGPU runtime).
    #[cfg(feature = "metal")]
    Metal(WgpuDevice),

    /// The [ROCm backend](Rocm) device.
    #[cfg(feature = "rocm")]
    Rocm(RocmDevice),

    /// The [Vulkan backend](Vulkan) device.
    #[cfg(feature = "vulkan")]
    Vulkan(WgpuDevice),

    /// The [Wgpu backend](Wgpu) device (via WGPU runtime with auto-selected compiler).
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuDevice),

    /// The [WebGPU backend](WebGpu) device (via WGPU runtime).
    #[cfg(feature = "webgpu")]
    WebGpu(WgpuDevice),

    /// The [Flex backend](Flex) device (CPU-only).
    #[cfg(any(feature = "flex", default_backend))]
    Flex(FlexDevice),

    /// The [NdArray backend](NdArray) device (CPU-only).
    #[cfg(feature = "ndarray")]
    NdArray(NdArrayDevice),

    /// The [LibTorch backend](LibTorch) device.
    #[cfg(feature = "tch")]
    LibTorch(LibTorchDevice),

    /// The [remote backend](Remote) device, identified by a network address.
    #[cfg(feature = "remote")]
    Remote(RemoteDevice),

    /// The [autodiff enabled backend](Autodiff) device.
    #[cfg(feature = "autodiff")]
    Autodiff(AutodiffDevice),
}

#[cfg(feature = "autodiff")]
// This tuple struct mainly restricts users from creating Autodiff(Autodiff) devices.
/// A wrapper that enables automatic differentiation for a [`DispatchDevice`].
///
/// Use [`DispatchDevice::autodiff`] to construct this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutodiffDevice {
    pub(crate) inner: Box<DispatchDevice>,
    pub(crate) checkpointing: CheckpointingStrategy,
}

#[cfg(feature = "autodiff")]
impl AutodiffDevice {
    pub(crate) fn new(device: DispatchDevice, checkpointing: CheckpointingStrategy) -> Self {
        Self {
            inner: Box::new(device),
            checkpointing,
        }
    }

    /// Returns the underlying device, removing the autodiff capability.
    pub fn inner(self) -> DispatchDevice {
        *self.inner
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
/// Checkpointing strategy for autodiff.
#[repr(u8)]
pub enum CheckpointingStrategy {
    Balanced,
    #[default]
    None,
}

#[cfg(feature = "autodiff")]
pub(crate) fn validate_checkpointing(
    lhs: Option<crate::CheckpointingStrategy>,
    rhs: Option<crate::CheckpointingStrategy>,
) -> Option<crate::CheckpointingStrategy> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => {
            assert_eq!(
                lhs, rhs,
                "Autodiff strategy mismatch: {lhs:?} vs {rhs:?}. Tensors in the same operation must share a strategy."
            );
            Some(lhs)
        }
        (None, None) => None,
        // When tensors are created on non-autodiff device there is no checkpointing, but
        // tensor created with autodiff which moved out (`tensor.inner()`) will still carry the state.
        // In such cases, we can "promote" the checkpointing.
        (None, rhs) => rhs,
        (lhs, None) => lhs,
    }
}

impl core::fmt::Debug for DispatchDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(device) => f.debug_tuple("Cpu").field(device).finish(),
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => f.debug_tuple("Cuda").field(device).finish(),
            #[cfg(feature = "metal")]
            Self::Metal(device) => f.debug_tuple("Metal").field(device).finish(),
            #[cfg(feature = "rocm")]
            Self::Rocm(device) => f.debug_tuple("Rocm").field(device).finish(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => f.debug_tuple("Vulkan").field(device).finish(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(device) => f.debug_tuple("Wgpu").field(device).finish(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(device) => f.debug_tuple("WebGpu").field(device).finish(),
            #[cfg(any(feature = "flex", default_backend))]
            Self::Flex(device) => f.debug_tuple("Flex").field(device).finish(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(device) => f.debug_tuple("NdArray").field(device).finish(),
            #[cfg(feature = "tch")]
            Self::LibTorch(device) => f.debug_tuple("LibTorch").field(device).finish(),
            #[cfg(feature = "remote")]
            Self::Remote(device) => f.debug_tuple("Remote").field(device).finish(),
            #[cfg(feature = "autodiff")]
            // Format without `AutodiffDevice` wrapper
            Self::Autodiff(device) => f.debug_tuple("Autodiff").field(&device.inner).finish(),
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
                    "cuda" => {
                        #[cfg(feature = "cuda")]
                        return Self::Cuda(CudaDevice::default());
                        panic!(
                            "BURN_DEVICE=cuda requested, but the 'cuda' feature is not enabled."
                        );
                    }
                    "metal" => {
                        #[cfg(feature = "metal")]
                        return Self::Metal(burn_wgpu::WgpuDevice::default());
                        panic!(
                            "BURN_DEVICE=metal requested, but the 'metal' feature is not enabled."
                        );
                    }
                    "rocm" => {
                        #[cfg(feature = "rocm")]
                        return Self::Rocm(RocmDevice::default());
                        panic!(
                            "BURN_DEVICE=rocm requested, but the 'rocm' feature is not enabled."
                        );
                    }
                    "vulkan" => {
                        #[cfg(feature = "vulkan")]
                        return Self::Vulkan(burn_wgpu::WgpuDevice::default());
                        panic!(
                            "BURN_DEVICE=vulkan requested, but the 'vulkan' feature is not enabled."
                        );
                    }
                    "webgpu" => {
                        #[cfg(feature = "webgpu")]
                        return Self::WebGpu(burn_wgpu::WgpuDevice::default());
                        panic!(
                            "BURN_DEVICE=webgpu requested, but the 'webgpu' feature is not enabled."
                        );
                    }
                    "wgpu" => {
                        #[cfg(feature = "wgpu")]
                        return Self::Wgpu(burn_wgpu::WgpuDevice::default());
                        panic!(
                            "BURN_DEVICE=wgpu requested, but the 'wgpu' feature is not enabled."
                        );
                    }
                    "cpu" => {
                        #[cfg(feature = "cpu")]
                        return Self::Cpu(CpuDevice);
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

        #[cfg(feature = "cuda")]
        return Self::Cuda(CudaDevice::default());

        #[cfg(feature = "metal")]
        return Self::Metal(burn_wgpu::WgpuDevice::default());

        #[cfg(feature = "rocm")]
        return Self::Rocm(RocmDevice::default());

        #[cfg(feature = "vulkan")]
        return Self::Vulkan(burn_wgpu::WgpuDevice::default());

        #[cfg(feature = "webgpu")]
        return Self::WebGpu(burn_wgpu::WgpuDevice::default());

        #[cfg(feature = "wgpu")]
        return Self::Wgpu(burn_wgpu::WgpuDevice::default());

        #[cfg(feature = "cpu")]
        return Self::Cpu(CpuDevice);

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
            (DispatchDevice::Autodiff(a), DispatchDevice::Autodiff(b)) => a == b,
            // If one is Autodiff, compare it to the raw device
            #[cfg(feature = "autodiff")]
            (DispatchDevice::Autodiff(a), b) => a.inner.as_ref() == b,
            #[cfg(feature = "autodiff")]
            (a, DispatchDevice::Autodiff(b)) => a == b.inner.as_ref(),
            #[cfg(feature = "cpu")]
            (Self::Cpu(a), Self::Cpu(b)) => a == b,
            #[cfg(feature = "cuda")]
            (Self::Cuda(a), Self::Cuda(b)) => a == b,
            #[cfg(feature = "metal")]
            (Self::Metal(a), Self::Metal(b)) => a == b,
            #[cfg(feature = "rocm")]
            (Self::Rocm(a), Self::Rocm(b)) => a == b,
            #[cfg(feature = "vulkan")]
            (Self::Vulkan(a), Self::Vulkan(b)) => a == b,
            #[cfg(feature = "wgpu")]
            (Self::Wgpu(a), Self::Wgpu(b)) => a == b,
            #[cfg(feature = "webgpu")]
            (Self::WebGpu(a), Self::WebGpu(b)) => a == b,
            #[cfg(any(feature = "flex", default_backend))]
            (Self::Flex(a), Self::Flex(b)) => a == b,
            #[cfg(feature = "ndarray")]
            (Self::NdArray(a), Self::NdArray(b)) => a == b,
            #[cfg(feature = "tch")]
            (Self::LibTorch(a), Self::LibTorch(b)) => a == b,
            #[cfg(feature = "remote")]
            (Self::Remote(a), Self::Remote(b)) => a == b,
            #[allow(unreachable_patterns)]
            (_, _) => false,
        }
    }
}

const INTERNAL_ID_MASK: u16 = 0x00FF;
const BACKEND_SHIFT: u32 = 8;

impl DispatchDevice {
    #[cfg(feature = "autodiff")]
    /// Creates a new [`DispatchDevice`] with [automatic differentiation](Autodiff) enabled.
    pub fn autodiff(device: impl Into<DispatchDevice>) -> DispatchDevice {
        Self::autodiff_checkpointed(device, CheckpointingStrategy::None)
    }
    #[cfg(feature = "autodiff")]
    /// Creates a new [`DispatchDevice`] with [automatic differentiation](Autodiff) enabled.
    pub fn autodiff_checkpointed(
        device: impl Into<DispatchDevice>,
        checkpointing: CheckpointingStrategy,
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
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => DispatchDeviceId::Cpu,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => DispatchDeviceId::Cuda,
            #[cfg(feature = "metal")]
            Self::Metal(_) => DispatchDeviceId::Metal,
            #[cfg(feature = "rocm")]
            Self::Rocm(_) => DispatchDeviceId::Rocm,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => DispatchDeviceId::Vulkan,
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => DispatchDeviceId::Wgpu,
            #[cfg(feature = "webgpu")]
            Self::WebGpu(_) => DispatchDeviceId::WebGpu,
            #[cfg(any(feature = "flex", default_backend))]
            Self::Flex(_) => DispatchDeviceId::Flex,
            #[cfg(feature = "ndarray")]
            Self::NdArray(_) => DispatchDeviceId::NdArray,
            #[cfg(feature = "tch")]
            Self::LibTorch(_) => DispatchDeviceId::LibTorch,
            #[cfg(feature = "remote")]
            Self::Remote(_) => DispatchDeviceId::Remote,
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
    Cpu = 0,
    Cuda = 1,
    Wgpu = 2,
    Rocm = 3,
    Flex = 4,
    LibTorch = 5,
    NdArray = 6,
    Metal = 7,
    Vulkan = 8,
    WebGpu = 9,
    Remote = 10,
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
            #[cfg(feature = "cpu")]
            0 => Ok(Self::Cpu),
            #[cfg(feature = "cuda")]
            1 => Ok(Self::Cuda),
            #[cfg(feature = "wgpu")]
            2 => Ok(Self::Wgpu),
            #[cfg(feature = "rocm")]
            3 => Ok(Self::Rocm),
            #[cfg(any(feature = "flex", default_backend))]
            4 => Ok(Self::Flex),
            #[cfg(feature = "tch")]
            5 => Ok(Self::LibTorch),
            #[cfg(feature = "ndarray")]
            6 => Ok(Self::NdArray),
            #[cfg(feature = "metal")]
            7 => Ok(Self::Metal),
            #[cfg(feature = "vulkan")]
            8 => Ok(Self::Vulkan),
            #[cfg(feature = "webgpu")]
            9 => Ok(Self::WebGpu),
            #[cfg(feature = "remote")]
            10 => Ok(Self::Remote),
            _ => Err(()),
        }
    }
}

impl DeviceOps for DispatchDevice {
    fn defaults(&self) -> DeviceSettings {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(device) => device.defaults(),
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => device.defaults(),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.defaults(),
            #[cfg(feature = "rocm")]
            Self::Rocm(device) => device.defaults(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.defaults(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(device) => device.defaults(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(device) => device.defaults(),
            #[cfg(any(feature = "flex", default_backend))]
            Self::Flex(device) => device.defaults(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(device) => device.defaults(),
            #[cfg(feature = "tch")]
            Self::LibTorch(device) => device.defaults(),
            #[cfg(feature = "remote")]
            Self::Remote(device) => device.defaults(),
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
            #[cfg(feature = "cpu")]
            DispatchDeviceId::Cpu => Self::Cpu(CpuDevice::from_id(device_id)),
            #[cfg(feature = "cuda")]
            DispatchDeviceId::Cuda => Self::Cuda(CudaDevice::from_id(device_id)),
            #[cfg(feature = "metal")]
            DispatchDeviceId::Metal => Self::Metal(WgpuDevice::from_id(device_id)),
            #[cfg(feature = "rocm")]
            DispatchDeviceId::Rocm => Self::Rocm(RocmDevice::from_id(device_id)),
            #[cfg(feature = "vulkan")]
            DispatchDeviceId::Vulkan => Self::Vulkan(WgpuDevice::from_id(device_id)),
            #[cfg(feature = "wgpu")]
            DispatchDeviceId::Wgpu => Self::Wgpu(WgpuDevice::from_id(device_id)),
            #[cfg(feature = "webgpu")]
            DispatchDeviceId::WebGpu => Self::WebGpu(WgpuDevice::from_id(device_id)),
            #[cfg(any(feature = "flex", default_backend))]
            DispatchDeviceId::Flex => Self::Flex(FlexDevice::from_id(device_id)),
            #[cfg(feature = "ndarray")]
            DispatchDeviceId::NdArray => Self::NdArray(NdArrayDevice::from_id(device_id)),
            #[cfg(feature = "tch")]
            DispatchDeviceId::LibTorch => Self::LibTorch(LibTorchDevice::from_id(device_id)),
            #[cfg(feature = "remote")]
            DispatchDeviceId::Remote => Self::Remote(RemoteDevice::from_id(device_id)),
            _ => unreachable!("No backend feature enabled."),
        }
    }

    fn to_id(&self) -> DeviceId {
        let mut device_id = match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(device) => device.to_id(),
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => device.to_id(),
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.to_id(),
            #[cfg(feature = "rocm")]
            Self::Rocm(device) => device.to_id(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.to_id(),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(device) => device.to_id(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(device) => device.to_id(),
            #[cfg(any(feature = "flex", default_backend))]
            Self::Flex(device) => device.to_id(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(device) => device.to_id(),
            #[cfg(feature = "tch")]
            Self::LibTorch(device) => device.to_id(),
            #[cfg(feature = "remote")]
            Self::Remote(device) => device.to_id(),
            #[cfg(feature = "autodiff")]
            Self::Autodiff(device) => device.inner.to_id(),
        };
        device_id.type_id = self.encode_type_id(device_id.type_id);
        device_id
    }
}

#[cfg(feature = "cpu")]
impl From<CpuDevice> for DispatchDevice {
    fn from(device: CpuDevice) -> Self {
        DispatchDevice::Cpu(device)
    }
}

#[cfg(feature = "cuda")]
impl From<CudaDevice> for DispatchDevice {
    fn from(device: CudaDevice) -> Self {
        DispatchDevice::Cuda(device)
    }
}

#[cfg(feature = "rocm")]
impl From<RocmDevice> for DispatchDevice {
    fn from(device: RocmDevice) -> Self {
        DispatchDevice::Rocm(device)
    }
}

// A bare `WgpuDevice` maps to the auto-compiler [`DispatchDevice::Wgpu`] variant. To target a
// specific wgpu specialization (Metal, Vulkan, WebGpu) construct the variant explicitly.
#[cfg(all(
    feature = "wgpu",
    not(any(feature = "metal", feature = "vulkan", feature = "webgpu"))
))]
impl From<WgpuDevice> for DispatchDevice {
    fn from(device: WgpuDevice) -> Self {
        DispatchDevice::Wgpu(device)
    }
}

#[cfg(all(feature = "metal", not(any(feature = "vulkan", feature = "webgpu"))))]
impl From<WgpuDevice> for DispatchDevice {
    fn from(device: WgpuDevice) -> Self {
        DispatchDevice::Metal(device)
    }
}

#[cfg(all(feature = "vulkan", not(any(feature = "metal", feature = "webgpu"))))]
impl From<WgpuDevice> for DispatchDevice {
    fn from(device: WgpuDevice) -> Self {
        DispatchDevice::Vulkan(device)
    }
}

#[cfg(all(feature = "webgpu", not(any(feature = "metal", feature = "vulkan"))))]
impl From<WgpuDevice> for DispatchDevice {
    fn from(device: WgpuDevice) -> Self {
        DispatchDevice::WebGpu(device)
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
