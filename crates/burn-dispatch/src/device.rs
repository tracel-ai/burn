use burn_backend::{DeviceId, DeviceOps};

use crate::backends::*;

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
    #[cfg(wgpu_metal)]
    Metal(WgpuDevice),

    /// The [ROCm backend](Rocm) device.
    #[cfg(feature = "rocm")]
    Rocm(RocmDevice),

    /// The [Vulkan backend](Vulkan) device.
    #[cfg(wgpu_vulkan)]
    Vulkan(WgpuDevice),

    /// The [WebGPU backend](WebGpu) device (via WGPU runtime).
    #[cfg(wgpu_webgpu)]
    WebGpu(WgpuDevice),

    /// The [NdArray backend](NdArray) device (CPU-only).
    #[cfg(feature = "ndarray")]
    NdArray(NdArrayDevice),

    /// The [LibTorch backend](LibTorch) device.
    #[cfg(feature = "tch")]
    LibTorch(LibTorchDevice),

    /// The [autodiff enabled backend](Autodiff) device.
    #[cfg(feature = "autodiff")]
    Autodiff(AutodiffDevice),
}

// This tuple struct mainly restricts users from creating Autodiff(Autodiff) devices.
/// A wrapper that enables automatic differentiation for a [`DispatchDevice`].
///
/// Use [`DispatchDevice::new_autodiff`] to construct this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutodiffDevice(pub(crate) Box<DispatchDevice>);

// Useful for match in dispatch macros
impl core::ops::Deref for AutodiffDevice {
    type Target = DispatchDevice;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::fmt::Debug for DispatchDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(device) => f.debug_tuple("Cpu").field(device).finish(),
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => f.debug_tuple("Cuda").field(device).finish(),
            #[cfg(wgpu_metal)]
            Self::Metal(device) => f.debug_tuple("Metal").field(device).finish(),
            #[cfg(feature = "rocm")]
            Self::Rocm(device) => f.debug_tuple("Rocm").field(device).finish(),
            #[cfg(wgpu_vulkan)]
            Self::Vulkan(device) => f.debug_tuple("Vulkan").field(device).finish(),
            #[cfg(wgpu_webgpu)]
            Self::WebGpu(device) => f.debug_tuple("WebGpu").field(device).finish(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(device) => f.debug_tuple("NdArray").field(device).finish(),
            #[cfg(feature = "tch")]
            Self::LibTorch(device) => f.debug_tuple("LibTorch").field(device).finish(),
            #[cfg(feature = "autodiff")]
            // Format without `AutodiffDevice` wrapper
            Self::Autodiff(device) => f.debug_tuple("Autodiff").field(&device.0).finish(),
        }
    }
}

impl Default for DispatchDevice {
    #[allow(unreachable_code)]
    fn default() -> Self {
        // TODO: which priority?

        #[cfg(feature = "cpu")]
        return Self::Cpu(CpuDevice);

        #[cfg(feature = "cuda")]
        return Self::Cuda(CudaDevice::default());

        #[cfg(wgpu_metal)]
        return Self::Metal(burn_wgpu::WgpuDevice::default());

        #[cfg(feature = "rocm")]
        return Self::Rocm(RocmDevice::default());

        #[cfg(wgpu_vulkan)]
        return Self::Vulkan(burn_wgpu::WgpuDevice::default());

        #[cfg(wgpu_webgpu)]
        return Self::WebGpu(burn_wgpu::WgpuDevice::default());

        #[cfg(feature = "ndarray")]
        return Self::NdArray(NdArrayDevice::default());

        #[cfg(feature = "tch")]
        return Self::LibTorch(LibTorchDevice::default());
    }
}

impl PartialEq for DispatchDevice {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            // If both are Autodiff, compare the inner devices
            #[cfg(feature = "autodiff")]
            (DispatchDevice::Autodiff(a), DispatchDevice::Autodiff(b)) => a == b,
            // If one is Autodiff, compare it to the raw device
            #[cfg(feature = "autodiff")]
            (DispatchDevice::Autodiff(a), b) => a.0.as_ref() == b,
            #[cfg(feature = "autodiff")]
            (a, DispatchDevice::Autodiff(b)) => a == b.0.as_ref(),
            #[cfg(feature = "cpu")]
            (Self::Cpu(a), Self::Cpu(b)) => a == b,
            #[cfg(feature = "cuda")]
            (Self::Cuda(a), Self::Cuda(b)) => a == b,
            #[cfg(wgpu_metal)]
            (Self::Metal(a), Self::Metal(b)) => a == b,
            #[cfg(feature = "rocm")]
            (Self::Rocm(a), Self::Rocm(b)) => a == b,
            #[cfg(wgpu_vulkan)]
            (Self::Vulkan(a), Self::Vulkan(b)) => a == b,
            #[cfg(wgpu_webgpu)]
            (Self::WebGpu(a), Self::WebGpu(b)) => a == b,
            #[cfg(feature = "ndarray")]
            (Self::NdArray(a), Self::NdArray(b)) => a == b,
            #[cfg(feature = "tch")]
            (Self::LibTorch(a), Self::LibTorch(b)) => a == b,
            #[allow(unreachable_patterns)]
            (_, _) => false,
        }
    }
}

/// Base multiplier to avoid type_id clashes between backends.
/// Limits the number of device types per backend, but this is a sensible limit.
const TYPE_ID_BASE: u16 = 10;

impl DispatchDevice {
    /// Creates a new [`DispatchDevice`] with [automatic differentiation](Autodiff) enabled.
    pub fn new_autodiff(device: impl Into<DispatchDevice>) -> DispatchDevice {
        let device = device.into();
        DispatchDevice::Autodiff(AutodiffDevice(Box::new(device)))
    }

    /// Returns a unique number per variant to encode into type_id.
    fn backend_id(&self) -> BackendId {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => BackendId::Cpu,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => BackendId::Cuda,
            #[cfg(wgpu_metal)]
            Self::Metal(_) => BackendId::Metal,
            #[cfg(feature = "rocm")]
            Self::Rocm(_) => BackendId::Rocm,
            #[cfg(wgpu_vulkan)]
            Self::Vulkan(_) => BackendId::Vulkan,
            #[cfg(wgpu_webgpu)]
            Self::WebGpu(_) => BackendId::WebGpu,
            #[cfg(feature = "ndarray")]
            Self::NdArray(_) => BackendId::NdArray,
            #[cfg(feature = "tch")]
            Self::LibTorch(_) => BackendId::LibTorch,
            #[cfg(feature = "autodiff")]
            Self::Autodiff(device) => device.0.backend_id(),
        }
    }

    /// Encode variant ID and backend type ID into a unique `type_id`.
    fn encode_type_id(&self, backend_type_id: u16) -> u16 {
        u16::from(self.backend_id()) * TYPE_ID_BASE + backend_type_id
    }

    /// Decode an encoded `type_id` into variant ID and backend type ID.
    fn decode_type_id(type_id: u16) -> (BackendId, u16) {
        let variant = type_id / TYPE_ID_BASE;
        let backend_type_id = type_id % TYPE_ID_BASE;
        (
            BackendId::try_from(variant).expect("Unknown DispatchDevice variant"),
            backend_type_id,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
enum BackendId {
    #[cfg(feature = "cpu")]
    Cpu = 0,
    #[cfg(feature = "cuda")]
    Cuda = 1,
    #[cfg(wgpu_metal)]
    Metal = 2,
    #[cfg(feature = "rocm")]
    Rocm = 3,
    #[cfg(wgpu_vulkan)]
    Vulkan = 4,
    #[cfg(wgpu_webgpu)]
    WebGpu = 5,
    #[cfg(feature = "ndarray")]
    NdArray = 6,
    #[cfg(feature = "tch")]
    LibTorch = 7,
}

impl From<BackendId> for u16 {
    fn from(variant: BackendId) -> Self {
        variant as u16
    }
}

impl TryFrom<u16> for BackendId {
    type Error = ();

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            #[cfg(feature = "cpu")]
            0 => Ok(Self::Cpu),
            #[cfg(feature = "cuda")]
            1 => Ok(Self::Cuda),
            #[cfg(wgpu_metal)]
            2 => Ok(Self::Metal),
            #[cfg(feature = "rocm")]
            3 => Ok(Self::Rocm),
            #[cfg(wgpu_vulkan)]
            4 => Ok(Self::Vulkan),
            #[cfg(wgpu_webgpu)]
            5 => Ok(Self::WebGpu),
            #[cfg(feature = "ndarray")]
            6 => Ok(Self::NdArray),
            #[cfg(feature = "tch")]
            7 => Ok(Self::LibTorch),
            _ => Err(()),
        }
    }
}

impl DeviceOps for DispatchDevice {
    fn inner(&self) -> &Self {
        match self {
            #[cfg(feature = "autodiff")]
            DispatchDevice::Autodiff(device) => &device.0,
            device => device,
        }
    }
}

impl burn_std::device::Device for DispatchDevice {
    fn from_id(mut device_id: DeviceId) -> Self {
        let (dispatch_id, backend_type_id) = Self::decode_type_id(device_id.type_id);
        device_id.type_id = backend_type_id;

        match dispatch_id {
            #[cfg(feature = "cpu")]
            BackendId::Cpu => Self::Cpu(CpuDevice::from_id(device_id)),
            #[cfg(feature = "cuda")]
            BackendId::Cuda => Self::Cuda(CudaDevice::from_id(device_id)),
            #[cfg(wgpu_metal)]
            BackendId::Metal => Self::Metal(WgpuDevice::from_id(device_id)),
            #[cfg(feature = "rocm")]
            BackendId::Rocm => Self::Rocm(RocmDevice::from_id(device_id)),
            #[cfg(wgpu_vulkan)]
            BackendId::Vulkan => Self::Vulkan(WgpuDevice::from_id(device_id)),
            #[cfg(wgpu_webgpu)]
            BackendId::WebGpu => Self::WebGpu(WgpuDevice::from_id(device_id)),
            #[cfg(feature = "ndarray")]
            BackendId::NdArray => Self::NdArray(NdArrayDevice::from_id(device_id)),
            #[cfg(feature = "tch")]
            BackendId::LibTorch => Self::LibTorch(LibTorchDevice::from_id(device_id)),
        }
    }

    fn to_id(&self) -> DeviceId {
        let mut device_id = match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(device) => device.to_id(),
            #[cfg(feature = "cuda")]
            Self::Cuda(device) => device.to_id(),
            #[cfg(wgpu_metal)]
            Self::Metal(device) => device.to_id(),
            #[cfg(feature = "rocm")]
            Self::Rocm(device) => device.to_id(),
            #[cfg(wgpu_vulkan)]
            Self::Vulkan(device) => device.to_id(),
            #[cfg(wgpu_webgpu)]
            Self::WebGpu(device) => device.to_id(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(device) => device.to_id(),
            #[cfg(feature = "tch")]
            Self::LibTorch(device) => device.to_id(),
            #[cfg(feature = "autodiff")]
            Self::Autodiff(device) => device.0.to_id(),
        };
        device_id.type_id = self.encode_type_id(device_id.type_id);
        device_id
    }

    fn device_count(type_id: u16) -> usize {
        let (dispatch_id, backend_type_id) = Self::decode_type_id(type_id);
        match dispatch_id {
            #[cfg(feature = "cpu")]
            BackendId::Cpu => CpuDevice::device_count(backend_type_id),
            #[cfg(feature = "cuda")]
            BackendId::Cuda => CudaDevice::device_count(backend_type_id),
            #[cfg(wgpu_metal)]
            BackendId::Metal => WgpuDevice::device_count(backend_type_id),
            #[cfg(feature = "rocm")]
            BackendId::Rocm => RocmDevice::device_count(backend_type_id),
            #[cfg(wgpu_vulkan)]
            BackendId::Vulkan => WgpuDevice::device_count(backend_type_id),
            #[cfg(wgpu_webgpu)]
            BackendId::WebGpu => WgpuDevice::device_count(backend_type_id),
            #[cfg(feature = "ndarray")]
            BackendId::NdArray => NdArrayDevice::device_count(backend_type_id),
            #[cfg(feature = "tch")]
            BackendId::LibTorch => LibTorchDevice::device_count(backend_type_id),
        }
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

#[cfg(wgpu_metal)]
impl From<WgpuDevice> for DispatchDevice {
    fn from(device: WgpuDevice) -> Self {
        DispatchDevice::Metal(device)
    }
}

#[cfg(feature = "rocm")]
impl From<RocmDevice> for DispatchDevice {
    fn from(device: RocmDevice) -> Self {
        DispatchDevice::Rocm(device)
    }
}

#[cfg(wgpu_vulkan)]
impl From<WgpuDevice> for DispatchDevice {
    fn from(device: WgpuDevice) -> Self {
        DispatchDevice::Vulkan(device)
    }
}

#[cfg(wgpu_webgpu)]
impl From<WgpuDevice> for DispatchDevice {
    fn from(device: WgpuDevice) -> Self {
        DispatchDevice::WebGpu(device)
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

#[cfg(feature = "tch")]
impl From<LibTorchDevice> for DispatchDevice {
    fn from(device: LibTorchDevice) -> Self {
        DispatchDevice::LibTorch(device)
    }
}
