use burn_backend::{DeviceId, DeviceOps};

use crate::backends::*;

/// Represents a device for the [`Dispatch`](crate::Dispatch).
///
/// Each variant corresponds to a backend that the [`Dispatch`](crate::Dispatch) can dispatch operations to.
///
/// # Example
///
/// ```ignore
/// use burn::Device;
///
/// #[cfg(feature = "cpu")]
/// let cpu_device = Device::Cpu(Default::default());
///
/// #[cfg(feature = "cuda")]
/// let cuda_device = Device::Cuda(Default::default());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
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

    /// The [WebGPU backend](WebGpu) device (via WGPU runtime).
    #[cfg(feature = "webgpu")]
    WebGpu(WgpuDevice),

    /// The [NdArray backend](NdArray) device (CPU-only).
    #[cfg(feature = "ndarray")]
    NdArray(NdArrayDevice),

    /// The [LibTorch backend](LibTorch) device.
    #[cfg(feature = "tch")]
    LibTorch(LibTorchDevice),
}

impl Default for Device {
    #[allow(unreachable_code)]
    fn default() -> Self {
        // TODO: which priority?

        #[cfg(feature = "cpu")]
        return Self::Cpu(CpuDevice::default());

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

        #[cfg(feature = "ndarray")]
        return Self::NdArray(NdArrayDevice::default());

        #[cfg(feature = "tch")]
        return Self::LibTorch(LibTorchDevice::default());
    }
}

/// Base multiplier to avoid type_id clashes between backends.
/// Limits the number of device types per backend, but this is a sensible limit.
const TYPE_ID_BASE: u16 = 10;

impl Device {
    /// Returns a unique number per variant to encode into type_id.
    fn backend_id(&self) -> BackendId {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => BackendId::Cpu,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => BackendId::Cuda,
            #[cfg(feature = "metal")]
            Self::Metal(_) => BackendId::Metal,
            #[cfg(feature = "rocm")]
            Self::Rocm(_) => BackendId::Rocm,
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => BackendId::Vulkan,
            #[cfg(feature = "webgpu")]
            Self::WebGpu(_) => BackendId::WebGpu,
            #[cfg(feature = "ndarray")]
            Self::NdArray(_) => BackendId::NdArray,
            #[cfg(feature = "tch")]
            Self::LibTorch(_) => BackendId::LibTorch,
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
            BackendId::try_from(variant).expect("Unknown Device variant"),
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
    #[cfg(feature = "metal")]
    Metal = 2,
    #[cfg(feature = "rocm")]
    Rocm = 3,
    #[cfg(feature = "vulkan")]
    Vulkan = 4,
    #[cfg(feature = "webgpu")]
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
            #[cfg(feature = "metal")]
            2 => Ok(Self::Metal),
            #[cfg(feature = "rocm")]
            3 => Ok(Self::Rocm),
            #[cfg(feature = "vulkan")]
            4 => Ok(Self::Vulkan),
            #[cfg(feature = "webgpu")]
            5 => Ok(Self::WebGpu),
            #[cfg(feature = "ndarray")]
            6 => Ok(Self::NdArray),
            #[cfg(feature = "tch")]
            7 => Ok(Self::LibTorch),
            _ => Err(()),
        }
    }
}

impl DeviceOps for Device {}

impl burn_std::device::Device for Device {
    fn from_id(mut device_id: DeviceId) -> Self {
        let (dispatch_id, backend_type_id) = Self::decode_type_id(device_id.type_id);
        device_id.type_id = backend_type_id;

        match dispatch_id {
            #[cfg(feature = "cpu")]
            BackendId::Cpu => Self::Cpu(CpuDevice::from_id(device_id)),
            #[cfg(feature = "cuda")]
            BackendId::Cuda => Self::Cuda(CudaDevice::from_id(device_id)),
            #[cfg(feature = "metal")]
            BackendId::Metal => Self::Metal(WgpuDevice::from_id(device_id)),
            #[cfg(feature = "rocm")]
            BackendId::Rocm => Self::Rocm(RocmDevice::from_id(device_id)),
            #[cfg(feature = "vulkan")]
            BackendId::Vulkan => Self::Vulkan(WgpuDevice::from_id(device_id)),
            #[cfg(feature = "webgpu")]
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
            #[cfg(feature = "metal")]
            Self::Metal(device) => device.to_id(),
            #[cfg(feature = "rocm")]
            Self::Rocm(device) => device.to_id(),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(device) => device.to_id(),
            #[cfg(feature = "webgpu")]
            Self::WebGpu(device) => device.to_id(),
            #[cfg(feature = "ndarray")]
            Self::NdArray(device) => device.to_id(),
            #[cfg(feature = "tch")]
            Self::LibTorch(device) => device.to_id(),
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
            #[cfg(feature = "metal")]
            BackendId::Metal => WgpuDevice::device_count(backend_type_id),
            #[cfg(feature = "rocm")]
            BackendId::Rocm => RocmDevice::device_count(backend_type_id),
            #[cfg(feature = "vulkan")]
            BackendId::Vulkan => WgpuDevice::device_count(backend_type_id),
            #[cfg(feature = "webgpu")]
            BackendId::WebGpu => WgpuDevice::device_count(backend_type_id),
            #[cfg(feature = "ndarray")]
            BackendId::NdArray => NdArrayDevice::device_count(backend_type_id),
            #[cfg(feature = "tch")]
            BackendId::LibTorch => LibTorchDevice::device_count(backend_type_id),
        }
    }
}

// #[cfg(feature = "cpu")]
// impl From<CpuDevice> for Device {
//     fn from(device: CpuDevice) -> Self {
//         Device::Cpu(device)
//     }
// }

// #[cfg(feature = "cuda")]
// impl From<CudaDevice> for Device {
//     fn from(device: CudaDevice) -> Self {
//         Device::Cuda(device)
//     }
// }

// #[cfg(feature = "metal")]
// impl From<WgpuDevice> for Device {
//     fn from(device: WgpuDevice) -> Self {
//         Device::Metal(device)
//     }
// }

// #[cfg(feature = "rocm")]
// impl From<RocmDevice> for Device {
//     fn from(device: RocmDevice) -> Self {
//         Device::Rocm(device)
//     }
// }

// #[cfg(feature = "vulkan")]
// impl From<WgpuDevice> for Device {
//     fn from(device: WgpuDevice) -> Self {
//         Device::Vulkan(device)
//     }
// }

// #[cfg(feature = "webgpu")]
// impl From<WgpuDevice> for Device {
//     fn from(device: WgpuDevice) -> Self {
//         Device::WebGpu(device)
//     }
// }

// #[cfg(feature = "ndarray")]
// impl From<NdArrayDevice> for Device {
//     fn from(device: NdArrayDevice) -> Self {
//         Device::NdArray(device)
//     }
// }

// #[cfg(feature = "tch")]
// impl From<LibTorchDevice> for Device {
//     fn from(device: LibTorchDevice) -> Self {
//         Device::LibTorch(device)
//     }
// }

// #[cfg(feature = "tch")]
// impl From<LibTorchDevice> for Device {
//     fn from(device: LibTorchDevice) -> Self {
//         Device::LibTorch(device)
//     }
// }
