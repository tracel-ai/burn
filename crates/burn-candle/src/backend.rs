use std::marker::PhantomData;

use burn_tensor::{
    backend::{Backend, DeviceId, DeviceOps},
    quantization::{QTensorPrimitive, QuantizationStrategy},
    Device,
};
use candle_core::{backend::BackendDevice, DeviceLocation};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    CandleQTensor, CandleTensor,
};

/// Tensor backend that uses the [candle](candle_core) crate for executing tensor operations.
///
/// It is compatible with a wide range of hardware configurations, including CPUs and GPUs
/// that support CUDA or Metal. Additionally, the backend can be compiled to `wasm` when using the CPU.
#[derive(Clone, Default, Debug)]
pub struct Candle<F = f32, I = i64>
where
    F: FloatCandleElement,
    I: IntCandleElement,
{
    _float: PhantomData<F>,
    _int: PhantomData<I>,
}

/// The device type for the candle backend.
#[derive(Clone, Debug, PartialEq, Eq)]
/// The device struct when using the `candle` backend.
///
/// To create a Cuda or Metal device from the index, use the associated methods to create the variant:
/// ```no_run
/// use burn_candle::CandleDevice;
///
/// // Create a Cuda device from its index
/// let device = CandleDevice::cuda(0);
/// // Create a Metal device from its index
/// let device = CandleDevice::metal(0);
/// ```
pub enum CandleDevice {
    /// CPU device.
    Cpu,

    /// Cuda device with the given index. The index is the index of the Cuda device in the list of
    /// all Cuda devices found on the system.
    Cuda(CudaDevice),

    /// Metal device with the given index. The index is the index of the Metal device in the list of
    /// all Metal devices found on the system.
    Metal(MetalDevice),
}

impl CandleDevice {
    /// Create a Cuda device with the given index.
    /// The index is the index of the Cuda device in the list of all Cuda devices found on the system.
    pub fn cuda(index: usize) -> Self {
        CandleDevice::Cuda(CudaDevice {
            device: candle_core::CudaDevice::new(index).unwrap(),
            index,
        })
    }

    /// Create a Metal device with the given index.
    /// The index is the index of the Metal device in the list of all Metal devices found on the system.
    pub fn metal(index: usize) -> Self {
        CandleDevice::Metal(MetalDevice {
            device: candle_core::MetalDevice::new(index).unwrap(),
            index,
        })
    }
}

#[derive(Clone, Debug)]
/// A Cuda device for the `candle` backend.
pub struct CudaDevice {
    pub(crate) device: candle_core::CudaDevice,
    /// The index of the Cuda device in the list of all devices on the system.
    pub index: usize,
}

impl PartialEq for CudaDevice {
    fn eq(&self, other: &Self) -> bool {
        self.device.same_device(&other.device) && self.index == other.index
    }
}

impl Eq for CudaDevice {}

#[derive(Clone, Debug)]
/// A Metal device for the `candle` backend.
pub struct MetalDevice {
    pub(crate) device: candle_core::MetalDevice,
    /// The index of the Metal device in the list of all devices on the system.
    pub index: usize,
}

impl PartialEq for MetalDevice {
    fn eq(&self, other: &Self) -> bool {
        self.device.same_device(&other.device) && self.index == other.index
    }
}

impl Eq for MetalDevice {}

impl From<CandleDevice> for candle_core::Device {
    fn from(device: CandleDevice) -> Self {
        match device {
            CandleDevice::Cpu => candle_core::Device::Cpu,
            CandleDevice::Cuda(device) => candle_core::Device::Cuda(device.device),
            CandleDevice::Metal(device) => candle_core::Device::Metal(device.device),
        }
    }
}

impl From<candle_core::Device> for CandleDevice {
    fn from(device: candle_core::Device) -> Self {
        match device.location() {
            DeviceLocation::Cpu => CandleDevice::Cpu,
            DeviceLocation::Cuda { gpu_id } => {
                if let candle_core::Device::Cuda(device) = device {
                    CandleDevice::Cuda(CudaDevice {
                        device,
                        index: gpu_id,
                    })
                } else {
                    panic!("Expected CUDA device.");
                }
            }
            DeviceLocation::Metal { gpu_id } => {
                if let candle_core::Device::Metal(device) = device {
                    CandleDevice::Metal(MetalDevice {
                        device,
                        index: gpu_id,
                    })
                } else {
                    panic!("Expected Metal device.");
                }
            }
        }
    }
}

impl DeviceOps for CandleDevice {
    fn id(&self) -> burn_tensor::backend::DeviceId {
        match self {
            CandleDevice::Cpu => DeviceId::new(0, 0),
            CandleDevice::Cuda(device) => DeviceId::new(1, device.index as u32),
            CandleDevice::Metal(device) => DeviceId::new(2, device.index as u32),
        }
    }
}

impl Default for CandleDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

impl<F: FloatCandleElement, I: IntCandleElement> Backend for Candle<F, I> {
    type Device = CandleDevice;

    type FloatTensorPrimitive = CandleTensor;
    type FloatElem = F;

    type IntTensorPrimitive = CandleTensor;
    type IntElem = I;

    type BoolTensorPrimitive = CandleTensor;
    type BoolElem = u8;

    type QuantizedTensorPrimitive = CandleQTensor;
    type QuantizedEncoding = u8;

    fn ad_enabled() -> bool {
        false
    }

    fn name() -> String {
        "candle".to_string()
    }

    fn seed(seed: u64) {
        // TODO submit an issue at Candle
        panic!("Manual seed not supported by Candle. ")
    }

    fn sync(device: &Device<Self>) {
        let device: candle_core::Device = (device.clone()).into();

        match device {
            candle_core::Device::Cpu => (),
            candle_core::Device::Cuda(device) => {
                #[cfg(feature = "cuda")]
                device.synchronize().unwrap();
            }
            candle_core::Device::Metal(device) => {
                // For some reason, device.wait_until_completed() does not seem to work,
                // and neither does writing and reading a value with into_data
                panic!("Device synchronization unavailable with Metal device on Candle backend")
            }
        }
    }
}
