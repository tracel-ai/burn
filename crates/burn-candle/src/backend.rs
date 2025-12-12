use std::marker::PhantomData;

use burn_std::{
    backtrace::BackTrace,
    rand::{SeedableRng, StdRng},
    stub::Mutex,
};
use burn_tensor::{
    Device,
    backend::{Backend, DeviceId, DeviceOps, ExecutionError},
    quantization::QTensorPrimitive,
};
use candle_core::{DeviceLocation, backend::BackendDevice};

use crate::{
    CandleTensor,
    element::{CandleElement, FloatCandleElement, IntCandleElement},
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

// Seed for CPU device
pub(crate) static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

pub(crate) fn get_seeded_rng() -> StdRng {
    let mut seed = SEED.lock().unwrap();
    match seed.as_ref() {
        Some(rng_seeded) => rng_seeded.clone(),
        None => burn_std::rand::get_seeded_rng(),
    }
}

pub(crate) fn set_seeded_rng(rng_seeded: StdRng) {
    let mut seed = SEED.lock().unwrap();
    *seed = Some(rng_seeded);
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
#[derive(Default)]
pub enum CandleDevice {
    /// CPU device.
    #[default]
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

    pub(crate) fn set_seed(&self, seed: u64) {
        match self {
            CandleDevice::Cpu => {
                // candle_core::cpu_backend::CpuDevice.set_seed(seed).unwrap();
                // Candle does not support seeding the CPU rng so we use a global seed
                let rng = StdRng::seed_from_u64(seed);
                set_seeded_rng(rng);
            }
            CandleDevice::Cuda(cuda_device) => cuda_device.device.set_seed(seed).unwrap(),
            CandleDevice::Metal(metal_device) => metal_device.device.set_seed(seed).unwrap(),
        }
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

impl burn_std::device::Device for CandleDevice {
    fn to_id(&self) -> burn_tensor::backend::DeviceId {
        match self {
            CandleDevice::Cuda(device) => DeviceId::new(0, device.index as u32),
            CandleDevice::Metal(device) => DeviceId::new(1, device.index as u32),
            CandleDevice::Cpu => DeviceId::new(2, 0),
        }
    }

    fn from_id(device_id: DeviceId) -> Self {
        match device_id.type_id {
            0 => CandleDevice::cuda(device_id.index_id as usize),
            1 => CandleDevice::metal(device_id.index_id as usize),
            _ => CandleDevice::Cpu,
        }
    }

    fn device_count(type_id: u16) -> usize {
        // TODO: Fix that
        1
    }
}
impl DeviceOps for CandleDevice {}

impl<F: FloatCandleElement, I: IntCandleElement> Backend for Candle<F, I> {
    type Device = CandleDevice;

    type FloatTensorPrimitive = CandleTensor;
    type FloatElem = F;

    type IntTensorPrimitive = CandleTensor;
    type IntElem = I;

    type BoolTensorPrimitive = CandleTensor;
    type BoolElem = u8;

    type QuantizedTensorPrimitive = CandleTensor;

    fn ad_enabled() -> bool {
        false
    }

    fn name(device: &Self::Device) -> String {
        match device {
            CandleDevice::Cpu => "candle<cpu>",
            CandleDevice::Cuda(..) => "candle<cuda>",
            CandleDevice::Metal(..) => "candle<metal>",
        }
        .to_string()
    }

    fn seed(device: &CandleDevice, seed: u64) {
        device.set_seed(seed);
    }

    fn sync(device: &Device<Self>) -> Result<(), ExecutionError> {
        let device: candle_core::Device = (device.clone()).into();

        match device {
            candle_core::Device::Cpu => (),
            candle_core::Device::Cuda(device) => {
                #[cfg(feature = "cuda")]
                device
                    .synchronize()
                    .map_err(|err| ExecutionError::Generic {
                        reason: format!("Can't sync the cuda device: {err}"),
                        backtrace: BackTrace::capture(),
                    })?;
            }
            candle_core::Device::Metal(device) => {
                // For some reason, device.wait_until_completed() does not seem to work,
                // and neither does writing and reading a value with into_data
                return Err(ExecutionError::Generic {
                    reason:
                        "Device synchronization unavailable with Metal device on Candle backend"
                            .into(),
                    backtrace: BackTrace::capture(),
                });
            }
        }

        Ok(())
    }
}
