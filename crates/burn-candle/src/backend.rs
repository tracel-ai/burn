use std::marker::PhantomData;

use burn_tensor::{
    backend::{Backend, DeviceId, DeviceOps, SyncType},
    Device,
};
use candle_core::DeviceLocation;

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    CandleTensor, PrecisionBridge,
};

/// Tensor backend that uses the [candle](candle_core) crate for executing tensor operations.
///
/// It is compatible with a wide range of hardware configurations, including CPUs and GPUs
/// that support CUDA or Metal. Additionally, the backend can be compiled to `wasm` when using the CPU.
#[derive(Clone, Copy, Default, Debug)]
pub struct Candle<F = f32, I = i64>
where
    F: FloatCandleElement,
    I: IntCandleElement,
{
    _float: PhantomData<F>,
    _int: PhantomData<I>,
}

/// The device type for the candle backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// The device struct when using the `candle` backend.
///
/// Note that you need to provide the device index when using Cuda.
pub enum CandleDevice {
    /// CPU device.
    Cpu,

    /// Cuda device with the given index. The index is the index of the Cuda device in the list of
    /// all Cuda devices found on the system.
    Cuda(usize),

    /// Metal device with the given index. The index is the index of the Metal device in the list of
    /// all Metal devices found on the system.
    Metal(usize),
}

impl From<CandleDevice> for candle_core::Device {
    fn from(device: CandleDevice) -> Self {
        match device {
            CandleDevice::Cpu => candle_core::Device::Cpu,
            CandleDevice::Cuda(ordinal) => candle_core::Device::new_cuda(ordinal).unwrap(),
            CandleDevice::Metal(ordinal) => candle_core::Device::new_metal(ordinal).unwrap(),
        }
    }
}

impl From<candle_core::Device> for CandleDevice {
    fn from(device: candle_core::Device) -> Self {
        match device.location() {
            DeviceLocation::Cpu => CandleDevice::Cpu,
            DeviceLocation::Cuda { gpu_id } => CandleDevice::Cuda(gpu_id),
            DeviceLocation::Metal { gpu_id } => CandleDevice::Metal(gpu_id),
        }
    }
}

impl DeviceOps for CandleDevice {
    fn id(&self) -> burn_tensor::backend::DeviceId {
        match self {
            CandleDevice::Cpu => DeviceId::new(0, 0),
            CandleDevice::Cuda(index) => DeviceId::new(1, *index as u32),
            CandleDevice::Metal(index) => DeviceId::new(2, *index as u32),
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

    type FullPrecisionBridge = PrecisionBridge<f32>;

    type FloatTensorPrimitive<const D: usize> = CandleTensor<Self::FloatElem, D>;
    type FloatElem = F;

    type IntTensorPrimitive<const D: usize> = CandleTensor<Self::IntElem, D>;
    type IntElem = I;

    type BoolTensorPrimitive<const D: usize> = CandleTensor<u8, D>;

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

    fn sync(device: &Device<Self>, sync_type: SyncType) {
        match sync_type {
            SyncType::Wait => {
                let device: candle_core::Device = (*device).into();

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
            SyncType::Flush => (), // Nothhing to flush.
        };
    }
}
