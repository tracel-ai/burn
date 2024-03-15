use std::marker::PhantomData;

use burn_tensor::backend::Backend;
use burn_tensor::{DynData, DynRankData, Element};
use candle_core::{DType, DeviceLocation, WithDType};

use crate::{
    element::{CandleElement, FloatCandleElement, IntCandleElement},
    CandleTensor,
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

impl Default for CandleDevice {
    fn default() -> Self {
        Self::Cpu
    }
}

fn candle_tensor_from_dyn_rank_data<E: Element + WithDType>(
    dyn_rank_data: DynRankData<E>,
    device: &CandleDevice,
) -> candle_core::Tensor {
    candle_core::Tensor::from_slice(&dyn_rank_data.value, dyn_rank_data.shape, &(*device).into())
        .unwrap()
}

fn dyn_rank_data_from_candle_tensor<E: Element + WithDType>(tensor: candle_core::Tensor) -> DynRankData<E> {
    let shape = tensor.shape().clone().into_dims();

    DynRankData::new(
        tensor
            .reshape(&[tensor.elem_count()])
            .unwrap()
            .try_into()
            .unwrap(),
        shape,
    )
}

#[derive(Clone, Debug)]
/// A [CandleTensor] with a dynamic rank and element type.
pub enum DynCandleTensor {
    /// Float tensor variant.
    Float(candle_core::Tensor),
    /// Integer tensor variant.
    Int(candle_core::Tensor),
    /// Boolean tensor variant.
    Bool(candle_core::Tensor),
}

impl DynCandleTensor {
    /// Returns the [candle_core::Tensor] internally used by this enum.
    pub fn into_inner(self) -> candle_core::Tensor {
        match self {
            DynCandleTensor::Float(dyn_tensor) => dyn_tensor,
            DynCandleTensor::Int(dyn_tensor) => dyn_tensor,
            DynCandleTensor::Bool(dyn_tensor) => dyn_tensor,
        }
    }
}

impl<F: FloatCandleElement, I: IntCandleElement> Backend for Candle<F, I> {
    type Device = CandleDevice;

    type FullPrecisionBackend = Candle<Self::FullPrecisionElem, Self::IntElem>;
    type FullPrecisionElem = f32;

    type FloatTensorPrimitive<const D: usize> = CandleTensor<Self::FloatElem, D>;
    type FloatElem = F;

    type IntTensorPrimitive<const D: usize> = CandleTensor<Self::IntElem, D>;
    type IntElem = I;

    type BoolTensorPrimitive<const D: usize> = CandleTensor<u8, D>;
    type DynTensorPrimitive = DynCandleTensor;

    fn name() -> String {
        "candle".to_string()
    }

    fn seed(seed: u64) {
        // TODO submit an issue at Candle
        panic!("Manual seed not supported by Candle. ")
    }

    fn dyn_from_data(
        data: DynData<Self::FullPrecisionElem, Self::IntElem>,
        device: &Self::Device,
    ) -> Self::DynTensorPrimitive {
        match data {
            DynData::Float(float_data) => {
                DynCandleTensor::Float(candle_tensor_from_dyn_rank_data(float_data.convert::<Self::FloatElem>(), device))
            }
            DynData::Int(int_data) => DynCandleTensor::Int(candle_tensor_from_dyn_rank_data(int_data, device)),
            DynData::Bool(bool_data) => DynCandleTensor::Bool(candle_tensor_from_dyn_rank_data(
                DynRankData::new(
                    bool_data.value.into_iter().map(|boolean| boolean as u8).collect(),
                    bool_data.shape,
                ),
                device,
            )),
        }
    }

    fn dyn_into_data(
        dyn_tensor: Self::DynTensorPrimitive,
    ) -> DynData<Self::FullPrecisionElem, Self::IntElem> {
        match dyn_tensor {
            DynCandleTensor::Float(dyn_tensor) => DynData::Float(dyn_rank_data_from_candle_tensor(dyn_tensor)),
            DynCandleTensor::Int(dyn_tensor) => DynData::Int(dyn_rank_data_from_candle_tensor(dyn_tensor)),
            DynCandleTensor::Bool(dyn_tensor) => {
                let dyn_rank_data = dyn_rank_data_from_candle_tensor::<u8>(dyn_tensor);

                DynData::Bool(DynRankData::new(
                    dyn_rank_data.value.into_iter().map(|boolean| boolean != 0).collect(),
                    dyn_rank_data.shape,
                ))
            }
        }
    }
}
