use crate::rand::NdArrayRng;
use crate::{NdArrayQTensor, NdArrayTensor};
use crate::{
    SharedArray,
    element::{FloatNdArrayElement, IntNdArrayElement, QuantElement},
};
use alloc::string::String;
use burn_ir::{BackendIr, HandleKind, TensorHandle};
use burn_std::stub::Mutex;
use burn_std::{DType, QuantLevel, QuantMode, QuantScheme, QuantStore, QuantValue};
use burn_tensor::backend::{Backend, DeviceId, DeviceOps};
use burn_tensor::ops::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use core::marker::PhantomData;
use rand::SeedableRng;

pub(crate) static SEED: Mutex<Option<NdArrayRng>> = Mutex::new(None);

/// The device type for the ndarray backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NdArrayDevice {
    /// The CPU device.
    #[default]
    Cpu,
}

impl DeviceOps for NdArrayDevice {}

impl burn_std::device::Device for NdArrayDevice {
    fn from_id(_device_id: DeviceId) -> Self {
        Self::Cpu
    }

    fn to_id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: 0,
        }
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }
}

/// Tensor backend that uses the [ndarray](ndarray) crate for executing tensor operations.
///
/// This backend is compatible with CPUs and can be compiled for almost any platform, including
/// `wasm`, `arm`, and `x86`.
#[derive(Clone, Copy, Default, Debug)]
pub struct NdArray<E = f32, I = i64, Q = i8>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    _e: PhantomData<E>,
    _i: PhantomData<I>,
    _q: PhantomData<Q>,
}

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> Backend for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    type Device = NdArrayDevice;

    type FloatTensorPrimitive = NdArrayTensor;
    type FloatElem = E;

    type IntTensorPrimitive = NdArrayTensor;
    type IntElem = I;

    type BoolTensorPrimitive = NdArrayTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = NdArrayQTensor;

    fn ad_enabled() -> bool {
        false
    }

    fn name(_device: &Self::Device) -> String {
        String::from("ndarray")
    }

    fn seed(_device: &Self::Device, seed: u64) {
        let rng = NdArrayRng::seed_from_u64(seed);
        let mut seed = SEED.lock().unwrap();
        *seed = Some(rng);
    }

    fn supports_dtype(_device: &Self::Device, dtype: DType) -> bool {
        match dtype {
            DType::F64
            | DType::F32
            | DType::Flex32
            | DType::I64
            | DType::I32
            | DType::I16
            | DType::I8
            | DType::U64
            | DType::U32
            | DType::U16
            | DType::U8
            | DType::Bool => true,
            DType::F16 | DType::BF16 => false,
            DType::QFloat(scheme) => {
                match scheme {
                    QuantScheme {
                        level: QuantLevel::Tensor | QuantLevel::Block(_),
                        mode: QuantMode::Symmetric,
                        #[cfg(not(feature = "export_tests"))]
                            value: QuantValue::Q8F | QuantValue::Q8S,
                        // For tests, "native" sub-byte quant serves as a reference for value equality.
                        // Values are stored as i8 regardless.
                        #[cfg(feature = "export_tests")]
                            value:
                            QuantValue::Q8F
                            | QuantValue::Q8S
                            | QuantValue::Q4F
                            | QuantValue::Q4S
                            | QuantValue::Q2F
                            | QuantValue::Q2S,
                        store: QuantStore::Native,
                        ..
                    } => true,
                    _scheme => false,
                }
            }
        }
    }
}

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> BackendIr for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    type Handle = HandleKind<Self>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        match handle.handle {
            HandleKind::Float(handle) => handle,
            _ => panic!("Expected float handle, got {}", handle.handle.name()),
        }
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        match handle.handle {
            HandleKind::Int(handle) => handle,
            _ => panic!("Expected int handle, got {}", handle.handle.name()),
        }
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        match handle.handle {
            HandleKind::Bool(handle) => handle,
            _ => panic!("Expected bool handle, got {}", handle.handle.name()),
        }
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        match handle.handle {
            HandleKind::Quantized(handle) => handle,
            _ => panic!("Expected quantized handle, got {}", handle.handle.name()),
        }
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        HandleKind::Float(tensor)
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        HandleKind::Int(tensor)
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        HandleKind::Bool(tensor)
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        HandleKind::Quantized(tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_tensor::backend::QTensorPrimitive;

    #[test]
    fn should_support_dtypes() {
        type B = NdArray<f32>;
        let device = Default::default();

        assert!(B::supports_dtype(&device, DType::F64));
        assert!(B::supports_dtype(&device, DType::F32));
        assert!(B::supports_dtype(&device, DType::Flex32));
        assert!(B::supports_dtype(&device, DType::I64));
        assert!(B::supports_dtype(&device, DType::I32));
        assert!(B::supports_dtype(&device, DType::I16));
        assert!(B::supports_dtype(&device, DType::I8));
        assert!(B::supports_dtype(&device, DType::U64));
        assert!(B::supports_dtype(&device, DType::U32));
        assert!(B::supports_dtype(&device, DType::U16));
        assert!(B::supports_dtype(&device, DType::U8));
        assert!(B::supports_dtype(&device, DType::Bool));
        assert!(B::supports_dtype(
            &device,
            DType::QFloat(NdArrayQTensor::default_scheme())
        ));

        assert!(!B::supports_dtype(&device, DType::F16));
        assert!(!B::supports_dtype(&device, DType::BF16));
        // QuantStore::U32 not supported
        assert!(!B::supports_dtype(
            &device,
            DType::QFloat(QuantScheme::default())
        ));
    }
}
