use alloc::string::String;
use core::marker::PhantomData;

use burn_backend::{Backend, DType, DTypeUsage, DTypeUsageSet, DeviceId, DeviceOps};
use burn_ir::{BackendIr, HandleKind, TensorHandle};
use burn_std::device::Device;
use burn_std::rand::{SeedableRng, StdRng};
use burn_std::stub::Mutex;

use crate::qtensor::FlexQTensor;
use crate::tensor::FlexTensor;

/// Type alias for the RNG used by Flex.
pub type FlexRng = StdRng;

/// Global seed storage for reproducible random number generation.
/// Uses Mutex for thread-safe RNG state management.
pub(crate) static SEED: Mutex<Option<FlexRng>> = Mutex::new(None);

/// Get a random number generator.
/// If a seed was set, clones and returns the seeded RNG.
/// Otherwise, creates a new RNG with OS entropy (std) or constant seed (no_std).
pub(crate) fn get_seeded_rng() -> FlexRng {
    burn_std::rand::get_seeded_rng()
}

/// CPU device for the Flex backend.
///
/// Unit struct since there's only one CPU device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FlexDevice;

impl Device for FlexDevice {
    fn to_id(&self) -> DeviceId {
        DeviceId::new(0, 0)
    }

    fn from_id(_id: DeviceId) -> Self {
        Self
    }
}

impl DeviceOps for FlexDevice {}

impl core::fmt::Display for FlexDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Flex")
    }
}

/// The Flex backend - a fast, portable CPU backend for Burn.
///
/// The `E` and `I` type parameters exist purely to match the shape of other Burn
/// backends (e.g. `NdArray<E, I, Q>`) so `Flex` slots into `burn-dispatch`'s
/// generic dispatch macros. The body of `Flex` uses runtime `DType` dispatch, so
/// both parameters are phantom: the `Backend` impl is only provided for the
/// defaults (`Flex<f32, i32>`) and uses `f32`/`i32` as the associated elem types
/// regardless. To keep source-level compatibility, write `Flex` (which resolves
/// to `Flex<f32, i32>` via the defaults); instantiating with other parameters
/// will not implement `Backend`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Flex<E = f32, I = i32> {
    _e: PhantomData<E>,
    _i: PhantomData<I>,
}

impl Backend for Flex {
    type Device = FlexDevice;

    type FloatTensorPrimitive = FlexTensor;
    /// Default float element type. Determines the dtype for `.float()` conversions and
    /// `Tensor::from_data` when no explicit dtype is provided.
    /// Prefer explicit dtypes via `(&device, DType::F32)`.
    type FloatElem = f32;

    type IntTensorPrimitive = FlexTensor;
    /// Default int element type. Determines the dtype for `.int()` conversions and
    /// `Tensor::from_data` when no explicit dtype is provided.
    /// Set to i32 to match burn's ecosystem default (test suite, record settings, burn-remote).
    /// Prefer explicit dtypes via `(&device, DType::I32)`.
    type IntElem = i32;

    type BoolTensorPrimitive = FlexTensor;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = FlexQTensor;

    fn name(_device: &Self::Device) -> String {
        "flex".into()
    }

    fn seed(_device: &Self::Device, seed: u64) {
        let rng = FlexRng::seed_from_u64(seed);
        let mut seed_lock = SEED.lock().unwrap();
        *seed_lock = Some(rng);
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }

    fn dtype_usage(_device: &Self::Device, dtype: DType) -> DTypeUsageSet {
        match dtype {
            // Full support for standard types
            DType::F64 | DType::F32 | DType::F16 | DType::BF16 => {
                DTypeUsage::Storage | DTypeUsage::Arithmetic
            }
            DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                DTypeUsage::Storage | DTypeUsage::Arithmetic
            }
            DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                DTypeUsage::Storage | DTypeUsage::Arithmetic
            }
            DType::Bool(_) => DTypeUsage::Storage | DTypeUsage::Arithmetic,
            // Quantized types: storage only for now
            DType::QFloat(_) => DTypeUsage::Storage.into(),
            _ => DTypeUsageSet::empty(),
        }
    }
}

impl BackendIr for Flex {
    type Handle = HandleKind<Self>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FlexTensor {
        match handle.handle {
            HandleKind::Float(t) => t,
            _ => panic!("Expected float handle, got {}", handle.handle.name()),
        }
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> FlexTensor {
        match handle.handle {
            HandleKind::Int(t) => t,
            _ => panic!("Expected int handle, got {}", handle.handle.name()),
        }
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> FlexTensor {
        match handle.handle {
            HandleKind::Bool(t) => t,
            _ => panic!("Expected bool handle, got {}", handle.handle.name()),
        }
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> FlexQTensor {
        match handle.handle {
            HandleKind::Quantized(t) => t,
            _ => panic!("Expected quantized handle, got {}", handle.handle.name()),
        }
    }

    fn float_tensor_handle(tensor: FlexTensor) -> Self::Handle {
        HandleKind::Float(tensor)
    }

    fn int_tensor_handle(tensor: FlexTensor) -> Self::Handle {
        HandleKind::Int(tensor)
    }

    fn bool_tensor_handle(tensor: FlexTensor) -> Self::Handle {
        HandleKind::Bool(tensor)
    }

    fn quantized_tensor_handle(tensor: FlexQTensor) -> Self::Handle {
        HandleKind::Quantized(tensor)
    }
}

// Ops traits are implemented in the ops module
