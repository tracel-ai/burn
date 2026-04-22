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

/// Fallback RNG when `SEED` is empty (consumed or never set).
///
/// The seeding flow is: `Backend::seed()` stores a `FlexRng` in `SEED`. Random
/// ops (`float_random`, `int_random`) call `SEED.lock().take()`, consuming it for
/// that op and falling back to this function for subsequent calls. This function
/// delegates to burn_std's own entropy source.
pub(crate) fn get_seeded_rng() -> FlexRng {
    burn_std::rand::get_seeded_rng()
}

/// CPU device for the Flex backend.
///
/// Unit struct since there's only one CPU device.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
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
        write!(f, "Cpu")
    }
}

impl core::fmt::Debug for FlexDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(self, f)
    }
}

/// The Flex backend, a fast, portable CPU backend for Burn.
///
/// The `E` and `I` type parameters exist purely to match the shape of other Burn
/// backends (e.g. `NdArray<E, I, Q>`) so `Flex` slots into `burn-dispatch`'s
/// generic dispatch macros. The body of `Flex` uses runtime `DType` dispatch, so
/// both parameters are phantom and unused at runtime.
///
/// # Limitations of the phantom generics
///
/// The `Backend` impl is provided only for the default instantiation
/// `Flex<f32, i32>`. Writing `Flex` (with no arguments) resolves to the default
/// and works exactly as before. Writing `Flex<f64, i64>` or any other non-default
/// combination is a valid Rust type but will not satisfy trait bounds requiring
/// `Backend`, producing errors like:
///
/// ```text
/// the trait bound `Flex<f64, i64>: Backend` is not satisfied
/// ```
///
/// This is a deliberate compromise for the initial migration: making `Flex`
/// generic over element types at the trait-impl level is a follow-up that would
/// require rewriting all `impl FooOps<Flex> for Flex` blocks plus internal
/// `Flex::method()` calls (tracked in
/// [#4762](https://github.com/tracel-ai/burn/issues/4762)). Until then, treat
/// the generic parameters as opaque shape placeholders; real element-type
/// selection happens at runtime via `DType`.
///
/// The bound is locked in by a compile-fail doctest so that if someone later
/// makes the `Backend` impl generic over `E`/`I`, this documentation gets
/// flagged as out of date:
///
/// ```compile_fail
/// use burn_backend::Backend;
/// use burn_flex::Flex;
/// fn requires_backend<B: Backend>() {}
/// requires_backend::<Flex<f64, i64>>();
/// ```
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
            // Bool storage: flex stores bools as 1 byte per element, so Native and
            // U8 are both supported (they share the same layout, only the tag
            // differs). Bool(U32) would require 4-byte-per-element storage
            // throughout the backend and is not yet implemented.
            DType::Bool(burn_std::BoolStore::Native | burn_std::BoolStore::U8) => {
                DTypeUsage::Storage | DTypeUsage::Arithmetic
            }
            DType::Bool(burn_std::BoolStore::U32) => DTypeUsageSet::empty(),
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

#[cfg(test)]
mod tests {
    use burn_backend::{Backend, DType};
    use burn_std::BoolStore;

    use super::*;

    #[test]
    fn supports_bool_native() {
        let device = FlexDevice;
        assert!(Flex::supports_dtype(
            &device,
            DType::Bool(BoolStore::Native)
        ));
    }

    #[test]
    fn supports_bool_u8() {
        let device = FlexDevice;
        assert!(Flex::supports_dtype(&device, DType::Bool(BoolStore::U8)));
    }

    #[test]
    fn does_not_support_bool_u32() {
        let device = FlexDevice;
        assert!(
            !Flex::supports_dtype(&device, DType::Bool(BoolStore::U32)),
            "Bool(U32) should not be supported: flex stores bools as 1 byte per element"
        );
    }

    #[test]
    fn bool_empty_preserves_native_dtype() {
        use burn_backend::ops::BoolTensorOps;
        let shape = burn_std::Shape::from(alloc::vec![3]);
        let t = Flex::bool_empty(shape, &FlexDevice, burn_std::BoolDType::Native);
        assert_eq!(t.dtype(), DType::Bool(BoolStore::Native));
    }

    #[test]
    fn bool_empty_preserves_u8_dtype() {
        use burn_backend::ops::BoolTensorOps;
        let shape = burn_std::Shape::from(alloc::vec![3]);
        let t = Flex::bool_empty(shape, &FlexDevice, burn_std::BoolDType::U8);
        assert_eq!(t.dtype(), DType::Bool(BoolStore::U8));
    }

    #[test]
    fn device_prints_as_cpu() {
        use alloc::format;
        assert_eq!(format!("{:?}", FlexDevice), "Cpu");
        assert_eq!(format!("{}", FlexDevice), "Cpu");
    }

    #[test]
    fn comparison_preserves_out_dtype_native() {
        let lhs = FlexTensor::from_data(burn_backend::TensorData::from([1.0f32, 2.0, 3.0]));
        let rhs = FlexTensor::from_data(burn_backend::TensorData::from([2.0f32, 2.0, 1.0]));
        let result = crate::ops::comparison::greater(lhs, rhs, burn_std::BoolDType::Native);
        assert_eq!(result.dtype(), DType::Bool(BoolStore::Native));
    }

    #[test]
    fn comparison_preserves_out_dtype_u8() {
        let lhs = FlexTensor::from_data(burn_backend::TensorData::from([1.0f32, 2.0, 3.0]));
        let rhs = FlexTensor::from_data(burn_backend::TensorData::from([2.0f32, 2.0, 1.0]));
        let result = crate::ops::comparison::greater(lhs, rhs, burn_std::BoolDType::U8);
        assert_eq!(result.dtype(), DType::Bool(BoolStore::U8));
    }

    #[test]
    #[should_panic(expected = "Bool(U32)")]
    fn comparison_u32_panics() {
        let lhs = FlexTensor::from_data(burn_backend::TensorData::from([1.0f32, 2.0]));
        let rhs = FlexTensor::from_data(burn_backend::TensorData::from([2.0f32, 1.0]));
        let _ = crate::ops::comparison::greater(lhs, rhs, burn_std::BoolDType::U32);
    }

    #[test]
    fn bool_not_preserves_u8_dtype() {
        use burn_backend::ops::BoolTensorOps;
        // Construct a Bool(U8) tensor directly to verify bool_not preserves
        // the dtype tag across the op. from_data would produce Bool(Native),
        // so we use make_bool_tensor to get the U8 tag.
        let t_u8 = crate::ops::comparison::make_bool_tensor(
            alloc::vec![1, 0, 1],
            burn_std::Shape::from(alloc::vec![3]),
            burn_std::BoolDType::U8,
        );
        let result = Flex::bool_not(t_u8);
        assert_eq!(result.dtype(), DType::Bool(BoolStore::U8));
        let data: &[u8] = result.bytes();
        assert_eq!(&data[..3], &[0, 1, 0]);
    }
}
