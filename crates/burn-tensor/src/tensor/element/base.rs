use core::cmp::Ordering;

use crate::{Distribution, cast::ToElement, quantization::QuantScheme};
#[cfg(feature = "cubecl")]
use cubecl::flex32;

use cubecl_quant::scheme::{QuantStore, QuantValue};
use half::{bf16, f16};
use rand::RngCore;
use serde::{Deserialize, Serialize};

/// Element trait for tensor.
pub trait Element:
    ToElement
    + ElementRandom
    + ElementConversion
    + ElementComparison
    + ElementLimits
    + bytemuck::CheckedBitPattern
    + bytemuck::NoUninit
    + bytemuck::Zeroable
    + core::fmt::Debug
    + core::fmt::Display
    + Default
    + Send
    + Sync
    + Copy
    + 'static
{
    /// The dtype of the element.
    fn dtype() -> DType;
}

/// Element conversion trait for tensor.
pub trait ElementConversion {
    /// Converts an element to another element.
    ///
    /// # Arguments
    ///
    /// * `elem` - The element to convert.
    ///
    /// # Returns
    ///
    /// The converted element.
    fn from_elem<E: ToElement>(elem: E) -> Self;

    /// Converts and returns the converted element.
    fn elem<E: Element>(self) -> E;
}

/// Element trait for random value of a tensor.
pub trait ElementRandom {
    /// Returns a random value for the given distribution.
    ///
    /// # Arguments
    ///
    /// * `distribution` - The distribution to sample from.
    /// * `rng` - The random number generator.
    ///
    /// # Returns
    ///
    /// The random value.
    fn random<R: RngCore>(distribution: Distribution, rng: &mut R) -> Self;
}

/// Element ordering trait.
pub trait ElementComparison {
    /// Returns and [Ordering] between `self` and `other`.
    fn cmp(&self, other: &Self) -> Ordering;
}

/// Element ordering trait.
pub trait ElementLimits {
    /// The minimum representable value
    const MIN: Self;
    /// The maximum representable value
    const MAX: Self;
}

/// Macro to implement the element trait for a type.
#[macro_export]
macro_rules! make_element {
    (
        ty $type:ident,
        convert $convert:expr,
        random $random:expr,
        cmp $cmp:expr,
        dtype $dtype:expr
    ) => {
        make_element!(ty $type, convert $convert, random $random, cmp $cmp, dtype $dtype, min $type::MIN, max $type::MAX);
    };
    (
        ty $type:ident,
        convert $convert:expr,
        random $random:expr,
        cmp $cmp:expr,
        dtype $dtype:expr,
        min $min:expr,
        max $max:expr
    ) => {
        impl Element for $type {
            #[inline(always)]
            fn dtype() -> $crate::DType {
                $dtype
            }
        }

        impl ElementConversion for $type {
            #[inline(always)]
            fn from_elem<E: ToElement>(elem: E) -> Self {
                #[allow(clippy::redundant_closure_call)]
                $convert(&elem)
            }
            #[inline(always)]
            fn elem<E: Element>(self) -> E {
                E::from_elem(self)
            }
        }

        impl ElementRandom for $type {
            fn random<R: RngCore>(distribution: Distribution, rng: &mut R) -> Self {
                #[allow(clippy::redundant_closure_call)]
                $random(distribution, rng)
            }
        }

        impl ElementComparison for $type {
            fn cmp(&self, other: &Self) -> Ordering {
                let a = self.elem::<$type>();
                let b = other.elem::<$type>();
                #[allow(clippy::redundant_closure_call)]
                $cmp(&a, &b)
            }
        }

        impl ElementLimits for $type {
            const MIN: Self = $min;
            const MAX: Self = $max;
        }
    };
}

make_element!(
    ty f64,
    convert ToElement::to_f64,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &f64, b: &f64| a.total_cmp(b),
    dtype DType::F64
);

make_element!(
    ty f32,
    convert ToElement::to_f32,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &f32, b: &f32| a.total_cmp(b),
    dtype DType::F32
);

make_element!(
    ty i64,
    convert ToElement::to_i64,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &i64, b: &i64| Ord::cmp(a, b),
    dtype DType::I64
);

make_element!(
    ty u64,
    convert ToElement::to_u64,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &u64, b: &u64| Ord::cmp(a, b),
    dtype DType::U64
);

make_element!(
    ty i32,
    convert ToElement::to_i32,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &i32, b: &i32| Ord::cmp(a, b),
    dtype DType::I32
);

make_element!(
    ty u32,
    convert ToElement::to_u32,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &u32, b: &u32| Ord::cmp(a, b),
    dtype DType::U32
);

make_element!(
    ty i16,
    convert ToElement::to_i16,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &i16, b: &i16| Ord::cmp(a, b),
    dtype DType::I16
);

make_element!(
    ty u16,
    convert ToElement::to_u16,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &u16, b: &u16| Ord::cmp(a, b),
    dtype DType::U16
);

make_element!(
    ty i8,
    convert ToElement::to_i8,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &i8, b: &i8| Ord::cmp(a, b),
    dtype DType::I8
);

make_element!(
    ty u8,
    convert ToElement::to_u8,
    random |distribution: Distribution, rng: &mut R| distribution.sampler(rng).sample(),
    cmp |a: &u8, b: &u8| Ord::cmp(a, b),
    dtype DType::U8
);

make_element!(
    ty f16,
    convert ToElement::to_f16,
    random |distribution: Distribution, rng: &mut R| {
        let sample: f32 = distribution.sampler(rng).sample();
        f16::from_elem(sample)
    },
    cmp |a: &f16, b: &f16| a.total_cmp(b),
    dtype DType::F16
);
make_element!(
    ty bf16,
    convert ToElement::to_bf16,
    random |distribution: Distribution, rng: &mut R| {
        let sample: f32 = distribution.sampler(rng).sample();
        bf16::from_elem(sample)
    },
    cmp |a: &bf16, b: &bf16| a.total_cmp(b),
    dtype DType::BF16
);

#[cfg(feature = "cubecl")]
make_element!(
    ty flex32,
    convert |elem: &dyn ToElement| flex32::from_f32(elem.to_f32()),
    random |distribution: Distribution, rng: &mut R| {
        let sample: f32 = distribution.sampler(rng).sample();
        flex32::from_elem(sample)
    },
    cmp |a: &flex32, b: &flex32| a.total_cmp(b),
    dtype DType::Flex32,
    min flex32::from_f32(half::f16::MIN.to_f32_const()),
    max flex32::from_f32(half::f16::MAX.to_f32_const())
);

make_element!(
    ty bool,
    convert ToElement::to_bool,
    random |distribution: Distribution, rng: &mut R| {
        let sample: u8 = distribution.sampler(rng).sample();
        bool::from_elem(sample)
    },
    cmp |a: &bool, b: &bool| Ord::cmp(a, b),
    dtype DType::Bool,
    min false,
    max true
);

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F64,
    F32,
    Flex32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    QFloat(QuantScheme),
}

#[cfg(feature = "cubecl")]
impl From<cubecl::ir::ElemType> for DType {
    fn from(value: cubecl::ir::ElemType) -> Self {
        match value {
            cubecl::ir::ElemType::Float(float_kind) => match float_kind {
                cubecl::ir::FloatKind::F16 => DType::F16,
                cubecl::ir::FloatKind::BF16 => DType::BF16,
                cubecl::ir::FloatKind::Flex32 => DType::Flex32,
                cubecl::ir::FloatKind::F32 => DType::F32,
                cubecl::ir::FloatKind::F64 => DType::F64,
                cubecl::ir::FloatKind::TF32 => panic!("Not a valid DType for tensors."),
                cubecl::ir::FloatKind::E2M1
                | cubecl::ir::FloatKind::E2M3
                | cubecl::ir::FloatKind::E3M2
                | cubecl::ir::FloatKind::E4M3
                | cubecl::ir::FloatKind::E5M2
                | cubecl::ir::FloatKind::UE8M0 => {
                    unimplemented!("Not yet supported, will be used for quantization")
                }
            },
            cubecl::ir::ElemType::Int(int_kind) => match int_kind {
                cubecl::ir::IntKind::I8 => DType::I8,
                cubecl::ir::IntKind::I16 => DType::I16,
                cubecl::ir::IntKind::I32 => DType::I32,
                cubecl::ir::IntKind::I64 => DType::I64,
            },
            cubecl::ir::ElemType::UInt(uint_kind) => match uint_kind {
                cubecl::ir::UIntKind::U8 => DType::U8,
                cubecl::ir::UIntKind::U16 => DType::U16,
                cubecl::ir::UIntKind::U32 => DType::U32,
                cubecl::ir::UIntKind::U64 => DType::U64,
            },
            _ => panic!("Not a valid DType for tensors."),
        }
    }
}

impl DType {
    /// Returns the size of a type in bytes.
    pub const fn size(&self) -> usize {
        match self {
            DType::F64 => core::mem::size_of::<f64>(),
            DType::F32 => core::mem::size_of::<f32>(),
            DType::Flex32 => core::mem::size_of::<f32>(),
            DType::F16 => core::mem::size_of::<f16>(),
            DType::BF16 => core::mem::size_of::<bf16>(),
            DType::I64 => core::mem::size_of::<i64>(),
            DType::I32 => core::mem::size_of::<i32>(),
            DType::I16 => core::mem::size_of::<i16>(),
            DType::I8 => core::mem::size_of::<i8>(),
            DType::U64 => core::mem::size_of::<u64>(),
            DType::U32 => core::mem::size_of::<u32>(),
            DType::U16 => core::mem::size_of::<u16>(),
            DType::U8 => core::mem::size_of::<u8>(),
            DType::Bool => core::mem::size_of::<bool>(),
            DType::QFloat(scheme) => match scheme.store {
                QuantStore::Native => match scheme.value {
                    QuantValue::Q8F | QuantValue::Q8S => core::mem::size_of::<i8>(),
                    QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => {
                        // Sub-byte values have fractional size
                        0
                    }
                },
                QuantStore::U32 => core::mem::size_of::<u32>(),
            },
        }
    }
    /// Returns true if the data type is a floating point type.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DType::F64 | DType::F32 | DType::Flex32 | DType::F16 | DType::BF16
        )
    }
    /// Returns true if the data type is a signed integer type.
    pub fn is_int(&self) -> bool {
        matches!(self, DType::I64 | DType::I32 | DType::I16 | DType::I8)
    }
    /// Returns true if the data type is an unsigned integer type.
    pub fn is_uint(&self) -> bool {
        matches!(self, DType::U64 | DType::U32 | DType::U16 | DType::U8)
    }

    /// Returns true if the data type is a boolean type
    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }

    /// Returns the data type name.
    pub fn name(&self) -> &'static str {
        match self {
            DType::F64 => "f64",
            DType::F32 => "f32",
            DType::Flex32 => "flex32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::I16 => "i16",
            DType::I8 => "i8",
            DType::U64 => "u64",
            DType::U32 => "u32",
            DType::U16 => "u16",
            DType::U8 => "u8",
            DType::Bool => "bool",
            DType::QFloat(_) => "qfloat",
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy)]
pub enum FloatDType {
    F64,
    F32,
    Flex32,
    F16,
    BF16,
}

impl From<DType> for FloatDType {
    fn from(value: DType) -> Self {
        match value {
            DType::F64 => FloatDType::F64,
            DType::F32 => FloatDType::F32,
            DType::Flex32 => FloatDType::Flex32,
            DType::F16 => FloatDType::F16,
            DType::BF16 => FloatDType::BF16,
            _ => panic!("Expected float data type, got {value:?}"),
        }
    }
}

impl From<FloatDType> for DType {
    fn from(value: FloatDType) -> Self {
        match value {
            FloatDType::F64 => DType::F64,
            FloatDType::F32 => DType::F32,
            FloatDType::Flex32 => DType::Flex32,
            FloatDType::F16 => DType::F16,
            FloatDType::BF16 => DType::BF16,
        }
    }
}

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy)]
pub enum IntDType {
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
}

impl From<DType> for IntDType {
    fn from(value: DType) -> Self {
        match value {
            DType::I64 => IntDType::I64,
            DType::I32 => IntDType::I32,
            DType::I16 => IntDType::I16,
            DType::I8 => IntDType::I8,
            DType::U64 => IntDType::U64,
            DType::U32 => IntDType::U32,
            DType::U16 => IntDType::U16,
            DType::U8 => IntDType::U8,
            _ => panic!("Expected int data type, got {value:?}"),
        }
    }
}

impl From<IntDType> for DType {
    fn from(value: IntDType) -> Self {
        match value {
            IntDType::I64 => DType::I64,
            IntDType::I32 => DType::I32,
            IntDType::I16 => DType::I16,
            IntDType::I8 => DType::I8,
            IntDType::U64 => DType::U64,
            IntDType::U32 => DType::U32,
            IntDType::U16 => DType::U16,
            IntDType::U8 => DType::U8,
        }
    }
}
