use crate::ir::{Elem, FloatKind, IntKind};

/// The base element trait for the jit backend.
pub trait CubeElement: core::fmt::Debug + Send + Sync + 'static + Clone + bytemuck::Pod {
    /// Returns the name of the type.
    fn type_name() -> &'static str;
    /// Convert a slice of elements to a slice of bytes.
    fn as_bytes(slice: &[Self]) -> &[u8];
    /// Convert a slice of bytes to a slice of elements.
    fn from_bytes(bytes: &[u8]) -> &[Self];
    /// Element representation for `cubecl`.
    fn cube_elem() -> Elem;
    /// Highest possible value
    fn maximum_value() -> Self;
    /// Lowest possible value
    fn minimum_value() -> Self;
}

impl CubeElement for u32 {
    fn type_name() -> &'static str {
        "u32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_elem() -> Elem {
        Elem::UInt
    }
    fn maximum_value() -> Self {
        u32::MAX
    }
    fn minimum_value() -> Self {
        u32::MIN
    }
}

impl CubeElement for i32 {
    fn type_name() -> &'static str {
        "i32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_elem() -> Elem {
        Elem::Int(IntKind::I32)
    }
    fn maximum_value() -> Self {
        // Seems to cause problem for some GPU
        i32::MAX - 1
    }
    fn minimum_value() -> Self {
        // Seems to cause problem for some GPU
        i32::MIN + 1
    }
}

impl CubeElement for f32 {
    fn type_name() -> &'static str {
        "f32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_elem() -> Elem {
        Elem::Float(FloatKind::F32)
    }
    fn maximum_value() -> Self {
        f32::MAX
    }
    fn minimum_value() -> Self {
        f32::MIN
    }
}

impl CubeElement for half::f16 {
    fn type_name() -> &'static str {
        "f16"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_elem() -> Elem {
        Elem::Float(FloatKind::F16)
    }
    fn maximum_value() -> Self {
        half::f16::MAX
    }
    fn minimum_value() -> Self {
        half::f16::MIN
    }
}

impl CubeElement for half::bf16 {
    fn type_name() -> &'static str {
        "bf16"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn cube_elem() -> Elem {
        Elem::Float(FloatKind::BF16)
    }
    fn maximum_value() -> Self {
        half::bf16::MAX
    }
    fn minimum_value() -> Self {
        half::bf16::MIN
    }
}
