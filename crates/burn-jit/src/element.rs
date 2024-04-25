use crate::codegen::dialect::gpu;
use burn_tensor::Element;

/// The base element trait for the jit backend.
pub trait JitElement:
    burn_tensor::Element + core::fmt::Debug + Send + Sync + 'static + Clone + bytemuck::Pod
where
    Self: Sized,
{
    /// TODO: Remove when all wgsl static kernels are migrated.
    fn type_name() -> &'static str;
    /// Convert a slice of elements to a slice of bytes.
    fn as_bytes(slice: &[Self]) -> &[u8];
    /// Convert a slice of bytes to a slice of elements.
    fn from_bytes(bytes: &[u8]) -> &[Self];
    /// Element representation for `gpu`.
    fn gpu_elem() -> gpu::Elem;
    /// Highest possible value
    fn maximum_value() -> Self;
    /// Lowest possible value
    fn minimum_value() -> Self;
}

/// The float element type for the jit backend.
pub trait FloatElement: JitElement + Element {}

/// The int element type for the jit backend.
pub trait IntElement: JitElement + Element {}

impl JitElement for u32 {
    fn type_name() -> &'static str {
        "u32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn gpu_elem() -> gpu::Elem {
        gpu::Elem::UInt
    }
    fn maximum_value() -> Self {
        u32::MAX
    }
    fn minimum_value() -> Self {
        u32::MIN
    }
}

impl JitElement for i32 {
    fn type_name() -> &'static str {
        "i32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn gpu_elem() -> gpu::Elem {
        gpu::Elem::Int
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

impl JitElement for f32 {
    fn type_name() -> &'static str {
        "f32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    fn gpu_elem() -> gpu::Elem {
        gpu::Elem::Float
    }
    fn maximum_value() -> Self {
        f32::MAX
    }
    fn minimum_value() -> Self {
        f32::MIN
    }
}

impl FloatElement for f32 {}
impl IntElement for i32 {}
