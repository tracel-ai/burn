use burn_tensor::Element;

/// The base element trait for the wgou backend.
pub trait WgpuElement:
    burn_tensor::Element + core::fmt::Debug + Send + Sync + 'static + Clone + bytemuck::Pod
where
    Self: Sized,
{
    fn type_name() -> &'static str;
    fn as_bytes(slice: &[Self]) -> &[u8];
    fn from_bytes(bytes: &[u8]) -> &[Self];
    #[cfg(any(feature = "fusion", test))]
    fn elem_type() -> crate::fusion::codegen::Elem;
}

/// The float element type for the wgpu backend.
pub trait FloatElement: WgpuElement + Element {}

/// The int element type for the wgpu backend.
pub trait IntElement: WgpuElement + Element {}

impl WgpuElement for u32 {
    fn type_name() -> &'static str {
        "u32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    #[cfg(any(feature = "fusion", test))]
    fn elem_type() -> crate::fusion::codegen::Elem {
        crate::fusion::codegen::Elem::U32
    }
}

impl WgpuElement for i32 {
    fn type_name() -> &'static str {
        "i32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
    #[cfg(any(feature = "fusion", test))]
    fn elem_type() -> crate::fusion::codegen::Elem {
        crate::fusion::codegen::Elem::I32
    }
}

impl WgpuElement for f32 {
    fn type_name() -> &'static str {
        "f32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }

    #[cfg(any(feature = "fusion", test))]
    fn elem_type() -> crate::fusion::codegen::Elem {
        crate::fusion::codegen::Elem::F32
    }
}

impl FloatElement for f32 {}
impl IntElement for i32 {}
