use burn_tensor::Element;

pub trait WGPUElement: core::fmt::Debug + 'static + Clone
where
    Self: Sized,
{
    fn type_name() -> &'static str;
    fn as_bytes(slice: &[Self]) -> &[u8];
    fn from_bytes(bytes: &[u8]) -> &[Self];
}

pub trait FloatElement: WGPUElement + Element {}

pub trait IntElement: WGPUElement + Element {}

impl WGPUElement for u32 {
    fn type_name() -> &'static str {
        "u32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
}

impl WGPUElement for i32 {
    fn type_name() -> &'static str {
        "i32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
}

impl WGPUElement for i64 {
    fn type_name() -> &'static str {
        "i64"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
}

impl WGPUElement for f32 {
    fn type_name() -> &'static str {
        "f32"
    }
    fn as_bytes(slice: &[Self]) -> &[u8] {
        bytemuck::cast_slice(slice)
    }
    fn from_bytes(bytes: &[u8]) -> &[Self] {
        bytemuck::cast_slice(bytes)
    }
}

impl FloatElement for f32 {}
impl IntElement for i32 {}
impl IntElement for i64 {}
