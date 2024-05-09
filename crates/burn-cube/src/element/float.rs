use crate::{ExpandElement, RuntimeType};

pub trait Float:
    Clone
    + Copy
    + RuntimeType<ExpandType = ExpandElement>
    + std::cmp::PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    fn into_kind() -> burn_jit::gpu::FloatKind;
    fn new(val: f32, vectorization: usize) -> Self;
}

macro_rules! impl_float {
    ($type:ident) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: f32,
            pub vectorization: usize,
        }

        impl RuntimeType for $type {
            type ExpandType = ExpandElement;
        }

        impl Float for $type {
            fn into_kind() -> burn_jit::gpu::FloatKind {
                burn_jit::gpu::FloatKind::$type
            }
            fn new(val: f32, vectorization: usize) -> Self {
                Self { val, vectorization }
            }
        }
        impl From<f32> for $type {
            fn from(value: f32) -> Self {
                $type::new(value, 1)
            }
        }
    };
}

impl_float!(F16);
impl_float!(BF16);
impl_float!(F32);
impl_float!(F64);
