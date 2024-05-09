use crate::{ExpandElement, RuntimeType};

pub trait Int:
    Clone
    + Copy
    + RuntimeType<ExpandType = ExpandElement>
    + std::cmp::PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
{
    fn into_kind() -> burn_jit::gpu::IntKind;
    fn new(val: i32, vectorization: usize) -> Self;
}

macro_rules! impl_int {
    ($type:ident) => {
        #[derive(Clone, Copy)]
        pub struct $type {
            pub val: i32,
            pub vectorization: usize,
        }

        impl RuntimeType for $type {
            type ExpandType = ExpandElement;
        }

        impl Int for $type {
            fn into_kind() -> burn_jit::gpu::IntKind {
                burn_jit::gpu::IntKind::$type
            }
            fn new(val: i32, vectorization: usize) -> Self {
                Self { val, vectorization }
            }
        }

        impl From<i32> for $type {
            fn from(value: i32) -> Self {
                $type::new(value, 1)
            }
        }
    };
}

impl_int!(I32);
impl_int!(I64);
