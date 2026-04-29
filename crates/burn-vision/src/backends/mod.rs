pub(crate) mod cpu;
#[cfg(feature = "cubecl-backend")]
mod cube;

pub use cpu::{KernelShape, create_structuring_element};

/// Dispatches connected components based on `B::IntElem::dtype()`, binding a concrete
/// integer type to enable generic instantiations without extra trait bounds (after removing
/// `ElementComparison` from `Element`).
#[macro_export]
macro_rules! dispatch_int_dtype {
    ($dtype:expr, |$ty:ident| $body:expr) => {
        match $dtype {
            IntDType::I64 => {
                type $ty = i64;
                $body
            }
            IntDType::I32 => {
                type $ty = i32;
                $body
            }
            IntDType::I16 => {
                type $ty = i16;
                $body
            }
            IntDType::I8 => {
                type $ty = i8;
                $body
            }
            IntDType::U64 => {
                type $ty = u64;
                $body
            }
            IntDType::U32 => {
                type $ty = u32;
                $body
            }
            IntDType::U16 => {
                type $ty = u16;
                $body
            }
            IntDType::U8 => {
                type $ty = u8;
                $body
            }
        }
    };
}
