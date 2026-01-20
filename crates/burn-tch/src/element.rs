use burn_backend::Element;
use burn_backend::{bf16, f16};

/// The element type for the tch backend.
pub trait TchElement: Element + tch::kind::Element {
    /// Returns the associated tensor kind for [`tch::kind::Element`].
    fn kind() -> tch::Kind {
        Self::KIND
    }
}

impl TchElement for f64 {}
impl TchElement for f32 {}
impl TchElement for f16 {}
impl TchElement for bf16 {
    fn kind() -> tch::Kind {
        let mut kind = <Self as tch::kind::Element>::KIND;
        // Incorrect kind mapping in tch definitions, force bfloat16
        if matches!(Self::dtype(), burn_backend::DType::BF16) && kind == tch::Kind::Half {
            kind = tch::Kind::BFloat16
        }
        kind
    }
}

impl TchElement for i64 {}
impl TchElement for i32 {}
impl TchElement for i16 {}
impl TchElement for i8 {}

impl TchElement for u8 {}

impl TchElement for bool {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elem_kinds() {
        assert_eq!(f64::kind(), tch::Kind::Double);
        assert_eq!(f32::kind(), tch::Kind::Float);
        assert_eq!(f16::kind(), tch::Kind::Half);
        assert_eq!(bf16::kind(), tch::Kind::BFloat16);
        assert_eq!(i64::kind(), tch::Kind::Int64);
        assert_eq!(i32::kind(), tch::Kind::Int);
        assert_eq!(i16::kind(), tch::Kind::Int16);
        assert_eq!(i8::kind(), tch::Kind::Int8);
        assert_eq!(bool::kind(), tch::Kind::Bool);
    }
}
