use core::mem::size_of;

use half::{bf16, f16};

/// A generic trait for converting a value to a number.
/// Adapted from [num_traits::ToPrimitive] to support [bool].
///
/// A value can be represented by the target type when it lies within
/// the range of scalars supported by the target type.
/// For example, a negative integer cannot be represented by an unsigned
/// integer type, and an `i64` with a very high magnitude might not be
/// convertible to an `i32`.
/// On the other hand, conversions with possible precision loss or truncation
/// are admitted, like an `f32` with a decimal part to an integer type, or
/// even a large `f64` saturating to `f32` infinity.
///
/// The methods *panic* when the value cannot be represented by the target type.
pub trait ToElement {
    /// Converts the value of `self` to an `isize`.
    #[inline]
    fn to_isize(&self) -> isize {
        ToElement::to_isize(&self.to_i64())
    }

    /// Converts the value of `self` to an `i8`.
    #[inline]
    fn to_i8(&self) -> i8 {
        ToElement::to_i8(&self.to_i64())
    }

    /// Converts the value of `self` to an `i16`.
    #[inline]
    fn to_i16(&self) -> i16 {
        ToElement::to_i16(&self.to_i64())
    }

    /// Converts the value of `self` to an `i32`.
    #[inline]
    fn to_i32(&self) -> i32 {
        ToElement::to_i32(&self.to_i64())
    }

    /// Converts the value of `self` to an `i64`.
    fn to_i64(&self) -> i64;

    /// Converts the value of `self` to an `i128`.
    ///
    /// The default implementation converts through `to_i64()`. Types implementing
    /// this trait should override this method if they can represent a greater range.
    #[inline]
    fn to_i128(&self) -> i128 {
        i128::from(self.to_i64())
    }

    /// Converts the value of `self` to a `usize`.
    #[inline]
    fn to_usize(&self) -> usize {
        ToElement::to_usize(&self.to_u64())
    }

    /// Converts the value of `self` to a `u8`.
    #[inline]
    fn to_u8(&self) -> u8 {
        ToElement::to_u8(&self.to_u64())
    }

    /// Converts the value of `self` to a `u16`.
    #[inline]
    fn to_u16(&self) -> u16 {
        ToElement::to_u16(&self.to_u64())
    }

    /// Converts the value of `self` to a `u32`.
    #[inline]
    fn to_u32(&self) -> u32 {
        ToElement::to_u32(&self.to_u64())
    }

    /// Converts the value of `self` to a `u64`.
    fn to_u64(&self) -> u64;

    /// Converts the value of `self` to a `u128`.
    ///
    /// The default implementation converts through `to_u64()`. Types implementing
    /// this trait should override this method if they can represent a greater range.
    #[inline]
    fn to_u128(&self) -> u128 {
        u128::from(self.to_u64())
    }

    /// Converts the value of `self` to an `f32`. Overflows may map to positive
    /// or negative infinity.
    #[inline]
    fn to_f32(&self) -> f32 {
        ToElement::to_f32(&self.to_f64())
    }

    /// Converts the value of `self` to an `f64`. Overflows may map to positive
    /// or negative infinity.
    ///
    /// The default implementation tries to convert through `to_i64()`, and
    /// failing that through `to_u64()`. Types implementing this trait should
    /// override this method if they can represent a greater range.
    #[inline]
    fn to_f64(&self) -> f64 {
        ToElement::to_f64(&self.to_u64())
    }
}

macro_rules! impl_to_element_int_to_int {
    ($SrcT:ident : $( $(#[$cfg:meta])* fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> $DstT {
            let min = $DstT::MIN as $SrcT;
            let max = $DstT::MAX as $SrcT;
            if size_of::<$SrcT>() <= size_of::<$DstT>() || (min <= *self && *self <= max) {
                *self as $DstT
            } else {
                panic!("Element cannot be represented in the target type")
            }
        }
    )*}
}

macro_rules! impl_to_element_int_to_uint {
    ($SrcT:ident : $( $(#[$cfg:meta])* fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> $DstT {
            let max = $DstT::MAX as $SrcT;
            if 0 <= *self && (size_of::<$SrcT>() <= size_of::<$DstT>() || *self <= max) {
                *self as $DstT
            } else {
                panic!("Element cannot be represented in the target type")
            }
        }
    )*}
}

macro_rules! impl_to_element_int {
    ($T:ident) => {
        impl ToElement for $T {
            impl_to_element_int_to_int! { $T:
                fn to_isize -> isize;
                fn to_i8 -> i8;
                fn to_i16 -> i16;
                fn to_i32 -> i32;
                fn to_i64 -> i64;
                fn to_i128 -> i128;
            }

            impl_to_element_int_to_uint! { $T:
                fn to_usize -> usize;
                fn to_u8 -> u8;
                fn to_u16 -> u16;
                fn to_u32 -> u32;
                fn to_u64 -> u64;
                fn to_u128 -> u128;
            }

            #[inline]
            fn to_f32(&self) -> f32 {
                *self as f32
            }
            #[inline]
            fn to_f64(&self) -> f64 {
                *self as f64
            }
        }
    };
}

impl_to_element_int!(isize);
impl_to_element_int!(i8);
impl_to_element_int!(i16);
impl_to_element_int!(i32);
impl_to_element_int!(i64);
impl_to_element_int!(i128);

macro_rules! impl_to_element_uint_to_int {
    ($SrcT:ident : $( $(#[$cfg:meta])* fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> $DstT {
            let max = $DstT::MAX as $SrcT;
            if size_of::<$SrcT>() < size_of::<$DstT>() || *self <= max {
                *self as $DstT
            } else {
                panic!("Element cannot be represented in the target type")
            }
        }
    )*}
}

macro_rules! impl_to_element_uint_to_uint {
    ($SrcT:ident : $( $(#[$cfg:meta])* fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> $DstT {
            let max = $DstT::MAX as $SrcT;
            if size_of::<$SrcT>() <= size_of::<$DstT>() || *self <= max {
                *self as $DstT
            } else {
                panic!("Element cannot be represented in the target type")
            }
        }
    )*}
}

macro_rules! impl_to_element_uint {
    ($T:ident) => {
        impl ToElement for $T {
            impl_to_element_uint_to_int! { $T:
                fn to_isize -> isize;
                fn to_i8 -> i8;
                fn to_i16 -> i16;
                fn to_i32 -> i32;
                fn to_i64 -> i64;
                fn to_i128 -> i128;
            }

            impl_to_element_uint_to_uint! { $T:
                fn to_usize -> usize;
                fn to_u8 -> u8;
                fn to_u16 -> u16;
                fn to_u32 -> u32;
                fn to_u64 -> u64;
                fn to_u128 -> u128;
            }

            #[inline]
            fn to_f32(&self) -> f32 {
                *self as f32
            }
            #[inline]
            fn to_f64(&self) -> f64 {
                *self as f64
            }
        }
    };
}

impl_to_element_uint!(usize);
impl_to_element_uint!(u8);
impl_to_element_uint!(u16);
impl_to_element_uint!(u32);
impl_to_element_uint!(u64);
impl_to_element_uint!(u128);

macro_rules! impl_to_element_float_to_float {
    ($SrcT:ident : $( fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        fn $method(&self) -> $DstT {
            // We can safely cast all values, whether NaN, +-inf, or finite.
            // Finite values that are reducing size may saturate to +-inf.
            *self as $DstT
        }
    )*}
}

macro_rules! float_to_int_unchecked {
    // SAFETY: Must not be NaN or infinite; must be representable as the integer after truncating.
    // We already checked that the float is in the exclusive range `(MIN-1, MAX+1)`.
    ($float:expr => $int:ty) => {
        unsafe { $float.to_int_unchecked::<$int>() }
    };
}

macro_rules! impl_to_element_float_to_signed_int {
    ($f:ident : $( $(#[$cfg:meta])* fn $method:ident -> $i:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> $i {
            // Float as int truncates toward zero, so we want to allow values
            // in the exclusive range `(MIN-1, MAX+1)`.
            if size_of::<$f>() > size_of::<$i>() {
                // With a larger size, we can represent the range exactly.
                const MIN_M1: $f = $i::MIN as $f - 1.0;
                const MAX_P1: $f = $i::MAX as $f + 1.0;
                if *self > MIN_M1 && *self < MAX_P1 {
                    return float_to_int_unchecked!(*self => $i);
                }
            } else {
                // We can't represent `MIN-1` exactly, but there's no fractional part
                // at this magnitude, so we can just use a `MIN` inclusive boundary.
                const MIN: $f = $i::MIN as $f;
                // We can't represent `MAX` exactly, but it will round up to exactly
                // `MAX+1` (a power of two) when we cast it.
                const MAX_P1: $f = $i::MAX as $f;
                if *self >= MIN && *self < MAX_P1 {
                    return float_to_int_unchecked!(*self => $i);
                }
            }
            panic!("Float cannot be represented in the target signed int type")
        }
    )*}
}

macro_rules! impl_to_element_float_to_unsigned_int {
    ($f:ident : $( $(#[$cfg:meta])* fn $method:ident -> $u:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> $u {
            // Float as int truncates toward zero, so we want to allow values
            // in the exclusive range `(-1, MAX+1)`.
            if size_of::<$f>() > size_of::<$u>() {
                // With a larger size, we can represent the range exactly.
                const MAX_P1: $f = $u::MAX as $f + 1.0;
                if *self > -1.0 && *self < MAX_P1 {
                    return float_to_int_unchecked!(*self => $u);
                }
            } else {
                // We can't represent `MAX` exactly, but it will round up to exactly
                // `MAX+1` (a power of two) when we cast it.
                // (`u128::MAX as f32` is infinity, but this is still ok.)
                const MAX_P1: $f = $u::MAX as $f;
                if *self > -1.0 && *self < MAX_P1 {
                    return float_to_int_unchecked!(*self => $u);
                }
            }
            panic!("Float cannot be represented in the target unsigned int type")
        }
    )*}
}

macro_rules! impl_to_element_float {
    ($T:ident) => {
        impl ToElement for $T {
            impl_to_element_float_to_signed_int! { $T:
                fn to_isize -> isize;
                fn to_i8 -> i8;
                fn to_i16 -> i16;
                fn to_i32 -> i32;
                fn to_i64 -> i64;
                fn to_i128 -> i128;
            }

            impl_to_element_float_to_unsigned_int! { $T:
                fn to_usize -> usize;
                fn to_u8 -> u8;
                fn to_u16 -> u16;
                fn to_u32 -> u32;
                fn to_u64 -> u64;
                fn to_u128 -> u128;
            }

            impl_to_element_float_to_float! { $T:
                fn to_f32 -> f32;
                fn to_f64 -> f64;
            }
        }
    };
}

impl_to_element_float!(f32);
impl_to_element_float!(f64);

impl ToElement for f16 {
    #[inline]
    fn to_i64(&self) -> i64 {
        Self::to_f32(*self).to_i64()
    }
    #[inline]
    fn to_u64(&self) -> u64 {
        Self::to_f32(*self).to_u64()
    }
    #[inline]
    fn to_i8(&self) -> i8 {
        Self::to_f32(*self).to_i8()
    }
    #[inline]
    fn to_u8(&self) -> u8 {
        Self::to_f32(*self).to_u8()
    }
    #[inline]
    fn to_i16(&self) -> i16 {
        Self::to_f32(*self).to_i16()
    }
    #[inline]
    fn to_u16(&self) -> u16 {
        Self::to_f32(*self).to_u16()
    }
    #[inline]
    fn to_i32(&self) -> i32 {
        Self::to_f32(*self).to_i32()
    }
    #[inline]
    fn to_u32(&self) -> u32 {
        Self::to_f32(*self).to_u32()
    }
    #[inline]
    fn to_f32(&self) -> f32 {
        Self::to_f32(*self)
    }
    #[inline]
    fn to_f64(&self) -> f64 {
        Self::to_f64(*self)
    }
}

impl ToElement for bf16 {
    #[inline]
    fn to_i64(&self) -> i64 {
        Self::to_f32(*self).to_i64()
    }
    #[inline]
    fn to_u64(&self) -> u64 {
        Self::to_f32(*self).to_u64()
    }
    #[inline]
    fn to_i8(&self) -> i8 {
        Self::to_f32(*self).to_i8()
    }
    #[inline]
    fn to_u8(&self) -> u8 {
        Self::to_f32(*self).to_u8()
    }
    #[inline]
    fn to_i16(&self) -> i16 {
        Self::to_f32(*self).to_i16()
    }
    #[inline]
    fn to_u16(&self) -> u16 {
        Self::to_f32(*self).to_u16()
    }
    #[inline]
    fn to_i32(&self) -> i32 {
        Self::to_f32(*self).to_i32()
    }
    #[inline]
    fn to_u32(&self) -> u32 {
        Self::to_f32(*self).to_u32()
    }
    #[inline]
    fn to_f32(&self) -> f32 {
        Self::to_f32(*self)
    }
    #[inline]
    fn to_f64(&self) -> f64 {
        Self::to_f64(*self)
    }
}

impl ToElement for bool {
    #[inline]
    fn to_i64(&self) -> i64 {
        *self as i64
    }
    #[inline]
    fn to_u64(&self) -> u64 {
        *self as u64
    }
    #[inline]
    fn to_i8(&self) -> i8 {
        *self as i8
    }
    #[inline]
    fn to_u8(&self) -> u8 {
        *self as u8
    }
    #[inline]
    fn to_i16(&self) -> i16 {
        *self as i16
    }
    #[inline]
    fn to_u16(&self) -> u16 {
        *self as u16
    }
    #[inline]
    fn to_i32(&self) -> i32 {
        *self as i32
    }
    #[inline]
    fn to_u32(&self) -> u32 {
        *self as u32
    }
    #[inline]
    fn to_f32(&self) -> f32 {
        self.to_u8() as f32
    }
    #[inline]
    fn to_f64(&self) -> f64 {
        self.to_u8() as f64
    }
}

mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn to_element_float() {
        let f32_toolarge = 1e39f64;
        assert_eq!(f32_toolarge.to_f32(), f32::INFINITY);
        assert_eq!((-f32_toolarge).to_f32(), f32::NEG_INFINITY);
        assert_eq!((f32::MAX as f64).to_f32(), f32::MAX);
        assert_eq!((-f32::MAX as f64).to_f32(), -f32::MAX);
        assert_eq!(f64::INFINITY.to_f32(), f32::INFINITY);
        assert_eq!((f64::NEG_INFINITY).to_f32(), f32::NEG_INFINITY);
        assert!((f64::NAN).to_f32().is_nan());
    }

    #[test]
    #[should_panic]
    fn to_element_signed_to_u8_underflow() {
        let _x = (-1i8).to_u8();
    }

    #[test]
    #[should_panic]
    fn to_element_signed_to_u16_underflow() {
        let _x = (-1i8).to_u16();
    }

    #[test]
    #[should_panic]
    fn to_element_signed_to_u32_underflow() {
        let _x = (-1i8).to_u32();
    }

    #[test]
    #[should_panic]
    fn to_element_signed_to_u64_underflow() {
        let _x = (-1i8).to_u64();
    }

    #[test]
    #[should_panic]
    fn to_element_signed_to_u128_underflow() {
        let _x = (-1i8).to_u128();
    }

    #[test]
    #[should_panic]
    fn to_element_signed_to_usize_underflow() {
        let _x = (-1i8).to_usize();
    }

    #[test]
    #[should_panic]
    fn to_element_unsigned_to_u8_overflow() {
        let _x = 256.to_u8();
    }

    #[test]
    #[should_panic]
    fn to_element_unsigned_to_u16_overflow() {
        let _x = 65_536.to_u16();
    }

    #[test]
    #[should_panic]
    fn to_element_unsigned_to_u32_overflow() {
        let _x = 4_294_967_296u64.to_u32();
    }

    #[test]
    #[should_panic]
    fn to_element_unsigned_to_u64_overflow() {
        let _x = 18_446_744_073_709_551_616u128.to_u64();
    }

    #[test]
    fn to_element_int_to_float() {
        assert_eq!((-1).to_f32(), -1.0);
        assert_eq!((-1).to_f64(), -1.0);
        assert_eq!(255.to_f32(), 255.0);
        assert_eq!(65_535.to_f64(), 65_535.0);
    }

    #[test]
    fn to_element_float_to_int() {
        assert_eq!((-1.0).to_i8(), -1);
        assert_eq!(1.0.to_u8(), 1);
        assert_eq!(1.8.to_u16(), 1);
        assert_eq!(123.456.to_u32(), 123);
    }
}
