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
pub trait ToPrimitive {
    /// Converts the value of `self` to an `isize`. If the value cannot be
    /// represented by an `isize`, then `None` is returned.
    #[inline]
    fn to_isize(&self) -> Option<isize> {
        self.to_i64().as_ref().and_then(ToPrimitive::to_isize)
    }

    /// Converts the value of `self` to an `i8`. If the value cannot be
    /// represented by an `i8`, then `None` is returned.
    #[inline]
    fn to_i8(&self) -> Option<i8> {
        self.to_i64().as_ref().and_then(ToPrimitive::to_i8)
    }

    /// Converts the value of `self` to an `i16`. If the value cannot be
    /// represented by an `i16`, then `None` is returned.
    #[inline]
    fn to_i16(&self) -> Option<i16> {
        self.to_i64().as_ref().and_then(ToPrimitive::to_i16)
    }

    /// Converts the value of `self` to an `i32`. If the value cannot be
    /// represented by an `i32`, then `None` is returned.
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        self.to_i64().as_ref().and_then(ToPrimitive::to_i32)
    }

    /// Converts the value of `self` to an `i64`. If the value cannot be
    /// represented by an `i64`, then `None` is returned.
    fn to_i64(&self) -> Option<i64>;

    /// Converts the value of `self` to an `i128`. If the value cannot be
    /// represented by an `i128` (`i64` under the default implementation), then
    /// `None` is returned.
    ///
    /// The default implementation converts through `to_i64()`. Types implementing
    /// this trait should override this method if they can represent a greater range.
    #[inline]
    fn to_i128(&self) -> Option<i128> {
        self.to_i64().map(From::from)
    }

    /// Converts the value of `self` to a `usize`. If the value cannot be
    /// represented by a `usize`, then `None` is returned.
    #[inline]
    fn to_usize(&self) -> Option<usize> {
        self.to_u64().as_ref().and_then(ToPrimitive::to_usize)
    }

    /// Converts the value of `self` to a `u8`. If the value cannot be
    /// represented by a `u8`, then `None` is returned.
    #[inline]
    fn to_u8(&self) -> Option<u8> {
        self.to_u64().as_ref().and_then(ToPrimitive::to_u8)
    }

    /// Converts the value of `self` to a `u16`. If the value cannot be
    /// represented by a `u16`, then `None` is returned.
    #[inline]
    fn to_u16(&self) -> Option<u16> {
        self.to_u64().as_ref().and_then(ToPrimitive::to_u16)
    }

    /// Converts the value of `self` to a `u32`. If the value cannot be
    /// represented by a `u32`, then `None` is returned.
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        self.to_u64().as_ref().and_then(ToPrimitive::to_u32)
    }

    /// Converts the value of `self` to a `u64`. If the value cannot be
    /// represented by a `u64`, then `None` is returned.
    fn to_u64(&self) -> Option<u64>;

    /// Converts the value of `self` to a `u128`. If the value cannot be
    /// represented by a `u128` (`u64` under the default implementation), then
    /// `None` is returned.
    ///
    /// The default implementation converts through `to_u64()`. Types implementing
    /// this trait should override this method if they can represent a greater range.
    #[inline]
    fn to_u128(&self) -> Option<u128> {
        self.to_u64().map(From::from)
    }

    /// Converts the value of `self` to an `f32`. Overflows may map to positive
    /// or negative infinity, otherwise `None` is returned if the value cannot
    /// be represented by an `f32`.
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        self.to_f64().as_ref().and_then(ToPrimitive::to_f32)
    }

    /// Converts the value of `self` to an `f64`. Overflows may map to positive
    /// or negative infinity, otherwise `None` is returned if the value cannot
    /// be represented by an `f64`.
    ///
    /// The default implementation tries to convert through `to_i64()`, and
    /// failing that through `to_u64()`. Types implementing this trait should
    /// override this method if they can represent a greater range.
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        match self.to_i64() {
            Some(i) => i.to_f64(),
            None => self.to_u64().as_ref().and_then(ToPrimitive::to_f64),
        }
    }
}

macro_rules! impl_to_primitive_int_to_int {
    ($SrcT:ident : $( $(#[$cfg:meta])* fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> Option<$DstT> {
            let min = $DstT::MIN as $SrcT;
            let max = $DstT::MAX as $SrcT;
            if size_of::<$SrcT>() <= size_of::<$DstT>() || (min <= *self && *self <= max) {
                Some(*self as $DstT)
            } else {
                None
            }
        }
    )*}
}

macro_rules! impl_to_primitive_int_to_uint {
    ($SrcT:ident : $( $(#[$cfg:meta])* fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> Option<$DstT> {
            let max = $DstT::MAX as $SrcT;
            if 0 <= *self && (size_of::<$SrcT>() <= size_of::<$DstT>() || *self <= max) {
                Some(*self as $DstT)
            } else {
                None
            }
        }
    )*}
}

macro_rules! impl_to_primitive_int {
    ($T:ident) => {
        impl ToPrimitive for $T {
            impl_to_primitive_int_to_int! { $T:
                fn to_isize -> isize;
                fn to_i8 -> i8;
                fn to_i16 -> i16;
                fn to_i32 -> i32;
                fn to_i64 -> i64;
                fn to_i128 -> i128;
            }

            impl_to_primitive_int_to_uint! { $T:
                fn to_usize -> usize;
                fn to_u8 -> u8;
                fn to_u16 -> u16;
                fn to_u32 -> u32;
                fn to_u64 -> u64;
                fn to_u128 -> u128;
            }

            #[inline]
            fn to_f32(&self) -> Option<f32> {
                Some(*self as f32)
            }
            #[inline]
            fn to_f64(&self) -> Option<f64> {
                Some(*self as f64)
            }
        }
    };
}

impl_to_primitive_int!(isize);
impl_to_primitive_int!(i8);
impl_to_primitive_int!(i16);
impl_to_primitive_int!(i32);
impl_to_primitive_int!(i64);
impl_to_primitive_int!(i128);

macro_rules! impl_to_primitive_uint_to_int {
    ($SrcT:ident : $( $(#[$cfg:meta])* fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> Option<$DstT> {
            let max = $DstT::MAX as $SrcT;
            if size_of::<$SrcT>() < size_of::<$DstT>() || *self <= max {
                Some(*self as $DstT)
            } else {
                None
            }
        }
    )*}
}

macro_rules! impl_to_primitive_uint_to_uint {
    ($SrcT:ident : $( $(#[$cfg:meta])* fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> Option<$DstT> {
            let max = $DstT::MAX as $SrcT;
            if size_of::<$SrcT>() <= size_of::<$DstT>() || *self <= max {
                Some(*self as $DstT)
            } else {
                None
            }
        }
    )*}
}

macro_rules! impl_to_primitive_uint {
    ($T:ident) => {
        impl ToPrimitive for $T {
            impl_to_primitive_uint_to_int! { $T:
                fn to_isize -> isize;
                fn to_i8 -> i8;
                fn to_i16 -> i16;
                fn to_i32 -> i32;
                fn to_i64 -> i64;
                fn to_i128 -> i128;
            }

            impl_to_primitive_uint_to_uint! { $T:
                fn to_usize -> usize;
                fn to_u8 -> u8;
                fn to_u16 -> u16;
                fn to_u32 -> u32;
                fn to_u64 -> u64;
                fn to_u128 -> u128;
            }

            #[inline]
            fn to_f32(&self) -> Option<f32> {
                Some(*self as f32)
            }
            #[inline]
            fn to_f64(&self) -> Option<f64> {
                Some(*self as f64)
            }
        }
    };
}

impl_to_primitive_uint!(usize);
impl_to_primitive_uint!(u8);
impl_to_primitive_uint!(u16);
impl_to_primitive_uint!(u32);
impl_to_primitive_uint!(u64);
impl_to_primitive_uint!(u128);

macro_rules! impl_to_primitive_float_to_float {
    ($SrcT:ident : $( fn $method:ident -> $DstT:ident ; )*) => {$(
        #[inline]
        fn $method(&self) -> Option<$DstT> {
            // We can safely cast all values, whether NaN, +-inf, or finite.
            // Finite values that are reducing size may saturate to +-inf.
            Some(*self as $DstT)
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

macro_rules! impl_to_primitive_float_to_signed_int {
    ($f:ident : $( $(#[$cfg:meta])* fn $method:ident -> $i:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> Option<$i> {
            // Float as int truncates toward zero, so we want to allow values
            // in the exclusive range `(MIN-1, MAX+1)`.
            if size_of::<$f>() > size_of::<$i>() {
                // With a larger size, we can represent the range exactly.
                const MIN_M1: $f = $i::MIN as $f - 1.0;
                const MAX_P1: $f = $i::MAX as $f + 1.0;
                if *self > MIN_M1 && *self < MAX_P1 {
                    return Some(float_to_int_unchecked!(*self => $i));
                }
            } else {
                // We can't represent `MIN-1` exactly, but there's no fractional part
                // at this magnitude, so we can just use a `MIN` inclusive boundary.
                const MIN: $f = $i::MIN as $f;
                // We can't represent `MAX` exactly, but it will round up to exactly
                // `MAX+1` (a power of two) when we cast it.
                const MAX_P1: $f = $i::MAX as $f;
                if *self >= MIN && *self < MAX_P1 {
                    return Some(float_to_int_unchecked!(*self => $i));
                }
            }
            None
        }
    )*}
}

macro_rules! impl_to_primitive_float_to_unsigned_int {
    ($f:ident : $( $(#[$cfg:meta])* fn $method:ident -> $u:ident ; )*) => {$(
        #[inline]
        $(#[$cfg])*
        fn $method(&self) -> Option<$u> {
            // Float as int truncates toward zero, so we want to allow values
            // in the exclusive range `(-1, MAX+1)`.
            if size_of::<$f>() > size_of::<$u>() {
                // With a larger size, we can represent the range exactly.
                const MAX_P1: $f = $u::MAX as $f + 1.0;
                if *self > -1.0 && *self < MAX_P1 {
                    return Some(float_to_int_unchecked!(*self => $u));
                }
            } else {
                // We can't represent `MAX` exactly, but it will round up to exactly
                // `MAX+1` (a power of two) when we cast it.
                // (`u128::MAX as f32` is infinity, but this is still ok.)
                const MAX_P1: $f = $u::MAX as $f;
                if *self > -1.0 && *self < MAX_P1 {
                    return Some(float_to_int_unchecked!(*self => $u));
                }
            }
            None
        }
    )*}
}

macro_rules! impl_to_primitive_float {
    ($T:ident) => {
        impl ToPrimitive for $T {
            impl_to_primitive_float_to_signed_int! { $T:
                fn to_isize -> isize;
                fn to_i8 -> i8;
                fn to_i16 -> i16;
                fn to_i32 -> i32;
                fn to_i64 -> i64;
                fn to_i128 -> i128;
            }

            impl_to_primitive_float_to_unsigned_int! { $T:
                fn to_usize -> usize;
                fn to_u8 -> u8;
                fn to_u16 -> u16;
                fn to_u32 -> u32;
                fn to_u64 -> u64;
                fn to_u128 -> u128;
            }

            impl_to_primitive_float_to_float! { $T:
                fn to_f32 -> f32;
                fn to_f64 -> f64;
            }
        }
    };
}

impl_to_primitive_float!(f32);
impl_to_primitive_float!(f64);

impl ToPrimitive for f16 {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        Self::to_f32(*self).to_i64()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        Self::to_f32(*self).to_u64()
    }
    #[inline]
    fn to_i8(&self) -> Option<i8> {
        Self::to_f32(*self).to_i8()
    }
    #[inline]
    fn to_u8(&self) -> Option<u8> {
        Self::to_f32(*self).to_u8()
    }
    #[inline]
    fn to_i16(&self) -> Option<i16> {
        Self::to_f32(*self).to_i16()
    }
    #[inline]
    fn to_u16(&self) -> Option<u16> {
        Self::to_f32(*self).to_u16()
    }
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        Self::to_f32(*self).to_i32()
    }
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        Self::to_f32(*self).to_u32()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        Some(Self::to_f32(*self))
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        Some(Self::to_f64(*self))
    }
}

impl ToPrimitive for bf16 {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        Self::to_f32(*self).to_i64()
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        Self::to_f32(*self).to_u64()
    }
    #[inline]
    fn to_i8(&self) -> Option<i8> {
        Self::to_f32(*self).to_i8()
    }
    #[inline]
    fn to_u8(&self) -> Option<u8> {
        Self::to_f32(*self).to_u8()
    }
    #[inline]
    fn to_i16(&self) -> Option<i16> {
        Self::to_f32(*self).to_i16()
    }
    #[inline]
    fn to_u16(&self) -> Option<u16> {
        Self::to_f32(*self).to_u16()
    }
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        Self::to_f32(*self).to_i32()
    }
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        Self::to_f32(*self).to_u32()
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        Some(Self::to_f32(*self))
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        Some(Self::to_f64(*self))
    }
}

impl ToPrimitive for bool {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        Some(*self as i64)
    }
    #[inline]
    fn to_u64(&self) -> Option<u64> {
        Some(*self as u64)
    }
    #[inline]
    fn to_i8(&self) -> Option<i8> {
        Some(*self as i8)
    }
    #[inline]
    fn to_u8(&self) -> Option<u8> {
        Some(*self as u8)
    }
    #[inline]
    fn to_i16(&self) -> Option<i16> {
        Some(*self as i16)
    }
    #[inline]
    fn to_u16(&self) -> Option<u16> {
        Some(*self as u16)
    }
    #[inline]
    fn to_i32(&self) -> Option<i32> {
        Some(*self as i32)
    }
    #[inline]
    fn to_u32(&self) -> Option<u32> {
        Some(*self as u32)
    }
    #[inline]
    fn to_f32(&self) -> Option<f32> {
        Some(self.to_i64().unwrap() as f32)
    }
    #[inline]
    fn to_f64(&self) -> Option<f64> {
        Some(self.to_i64().unwrap() as f64)
    }
}

mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn to_primitive_float() {
        let f32_toolarge = 1e39f64;
        assert_eq!(f32_toolarge.to_f32(), Some(f32::INFINITY));
        assert_eq!((-f32_toolarge).to_f32(), Some(f32::NEG_INFINITY));
        assert_eq!((f32::MAX as f64).to_f32(), Some(f32::MAX));
        assert_eq!((-f32::MAX as f64).to_f32(), Some(-f32::MAX));
        assert_eq!(f64::INFINITY.to_f32(), Some(f32::INFINITY));
        assert_eq!((f64::NEG_INFINITY).to_f32(), Some(f32::NEG_INFINITY));
        assert!((f64::NAN).to_f32().map_or(false, |f| f.is_nan()));
    }

    #[test]
    fn to_primitive_int_unsigned_underflow() {
        assert_eq!((-1i8).to_u8(), None);
        assert_eq!((-1i8).to_u16(), None);
        assert_eq!((-1i8).to_u32(), None);
        assert_eq!((-1i8).to_u64(), None);
        assert_eq!((-1i8).to_u128(), None);
        assert_eq!((-1i8).to_usize(), None);
    }

    #[test]
    fn to_primitive_int_unsigned_overflow() {
        assert_eq!(256.to_u8(), None);
        assert_eq!(65_536.to_u16(), None);
        assert_eq!(4_294_967_296u64.to_u32(), None);
        assert_eq!(18_446_744_073_709_551_616u128.to_u64(), None);
    }

    #[test]
    fn to_primitive_int_to_float() {
        assert_eq!((-1).to_f32(), Some(-1.0));
        assert_eq!((-1).to_f64(), Some(-1.0));
        assert_eq!(255.to_f32(), Some(255.0));
        assert_eq!(65_535.to_f64(), Some(65_535.0));
    }

    #[test]
    fn to_primitive_float_to_int() {
        assert_eq!((-1.0).to_i8(), Some(-1));
        assert_eq!(1.0.to_u8(), Some(1));
        assert_eq!(1.8.to_u16(), Some(1));
        assert_eq!(123.456.to_u32(), Some(123));
    }
}
