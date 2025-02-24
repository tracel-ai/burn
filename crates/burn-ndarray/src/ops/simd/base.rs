/// Whether SIMD instructions are worth using
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
pub fn should_use_simd(len: usize) -> bool {
    len >= 128
}

/// Whether SIMD instructions are worth using
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
pub fn should_use_simd(_len: usize) -> bool {
    false
}

pub trait MinMax {
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}

macro_rules! impl_minmax {
    ($ty: ty) => {
        impl MinMax for $ty {
            fn min(self, other: Self) -> Self {
                Ord::min(self, other)
            }
            fn max(self, other: Self) -> Self {
                Ord::max(self, other)
            }
        }
    };
    ($($ty: ty),*) => {
        $(impl_minmax!($ty);)*
    }
}

impl_minmax!(u8, i8, u16, i16, u32, i32, u64, i64);

impl MinMax for f32 {
    fn min(self, other: Self) -> Self {
        self.min(other)
    }

    fn max(self, other: Self) -> Self {
        self.max(other)
    }
}

impl MinMax for f64 {
    fn min(self, other: Self) -> Self {
        self.min(other)
    }

    fn max(self, other: Self) -> Self {
        self.max(other)
    }
}
