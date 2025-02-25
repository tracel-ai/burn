use macerator::{SimdExt, Vectorizable};
use pulp::Simd;

/// Whether SIMD instructions are worth using
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
pub fn should_use_simd(len: usize) -> bool {
    len >= 32
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

type Vec4<T> = (T, T, T, T);

pub(crate) unsafe fn load4<S: Simd, T: Vectorizable>(simd: S, ptr: *const T) -> Vec4<T::Vector<S>> {
    let s0 = simd.vload(ptr);
    let s1 = simd.vload(ptr.add(T::lanes::<S>()));
    let s2 = simd.vload(ptr.add(2 * T::lanes::<S>()));
    let s3 = simd.vload(ptr.add(3 * T::lanes::<S>()));
    (s0, s1, s2, s3)
}

pub(crate) unsafe fn load4_unaligned<S: Simd, T: Vectorizable>(
    simd: S,
    ptr: *const T,
) -> Vec4<T::Vector<S>> {
    let s0 = simd.vload_unaligned(ptr);
    let s1 = simd.vload_unaligned(ptr.add(T::lanes::<S>()));
    let s2 = simd.vload_unaligned(ptr.add(2 * T::lanes::<S>()));
    let s3 = simd.vload_unaligned(ptr.add(3 * T::lanes::<S>()));
    (s0, s1, s2, s3)
}

pub(crate) unsafe fn load2<S: Simd, T: Vectorizable>(
    simd: S,
    ptr: *const T,
) -> (T::Vector<S>, T::Vector<S>) {
    let s0 = simd.vload(ptr);
    let s1 = simd.vload(ptr.add(T::lanes::<S>()));
    (s0, s1)
}

pub(crate) unsafe fn store4<S: Simd, T: Vectorizable>(
    simd: S,
    ptr: *mut T,
    s0: T::Vector<S>,
    s1: T::Vector<S>,
    s2: T::Vector<S>,
    s3: T::Vector<S>,
) {
    simd.vstore(ptr, s0);
    simd.vstore(ptr.add(T::lanes::<S>()), s1);
    simd.vstore(ptr.add(2 * T::lanes::<S>()), s2);
    simd.vstore(ptr.add(3 * T::lanes::<S>()), s3);
}

pub(crate) unsafe fn store4_unaligned<S: Simd, T: Vectorizable>(
    simd: S,
    ptr: *mut T,
    s0: T::Vector<S>,
    s1: T::Vector<S>,
    s2: T::Vector<S>,
    s3: T::Vector<S>,
) {
    simd.vstore_unaligned(ptr, s0);
    simd.vstore_unaligned(ptr.add(T::lanes::<S>()), s1);
    simd.vstore_unaligned(ptr.add(2 * T::lanes::<S>()), s2);
    simd.vstore_unaligned(ptr.add(3 * T::lanes::<S>()), s3);
}

pub(crate) unsafe fn store2<S: Simd, T: Vectorizable>(
    simd: S,
    ptr: *mut T,
    s0: T::Vector<S>,
    s1: T::Vector<S>,
) {
    simd.vstore(ptr, s0);
    simd.vstore(ptr.add(T::lanes::<S>()), s1);
}

pub(crate) unsafe fn store2_unaligned<S: Simd, T: Vectorizable>(
    simd: S,
    ptr: *mut T,
    s0: T::Vector<S>,
    s1: T::Vector<S>,
) {
    simd.vstore_unaligned(ptr, s0);
    simd.vstore_unaligned(ptr.add(T::lanes::<S>()), s1);
}
