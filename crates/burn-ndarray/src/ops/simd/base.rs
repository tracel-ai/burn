use core::{marker::PhantomData, mem::MaybeUninit};

use macerator::{Arch, Scalar, Simd};
use ndarray::{ArcArray, ArrayD, IxDyn, ShapeBuilder};

/// Whether SIMD instructions are worth using
#[cfg(all(
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32",
        target_arch = "loongarch64"
    ),
    not(test)
))]
pub fn should_use_simd(len: usize) -> bool {
    len >= 32
}

/// Whether SIMD instructions are worth using
#[cfg(all(
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32",
        target_arch = "loongarch64"
    )),
    not(test)
))]
pub fn should_use_simd(_len: usize) -> bool {
    false
}

#[cfg(test)]
pub fn should_use_simd(_len: usize) -> bool {
    true
}

pub(crate) fn lanes<E: Scalar>() -> usize {
    #[allow(non_camel_case_types)]
    struct lanes<__T0>(__T0);

    impl<E: Scalar> ::macerator::WithSimd for lanes<PhantomData<E>> {
        type Output = usize;
        #[inline(always)]
        fn with_simd<__S: ::macerator::Simd>(self) -> <Self as ::macerator::WithSimd>::Output {
            let Self(__ty) = self;
            #[allow(unused_unsafe)]
            unsafe {
                lanes_simd::<__S, E>(__ty)
            }
        }
    }
    (Arch::new()).dispatch(lanes(PhantomData::<E>))
}

fn lanes_simd<S: Simd, E: Scalar>(_ty: PhantomData<E>) -> usize {
    E::lanes::<S>()
}

pub(crate) fn uninit_array_like<In, Out>(reference: &ArcArray<In, IxDyn>) -> ArrayD<Out> {
    let shape = reference.raw_dim();
    let strides = reference.strides();
    let strides = strides.iter().map(|it| *it as usize).collect::<Vec<_>>();
    let shape_strides = shape.strides(IxDyn(&strides));
    let size = reference.len();
    let mut out_data: Vec<MaybeUninit<Out>> = Vec::with_capacity(size);
    unsafe { out_data.set_len(size) };
    unsafe { ArrayD::from_shape_vec_unchecked(shape_strides, out_data).assume_init() }
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
