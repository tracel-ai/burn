use burn_backend::ops::ComplexTensorOps;
use burn_std::ComplexScalar;
use burn_std::Element;
use burn_std::cast::ToElement;

use crate::Complex;
use crate::Float;

use crate::Tensor;

use crate::kind::Numeric;

use crate::ops::Numeric as _;
use crate::split::base::SplitBackend;
use crate::split::base::SplitTensor;

// Ideally we can separate out the numeric ops to those that aren't generalizable (i.e isn't just a series of linear ops) from those that are

// SplitTensor + SplitTensor
impl<const D: usize> core::ops::Add<Self> for SplitTensor<D, Complex> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        SplitBackend::complex_add(self.into(), rhs.into()).into()
    }
}

// SplitTensor + Tensor<D, Float> — adds real tensor to the real part
impl<const D: usize> core::ops::Add<Tensor<D, Float>> for SplitTensor<D, Complex> {
    type Output = Self;

    fn add(self, rhs: Tensor<D, Float>) -> Self::Output {
        self + Self::from_real(rhs)
    }
}

// SplitTensor + scalar (concrete types to avoid coherence conflict with ElementConversion)
macro_rules! impl_complex_tensor_add_scalar {
    ($($t:ty),*) => {
        $(
            impl<const D: usize> core::ops::Add<$t> for SplitTensor<D,Complex> {
                type Output = Self;

                fn add(self, rhs: $t) -> Self::Output {
                    Self::add_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
                }
            }
        )*
    }
}
impl_complex_tensor_add_scalar!(f32, f64, i32, i64, u32, u64);

impl<const D: usize, E: Element> core::ops::Add<ComplexScalar<E>> for SplitTensor<D, Complex> {
    type Output = Self;

    fn add(self, rhs: ComplexScalar<E>) -> Self::Output {
        Self::add_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}
// Tensor - tensor
impl<const D: usize> core::ops::Sub<Self> for SplitTensor<D, Complex> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        SplitBackend::complex_sub(self.into(), rhs.into()).into()
    }
}

// SplitTensor - Tensor<D, Float>
impl<const D: usize> core::ops::Sub<Tensor<D, Float>> for SplitTensor<D, Complex> {
    type Output = Self;

    fn sub(self, rhs: Tensor<D, Float>) -> Self::Output {
        self - Self::from_real(rhs)
    }
}

// SplitTensor - scalar
macro_rules! impl_complex_tensor_sub_scalar {
    ($($t:ty),*) => {
        $(
            impl<const D: usize> core::ops::Sub<$t> for SplitTensor<D,Complex> {
                type Output = Self;

                fn sub(self, rhs: $t) -> Self::Output {
                    Self::sub_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
                }
            }
        )*
    }
}
impl_complex_tensor_sub_scalar!(f32, f64, i32, i64, u32, u64);

impl<const D: usize, E: Element> core::ops::Sub<ComplexScalar<E>> for SplitTensor<D, Complex> {
    type Output = Self;

    fn sub(self, rhs: ComplexScalar<E>) -> Self::Output {
        Self::sub_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor * tensor
impl<const D: usize> core::ops::Mul<Self> for SplitTensor<D, Complex> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::mul(self, rhs)
    }
}

// SplitTensor * Tensor<D, K>
impl<const D: usize, K: Numeric> core::ops::Mul<Tensor<D, K>> for SplitTensor<D, Complex> {
    type Output = Self;

    fn mul(self, rhs: Tensor<D, K>) -> Self::Output {
        let prim = rhs.primitive;
        let [real, imag] = self.components;
        SplitTensor::new(K::mul(real, prim.clone()), K::mul(imag, prim))
    }
}

// SplitTensor * scalar
macro_rules! impl_complex_tensor_mul_scalar {
    ($($t:ty),*) => {
        $(
            impl<const D: usize> core::ops::Mul<$t> for SplitTensor<D,Complex> {
                type Output = Self;

                fn mul(self, rhs: $t) -> Self::Output {
                    Self::mul_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
                }
            }
        )*
    }
}
impl_complex_tensor_mul_scalar!(f32, f64, i32, i64, u32, u64);

impl<const D: usize, E: Element> core::ops::Mul<ComplexScalar<E>> for SplitTensor<D, Complex> {
    type Output = Self;

    fn mul(self, rhs: ComplexScalar<E>) -> Self::Output {
        Self::mul_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor / tensor
impl<const D: usize> core::ops::Div<Self> for SplitTensor<D, Complex> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        SplitBackend::complex_div(self.into(), rhs.into()).into()
    }
}

// SplitTensor / Tensor<D, Float>
impl<const D: usize> core::ops::Div<Tensor<D, Float>> for SplitTensor<D, Complex> {
    type Output = Self;

    fn div(self, rhs: Tensor<D, Float>) -> Self::Output {
        let prim = rhs.primitive;
        let [real, imag] = self.components;
        SplitTensor::new(Float::div(real, prim.clone()), Float::div(imag, prim))
    }
}

// SplitTensor / scalar
macro_rules! impl_complex_tensor_div_scalar {
    ($($t:ty),*) => {
        $(
            impl<const D: usize> core::ops::Div<$t> for SplitTensor<D,Complex> {
                type Output = Self;

                fn div(self, rhs: $t) -> Self::Output {
                    Self::div_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
                }
            }
        )*
    }
}
impl_complex_tensor_div_scalar!(f32, f64, i32, i64, u32, u64);

impl<const D: usize, E: Element> core::ops::Div<ComplexScalar<E>> for SplitTensor<D, Complex> {
    type Output = Self;

    fn div(self, rhs: ComplexScalar<E>) -> Self::Output {
        Self::div_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor % tensor
impl<const D: usize> core::ops::Rem<Self> for SplitTensor<D, Complex> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        SplitBackend::complex_remainder(self.into(), rhs.into()).into()
    }
}

// SplitTensor % Tensor<D, Float>
impl<const D: usize> core::ops::Rem<Tensor<D, Float>> for SplitTensor<D, Complex> {
    type Output = Self;

    fn rem(self, rhs: Tensor<D, Float>) -> Self::Output {
        let rhs = rhs.primitive;
        let [real, imag] = self.components;
        SplitTensor::new(
            Float::remainder(real, rhs.clone()),
            Float::remainder(imag, rhs),
        )
    }
}

// SplitTensor % scalar
macro_rules! impl_complex_tensor_rem_scalar {
    ($($t:ty),*) => {
        $(
            impl<const D: usize> core::ops::Rem<$t> for SplitTensor<D,Complex> {
                type Output = Self;

                fn rem(self, rhs: $t) -> Self::Output {
                    Self::remainder_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
                }
            }
        )*
    }
}
impl_complex_tensor_rem_scalar!(f32, f64, i32, i64, u32, u64);

impl<const D: usize, E: Element> core::ops::Rem<ComplexScalar<E>> for SplitTensor<D, Complex> {
    type Output = Self;

    fn rem(self, rhs: ComplexScalar<E>) -> Self::Output {
        Self::remainder_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

impl<const D: usize> core::ops::Neg for SplitTensor<D, Complex> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::neg(self)
    }
}
