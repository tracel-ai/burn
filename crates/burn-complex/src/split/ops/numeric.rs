use burn_backend::Backend;
use burn_std::Scalar;
use burn_std::cast::ToElement;
use burn_std::{Complex, Element};
use burn_tensor::{Float, Tensor};

use crate::split::SplitComplexTensor;

// SplitComplexTensor + SplitComplexTensor
impl<B: Backend, const D: usize> core::ops::Add<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::add(self, rhs)
    }
}

// SplitComplexTensor + Tensor<D, Float> — adds real tensor to the real part
impl<B: Backend, const D: usize> core::ops::Add<Tensor<D>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn add(self, rhs: Tensor<D>) -> Self::Output {
        let prim = rhs.tensor();
        let (real, imag) = self.into_parts();
        SplitComplexTensor::new(real + prim, imag)
    }
}

// SplitComplexTensor + scalar (concrete types to avoid coherence conflict with ElementConversion)
macro_rules! impl_complex_tensor_add_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Add<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn add(self, rhs: $t) -> Self::Output {
                    Self::add_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_add_scalar!(f32, f64, i32, i64, u32, u64);

impl<B: Backend, const D: usize, E: Element> core::ops::Add<Complex<E>>
    for SplitComplexTensor<B, D>
{
    type Output = Self;

    fn add(self, rhs: Complex<E>) -> Self::Output {
        Self::add_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}
// Tensor - tensor
impl<B: Backend, const D: usize> core::ops::Sub<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::sub(self, rhs)
    }
}

// SplitComplexTensor - Tensor<D, Float>
impl<B: Backend, const D: usize> core::ops::Sub<Tensor<D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn sub(self, rhs: Tensor<D, Float>) -> Self::Output {
        let prim = rhs.tensor();

        let (real, imag) = self.into_parts();
        SplitComplexTensor::new(real - prim, imag)
    }
}

// SplitComplexTensor - scalar
macro_rules! impl_complex_tensor_sub_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Sub<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn sub(self, rhs: $t) -> Self::Output {
                    Self::sub_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_sub_scalar!(f32, f64, i32, i64, u32, u64);

impl<B: Backend, const D: usize, E: Element> core::ops::Sub<Complex<E>>
    for SplitComplexTensor<B, D>
{
    type Output = Self;

    fn sub(self, rhs: Complex<E>) -> Self::Output {
        Self::sub_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor * tensor
impl<B: Backend, const D: usize> core::ops::Mul<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::mul(self, rhs)
    }
}

// SplitComplexTensor * Tensor<D, Float>
impl<B: Backend, const D: usize> core::ops::Mul<Tensor<D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn mul(self, rhs: Tensor<D, Float>) -> Self::Output {
        let prim = rhs.tensor();
        let (real, imag) = self.into_parts();
        SplitComplexTensor::new(real * prim, imag * prim)
    }
}

// SplitComplexTensor * scalar
macro_rules! impl_complex_tensor_mul_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Mul<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn mul(self, rhs: $t) -> Self::Output {
                    Self::mul_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_mul_scalar!(f32, f64, i32, i64, u32, u64);

impl<B: Backend, const D: usize, E: Element> core::ops::Mul<Complex<E>>
    for SplitComplexTensor<B, D>
{
    type Output = Self;

    fn mul(self, rhs: Complex<E>) -> Self::Output {
        Self::mul_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor / tensor
impl<B: Backend, const D: usize> core::ops::Div<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::div(self, rhs)
    }
}

// SplitComplexTensor / Tensor<D, Float>
impl<B: Backend, const D: usize> core::ops::Div<Tensor<D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn div(self, rhs: Tensor<D, Float>) -> Self::Output {
        let prim = rhs.tensor();
        let (real, imag) = self.into_parts();
        SplitComplexTensor::new(real / prim, imag / prim)
    }
}

// SplitComplexTensor / scalar
macro_rules! impl_complex_tensor_div_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Div<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn div(self, rhs: $t) -> Self::Output {
                    Self::div_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_div_scalar!(f32, f64, i32, i64, u32, u64);

impl<B: Backend, const D: usize, E: Element> core::ops::Div<Complex<E>>
    for SplitComplexTensor<B, D>
{
    type Output = Self;

    fn div(self, rhs: Complex<E>) -> Self::Output {
        Self::div_scalar(self, Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor % tensor
impl<B: Backend, const D: usize> core::ops::Rem<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self::remainder(self, rhs)
    }
}

// SplitComplexTensor % Tensor<D, Float>
impl<B: Backend, const D: usize> core::ops::Rem<Tensor<D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn rem(self, rhs: Tensor<D, Float>) -> Self::Output {
        let prim = rhs.tensor();
        let (real, imag) = self.into_parts();
        SplitComplexTensor::new(real % prim, imag % prim)
    }
}

// SplitComplexTensor % scalar
macro_rules! impl_complex_tensor_rem_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Rem<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn rem(self, rhs: $t) -> Self::Output {
                    Self::remainder_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_rem_scalar!(f32, f64, i32, i64, u32, u64);

impl<B: Backend, const D: usize, E: Element> core::ops::Rem<Complex<E>>
    for SplitComplexTensor<B, D>
{
    type Output = Self;

    fn rem(self, rhs: Complex<E>) -> Self::Output {
        Self::remainder_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

impl<B: Backend, const D: usize> core::ops::Neg for SplitComplexTensor<B, D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::neg(self)
    }
}
