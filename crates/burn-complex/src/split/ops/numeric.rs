use burn_tensor::cast::ToElement;
use burn_tensor::{Complex, Element, TensorMetadata};
use burn_tensor::{Float, Tensor, backend::Backend};

use crate::split::SplitComplexTensor;

// SplitComplexTensor + SplitComplexTensor
impl<B: Backend, const D: usize> core::ops::Add<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::add(self, rhs)
    }
}

// SplitComplexTensor + Tensor<B, D, Float> — adds real tensor to the real part
impl<B: Backend, const D: usize> core::ops::Add<Tensor<B, D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn add(self, rhs: Tensor<B, D, Float>) -> Self::Output {
        let prim = rhs.into_primitive().tensor();
        let device = B::float_device(&prim);
        let shape = prim.shape();
        let dtype = prim.dtype().into();
        let zeros = B::float_zeros(shape, &device, dtype);
        let complex_rhs = SplitComplexTensor::new(prim, zeros);
        Self::add(self, complex_rhs)
    }
}

// SplitComplexTensor + scalar (concrete types to avoid coherence conflict with ElementConversion)
macro_rules! impl_complex_tensor_add_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Add<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn add(self, rhs: $t) -> Self::Output {
                    Self::add_scalar(self, burn_tensor::Scalar::Float(rhs as f64))
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
        Self::add_scalar(self, burn_tensor::Scalar::Complex(rhs.to_complex64()))
    }
}
// Tensor - tensor
impl<B: Backend, const D: usize> core::ops::Sub<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::sub(self, rhs)
    }
}

// SplitComplexTensor - Tensor<B, D, Float>
impl<B: Backend, const D: usize> core::ops::Sub<Tensor<B, D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn sub(self, rhs: Tensor<B, D, Float>) -> Self::Output {
        let prim = rhs.into_primitive().tensor();
        let device = B::float_device(&prim);
        let shape = prim.shape();
        let dtype = prim.dtype().into();
        let zeros = B::float_zeros(shape, &device, dtype);
        let complex_rhs = SplitComplexTensor::new(prim, zeros);
        Self::sub(self, complex_rhs)
    }
}

// SplitComplexTensor - scalar
macro_rules! impl_complex_tensor_sub_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Sub<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn sub(self, rhs: $t) -> Self::Output {
                    Self::sub_scalar(self, burn_tensor::Scalar::Float(rhs as f64))
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
        Self::sub_scalar(self, burn_tensor::Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor * tensor
impl<B: Backend, const D: usize> core::ops::Mul<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::mul(self, rhs)
    }
}

// SplitComplexTensor * Tensor<B, D, Float>
impl<B: Backend, const D: usize> core::ops::Mul<Tensor<B, D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn mul(self, rhs: Tensor<B, D, Float>) -> Self::Output {
        let prim = rhs.into_primitive().tensor();
        let device = B::float_device(&prim);
        let shape = prim.shape();
        let dtype = prim.dtype().into();
        let zeros = B::float_zeros(shape, &device, dtype);
        let complex_rhs = SplitComplexTensor::new(prim, zeros);
        Self::mul(self, complex_rhs)
    }
}

// SplitComplexTensor * scalar
macro_rules! impl_complex_tensor_mul_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Mul<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn mul(self, rhs: $t) -> Self::Output {
                    Self::mul_scalar(self, burn_tensor::Scalar::Float(rhs as f64))
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
        Self::mul_scalar(self, burn_tensor::Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor / tensor
impl<B: Backend, const D: usize> core::ops::Div<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::div(self, rhs)
    }
}

// SplitComplexTensor / Tensor<B, D, Float>
impl<B: Backend, const D: usize> core::ops::Div<Tensor<B, D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn div(self, rhs: Tensor<B, D, Float>) -> Self::Output {
        let prim = rhs.into_primitive().tensor();
        let device = B::float_device(&prim);
        let shape = prim.shape();
        let dtype = prim.dtype().into();
        let zeros = B::float_zeros(shape, &device, dtype);
        let complex_rhs = SplitComplexTensor::new(prim, zeros);
        Self::div(self, complex_rhs)
    }
}

// SplitComplexTensor / scalar
macro_rules! impl_complex_tensor_div_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Div<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn div(self, rhs: $t) -> Self::Output {
                    Self::div_scalar(self, burn_tensor::Scalar::Float(rhs as f64))
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
        Self::div_scalar(self, burn_tensor::Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor % tensor
impl<B: Backend, const D: usize> core::ops::Rem<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self::remainder(self, rhs)
    }
}

// SplitComplexTensor % Tensor<B, D, Float>
impl<B: Backend, const D: usize> core::ops::Rem<Tensor<B, D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn rem(self, rhs: Tensor<B, D, Float>) -> Self::Output {
        let prim = rhs.into_primitive().tensor();
        let device = B::float_device(&prim);
        let shape = prim.shape();
        let dtype = prim.dtype().into();
        let zeros = B::float_zeros(shape, &device, dtype);
        let complex_rhs = SplitComplexTensor::new(prim, zeros);
        Self::remainder(self, complex_rhs)
    }
}

// SplitComplexTensor % scalar
macro_rules! impl_complex_tensor_rem_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Rem<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn rem(self, rhs: $t) -> Self::Output {
                    Self::remainder_scalar(self, burn_tensor::Scalar::Float(rhs as f64))
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
        Self::remainder_scalar(self, burn_tensor::Scalar::Complex(rhs.to_complex64()))
    }
}

impl<B: Backend, const D: usize> core::ops::Neg for SplitComplexTensor<B, D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::neg(self)
    }
}
