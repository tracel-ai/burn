use core::cmp::Ordering;

use serde::{Deserialize, Serialize};

trait Quantization {
    fn quantize<E, Q>(&self, values: &[E]) -> Vec<Q>;
    fn dequantize<E, Q>(&self, values: &[Q]) -> Vec<E>;
}

/// Affine quantization scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffineQuantization {
    /// The scaling factor.
    pub scale: f32,
    /// The zero-point offset.
    pub offset: f32,
}

impl Eq for AffineQuantization {}

impl PartialEq for AffineQuantization {
    fn eq(&self, other: &Self) -> bool {
        match (
            self.scale.total_cmp(&other.scale),
            self.offset.total_cmp(&other.offset),
        ) {
            (Ordering::Equal, Ordering::Equal) => true,
            _ => false,
        }
    }
}

impl Quantization for AffineQuantization {
    fn quantize<E, Q>(&self, values: &[E]) -> Vec<Q> {
        todo!()
    }

    fn dequantize<E, Q>(&self, values: &[Q]) -> Vec<E> {
        todo!()
    }
}

/// Symmetric quantization scheme.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetricQuantization {
    /// The scaling factor.
    pub scale: f32,
}

impl Eq for SymmetricQuantization {}

impl PartialEq for SymmetricQuantization {
    fn eq(&self, other: &Self) -> bool {
        match self.scale.total_cmp(&other.scale) {
            Ordering::Equal => true,
            _ => false,
        }
    }
}

impl Quantization for SymmetricQuantization {
    fn quantize<E, Q>(&self, values: &[E]) -> Vec<Q> {
        todo!()
    }

    fn dequantize<E, Q>(&self, values: &[Q]) -> Vec<E> {
        todo!()
    }
}

/// Quantization scheme/strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationStrategy {
    /// Affine/asymmetric quantization.
    Affine(AffineQuantization),
    /// Symmetric quantization.
    Symmetric(SymmetricQuantization),
}

impl QuantizationStrategy {
    /// Convert the values to a lower precision data type.
    pub fn quantize<E, Q>(&self, values: &[E]) -> Vec<Q> {
        match self {
            Self::Affine(m) => m.quantize(values),
            Self::Symmetric(m) => m.quantize(values),
        }
    }

    /// Convert the values back to a higher precision data type.
    pub fn dequantize<E, Q>(&self, values: &[Q]) -> Vec<E> {
        match self {
            Self::Affine(m) => m.dequantize(values),
            Self::Symmetric(m) => m.dequantize(values),
        }
    }
}
