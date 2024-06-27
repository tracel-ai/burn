use num_traits::{Float, PrimInt};

/// Quantization scheme/strategy.
#[derive(Debug, Clone)]
pub enum QuantizationStrategy {
    /// Per-tensor `int8` affine/asymmetric quantization.
    PerTensorAffineInt8(AffineQuantization<f32, i8>),
    /// Per-tensor `int8` symmetric quantization.
    PerTensorSymmetricInt8(SymmetricQuantization<f32>),
}

/// Affine quantization scheme.
#[derive(Debug, Clone)]
pub struct AffineQuantization<E: Float, Q: PrimInt> {
    /// The scaling factor.
    pub scale: E,
    /// The zero-point offset.
    pub offset: Q,
}

impl<E: Float, Q: PrimInt> AffineQuantization<E, Q> {
    /// Create a new affine quantization scheme for an input range `[alpha, beta]`.
    pub fn new(alpha: E, beta: E) -> Self {
        // Q range `[a, b]`
        let a = E::from(Q::min_value()).unwrap();
        let b = E::from(Q::max_value()).unwrap();

        // Compute scale and offset to convert a floating point value in range `[alpha, beta]` to the quantized range
        let range = beta - alpha;
        Self {
            scale: range / (b - a),
            offset: Q::from(E::round(((beta * a) - (alpha * b)) / range)).unwrap(),
        }
    }
}

/// Symmetric quantization scheme.
#[derive(Debug, Clone)]
pub struct SymmetricQuantization<E: Float> {
    /// The scaling factor.
    pub scale: E,
}

impl<E: Float> SymmetricQuantization<E> {
    /// Create a new symmetric quantization scheme for an input range `[alpha, beta]`.
    pub fn new<Q: PrimInt>(alpha: E, beta: E) -> Self {
        assert!(
            !Q::min_value().is_zero(),
            "Symmetric quantization is only valid for signed integers."
        );

        // Quantized range `[a, b]`
        let b = E::from(Q::max_value()).unwrap();
        let a = b.neg();

        // Compute scale to convert a floating point value in range `[-alpha, alpha]` to the quantized range
        let alpha = alpha.abs().max(beta.abs());
        Self {
            scale: (alpha + alpha) / (b - a),
        }
    }
}
