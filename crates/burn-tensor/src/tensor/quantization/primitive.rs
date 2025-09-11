use cubecl_quant::scheme::QuantScheme;

use crate::quantization::{QuantAcc, QuantPropagation};

/// Quantized tensor primitive.
pub trait QTensorPrimitive {
    /// Returns the quantization settings for the given tensor.
    fn scheme(&self) -> &QuantScheme;
    /// The precision used for the accumulation in various kernels.
    fn acc_precision(&self) -> QuantAcc {
        QuantAcc::F32
    }
    /// How quantization is propagated during computation.
    fn propagation(&self) -> QuantPropagation {
        QuantPropagation::Inhibit
    }

    /// Returns the default tensor quantization scheme.
    fn default_scheme() -> QuantScheme {
        QuantScheme::default()
    }
}
