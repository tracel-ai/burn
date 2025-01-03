use cubecl::prelude::Numeric;

/// Implicit convolution spec definiting each element types used in the computation.
pub trait ConvPrecision: Send + Sync + Clone + 'static {
    /// Element type of each input and output tensor of the kernel.
    type EG: Numeric;
    /// Element type of the intermediate representation of the inputs.
    type ES: Numeric;
    /// Element type of the intermediate representation of the output accumulator.
    type EA: Numeric;
}

impl<EG: Numeric, ES: Numeric, EA: Numeric> ConvPrecision for (EG, ES, EA) {
    type EG = EG;
    type ES = ES;
    type EA = EA;
}
