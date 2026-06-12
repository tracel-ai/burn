use burn_core as burn;

use burn::Tensor;
use burn::module::Module;

/// Identity Module.
#[derive(Module, Default, Debug)]
pub struct Identity;

impl Identity {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }

    /// Forward pass, returns the input tensor.
    pub fn forward<const R: usize>(&self, input: Tensor<R>) -> Tensor<R> {
        input
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let layer = Identity::new();

        assert_eq!(alloc::format!("{layer}"), "Identity");
    }
}
