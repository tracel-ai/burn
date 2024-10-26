use burn_fusion::OptimizationBuilder;

use crate::{
    fusion::{on_write::builder::FuseOnWriteBuilder, JitOptimization},
    JitRuntime,
};

use super::optimization::ElemwiseOptimization;

/// Fused element wise operations that are normally memory bound.
pub(crate) struct ElementWiseBuilder<R: JitRuntime> {
    builder: FuseOnWriteBuilder,
    device: R::Device,
}

impl<R: JitRuntime> ElementWiseBuilder<R> {
    pub fn new(device: R::Device) -> Self {
        Self {
            builder: FuseOnWriteBuilder::new(),
            device,
        }
    }
}

impl<R: JitRuntime> OptimizationBuilder<JitOptimization<R>> for ElementWiseBuilder<R> {
    fn register(&mut self, operation: &burn_tensor::repr::OperationDescription) {
        self.builder.register(operation)
    }

    fn build(&self) -> JitOptimization<R> {
        let client = R::client(&self.device);
        let trace = self.builder.build();
        let elementwise =
            ElemwiseOptimization::<R>::new(trace, client, self.device.clone(), self.len());

        JitOptimization::ElementWise2(elementwise)
    }

    fn reset(&mut self) {
        self.builder.reset()
    }

    fn status(&self) -> burn_fusion::OptimizationStatus {
        self.builder.status()
    }

    fn properties(&self) -> burn_fusion::OptimizationProperties {
        self.builder.properties()
    }

    fn len(&self) -> usize {
        self.builder.len()
    }
}
