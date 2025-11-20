use super::optimization::ElemwiseOptimization;
use crate::{
    optim::CubeOptimization,
    shared::{
        builder::FuseOptimizationBuilder,
        ir::FuseType,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
};
use burn_fusion::OptimizationBuilder;
use cubecl::Runtime;

/// Fused element wise operations that are normally memory bound.
pub struct ElementWiseBuilder<R: Runtime> {
    builder: FuseOptimizationBuilder,
    device: R::Device,
}

impl<R: Runtime> Clone for ElementWiseBuilder<R> {
    fn clone(&self) -> Self {
        Self {
            builder: self.builder.clone(),
            device: self.device.clone(),
        }
    }
}

impl<R: Runtime> ElementWiseBuilder<R> {
    pub fn new(device: R::Device, bool_precision: FuseType) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware.max_bindings;

        Self {
            builder: FuseOptimizationBuilder::new(
                max_bindings,
                bool_precision,
                FuseSettings {
                    broadcast: true,
                    output_shape_updates: true,
                    inplace: true,
                    vectorization: VectorizationSetting::Activated,
                    ref_layout: RefLayoutSetting::Any,
                },
            ),
            device,
        }
    }
}

impl<R: Runtime> OptimizationBuilder<CubeOptimization<R>> for ElementWiseBuilder<R> {
    fn register(&mut self, operation: &burn_ir::OperationIr) {
        self.builder.register(operation);
    }

    fn build(&self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let trace = self.builder.build();
        let elementwise =
            ElemwiseOptimization::<R>::new(trace, client, self.device.clone(), self.len());

        CubeOptimization::ElementWise(elementwise)
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

    fn clone_dyn(&self) -> Box<dyn OptimizationBuilder<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
