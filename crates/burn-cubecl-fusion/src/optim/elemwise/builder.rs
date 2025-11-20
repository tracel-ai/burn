use super::optimization::ElemwiseOptimization;
use crate::{
    engine::{
        builder::FuseTraceCompiler,
        ir::FuseType,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::CubeOptimization,
};
use burn_fusion::OperationFuser;
use cubecl::Runtime;

/// Fused element wise operations that are normally memory bound.
pub struct ElementWiseBuilder<R: Runtime> {
    builder: FuseTraceCompiler,
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
            builder: FuseTraceCompiler::new(
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

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for ElementWiseBuilder<R> {
    fn fuse(&mut self, operation: &burn_ir::OperationIr) {
        self.builder.fuse(operation);
    }

    fn finish(&self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let trace = self.builder.finish();
        let elementwise =
            ElemwiseOptimization::<R>::new(trace, client, self.device.clone(), self.len());

        CubeOptimization::ElementWise(elementwise)
    }

    fn reset(&mut self) {
        self.builder.reset()
    }

    fn status(&self) -> burn_fusion::FuserStatus {
        self.builder.status()
    }

    fn properties(&self) -> burn_fusion::FuserProperties {
        self.builder.properties()
    }

    fn len(&self) -> usize {
        self.builder.len()
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
