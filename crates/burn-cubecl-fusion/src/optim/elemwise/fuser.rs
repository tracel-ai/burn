use super::optimization::ElemwiseOptimization;
use crate::{
    engine::{
        fuser::TraceFuser,
        ir::FuseType,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::CubeOptimization,
};
use burn_fusion::OperationFuser;
use cubecl::Runtime;

/// Fuses element wise operations.
pub struct ElementWiseFuser<R: Runtime> {
    fuser: TraceFuser,
    device: R::Device,
}

impl<R: Runtime> Clone for ElementWiseFuser<R> {
    fn clone(&self) -> Self {
        Self {
            fuser: self.fuser.clone(),
            device: self.device.clone(),
        }
    }
}

impl<R: Runtime> ElementWiseFuser<R> {
    pub fn new(device: R::Device, bool_precision: FuseType) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware.max_bindings;

        Self {
            fuser: TraceFuser::new(
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

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for ElementWiseFuser<R> {
    fn fuse(&mut self, operation: &burn_ir::OperationIr) {
        self.fuser.fuse(operation);
    }

    fn finish(&self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let trace = self.fuser.finish();
        let elementwise =
            ElemwiseOptimization::<R>::new(trace, client, self.device.clone(), self.len());

        CubeOptimization::ElementWise(elementwise)
    }

    fn reset(&mut self) {
        self.fuser.reset()
    }

    fn status(&self) -> burn_fusion::FuserStatus {
        self.fuser.status()
    }

    fn properties(&self) -> burn_fusion::FuserProperties {
        self.fuser.properties()
    }

    fn len(&self) -> usize {
        self.fuser.len()
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
