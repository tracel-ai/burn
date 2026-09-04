use super::optimization::ElemwiseOptimization;
use crate::{
    engine::{
        fuser::TraceOperationFuser,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::CubeOptimization,
};
use burn_fusion::OperationFuser;
use burn_std::Shape;

/// Fuses element wise operations.
pub struct ElementWiseFuser {
    fuser: TraceOperationFuser,
    device: cubecl::Device,
}

impl Clone for ElementWiseFuser {
    fn clone(&self) -> Self {
        Self {
            fuser: self.fuser.clone(),
            device: self.device.clone(),
        }
    }
}

impl ElementWiseFuser {
    pub fn shape_id(&self) -> Shape {
        self.fuser.current_output_shape.clone()
    }
    pub fn new(device: cubecl::Device) -> Self {
        let client = device.client();
        let props = client.properties();
        let max_bindings = props.hardware.max_bindings;

        Self {
            fuser: TraceOperationFuser::new(
                max_bindings,
                FuseSettings {
                    broadcast: true,
                    output_shape_updates: true,
                    inplace: true,
                    vectorization: VectorizationSetting::Activated,
                    ref_layout: RefLayoutSetting::Any,
                    // The elementwise runner reads and writes every operand through
                    // the generic fused paths, so it is free to iterate in whatever
                    // order its inputs already sit in.
                    choose_output_layout: true,
                },
            ),
            device,
        }
    }
}

impl OperationFuser<CubeOptimization> for ElementWiseFuser {
    fn fuse(&mut self, operation: &burn_ir::OperationIr) {
        self.fuser.fuse(operation);
    }

    fn finish(&mut self) -> CubeOptimization {
        let client = self.device.client();
        let trace = self.fuser.finish();
        let elementwise = ElemwiseOptimization::new(trace, client, self.device.clone(), self.len());

        CubeOptimization::new(elementwise)
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

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization>> {
        Box::new(self.clone())
    }
}
