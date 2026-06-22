use crate::{
    engine::{
        fuser::TraceOperationFuser,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::{CubeOptimization, pooling::optimization::PoolingOptimization},
};
use burn_fusion::{FuserProperties, FuserStatus, OperationFuser};
use burn_ir::{ModuleOperationIr, OperationIr};
use burn_std::Shape;
use cubecl::Runtime;

/// Fuses element wise operations.
pub struct PoolingFuser<R: Runtime> {
    fuser: TraceOperationFuser,
    pooling_op: Option<OperationIr>,
    status: FuserStatus,
    device: R::Device,
}

impl<R: Runtime> Clone for PoolingFuser<R> {
    fn clone(&self) -> Self {
        Self {
            fuser: self.fuser.clone(),
            status: self.status.clone(),
            pooling_op: self.pooling_op.clone(),
            device: self.device.clone(),
        }
    }
}

impl<R: Runtime> PoolingFuser<R> {
    pub fn shape_id(&self) -> Shape {
        self.fuser.current_output_shape.clone()
    }
    pub fn new(device: R::Device) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware.max_bindings;
        let fuser = TraceOperationFuser::new(
            max_bindings,
            FuseSettings {
                broadcast: true,
                output_shape_updates: true,
                inplace: true,
                vectorization: VectorizationSetting::Activated,
                ref_layout: RefLayoutSetting::Any,
            },
        );

        Self {
            status: fuser.status(),
            pooling_op: None,
            fuser,
            device,
        }
    }
}

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for PoolingFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        if let FuserStatus::Closed = &self.status {
            return;
        }

        match operation {
            OperationIr::Module(ir) => {
                let is_pool = match ir {
                    ModuleOperationIr::AvgPool1d(_) => true,
                    ModuleOperationIr::AvgPool2d(_) => true,
                    ModuleOperationIr::AvgPool1dBackward(_) => true,
                    ModuleOperationIr::AvgPool2dBackward(_) => true,
                    ModuleOperationIr::AdaptiveAvgPool1d(_) => true,
                    ModuleOperationIr::AdaptiveAvgPool2d(_) => true,
                    ModuleOperationIr::AdaptiveAvgPool1dBackward(_) => true,
                    ModuleOperationIr::AdaptiveAvgPool2dBackward(_) => true,
                    ModuleOperationIr::MaxPool1d(_) => true,
                    ModuleOperationIr::MaxPool1dWithIndices(_) => true,
                    ModuleOperationIr::MaxPool1dWithIndicesBackward(_) => true,
                    ModuleOperationIr::MaxPool2d(_) => true,
                    ModuleOperationIr::MaxPool2dWithIndices(_) => true,
                    ModuleOperationIr::MaxPool2dWithIndicesBackward(_) => true,
                    _ => false,
                };

                if is_pool {
                    self.pooling_op = Some(operation.clone());
                }
                self.status = FuserStatus::Closed;
            }
            _ => {
                self.fuser.fuse(operation);
                self.status = self.fuser.status();
            }
        };
    }

    fn finish(&mut self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let trace = self.fuser.finish();
        let pooling = PoolingOptimization::new(trace, client, self.device.clone(), self.len());
        CubeOptimization::Pooling(pooling)
    }

    fn reset(&mut self) {
        self.fuser.reset();
        self.pooling_op = None;
        self.status = self.fuser.status();
    }

    fn status(&self) -> FuserStatus {
        self.status.clone()
    }

    fn properties(&self) -> FuserProperties {
        let mut properties = self.fuser.properties();
        properties.score += match &self.pooling_op {
            Some(_) => 100, // Let'sssssssss goooooooooooooooooooooooooooooooo
            None => 0,
        };
        properties.ready = properties.ready && self.pooling_op.is_some();
        properties
    }

    fn len(&self) -> usize {
        self.fuser.len()
            + match &self.pooling_op {
                Some(_) => 1,
                None => 0,
            }
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
