use crate::{
    engine::{
        fuser::TraceOperationFuser,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
        trace::OutputLayout,
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
                broadcast: false,
                output_shape_updates: false,
                inplace: false,
                vectorization: VectorizationSetting::Deactivated,
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
                let layout_info = match ir {
                    ModuleOperationIr::AvgPool1d(op) => Some((&op.x, 2)),
                    ModuleOperationIr::AvgPool2d(op) => Some((&op.x, 3)),
                    ModuleOperationIr::AvgPool1dBackward(op) => Some((&op.x, 2)),
                    ModuleOperationIr::AvgPool2dBackward(op) => Some((&op.x, 3)),
                    ModuleOperationIr::AdaptiveAvgPool1d(op) => Some((&op.x, 2)),
                    ModuleOperationIr::AdaptiveAvgPool2d(op) => Some((&op.x, 3)),
                    ModuleOperationIr::AdaptiveAvgPool1dBackward(op) => Some((&op.x, 2)),
                    ModuleOperationIr::AdaptiveAvgPool2dBackward(op) => Some((&op.x, 3)),
                    ModuleOperationIr::MaxPool1d(op) => Some((&op.x, 2)),
                    ModuleOperationIr::MaxPool1dWithIndices(op) => Some((&op.x, 2)),
                    ModuleOperationIr::MaxPool1dWithIndicesBackward(op) => Some((&op.x, 2)),
                    ModuleOperationIr::MaxPool2d(op) => Some((&op.x, 3)),
                    ModuleOperationIr::MaxPool2dWithIndices(op) => Some((&op.x, 3)),
                    ModuleOperationIr::MaxPool2dWithIndicesBackward(op) => Some((&op.x, 3)),
                    _ => None,
                };

                if let Some((x, swap_dim)) = layout_info {
                    self.pooling_op = Some(operation.clone());
                    self.fuser
                        .output_layout(x, OutputLayout::SwapDims(1, swap_dim));
                    self.status = FuserStatus::Closed;
                } else {
                    self.fuser.fuse(operation);
                    self.status = self.fuser.status();
                }
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
            Some(_) => 100, // TODO : proper score calculation
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
