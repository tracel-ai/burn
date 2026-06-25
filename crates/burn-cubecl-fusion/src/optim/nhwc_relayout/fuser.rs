use crate::{
    engine::{
        fuser::TraceOperationFuser,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::{CubeOptimization, nhwc_relayout::optimization::NHWCRelayoutOptimization},
};
use burn_fusion::{FuserProperties, FuserStatus, OperationFuser};
use burn_ir::{ModuleOperationIr, OperationIr};
use burn_std::Shape;
use cubecl::Runtime;

/// Fuses element wise operations.
#[derive(Clone)]
pub struct NHWCRelayoutFuser<R: Runtime> {
    fuser: TraceOperationFuser,
    op: Option<OperationIr>,
    status: FuserStatus,
    device: R::Device,
}

impl<R: Runtime> NHWCRelayoutFuser<R> {
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
            op: None,
            fuser,
            device,
        }
    }
}

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for NHWCRelayoutFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        if let FuserStatus::Closed = &self.status {
            return;
        }

        match operation {
            OperationIr::Module(ir) => {
                let op = match ir {
                    ModuleOperationIr::AvgPool1d(op) => Some(&op.x),
                    ModuleOperationIr::AvgPool2d(op) => Some(&op.x),
                    ModuleOperationIr::AvgPool1dBackward(op) => Some(&op.x),
                    ModuleOperationIr::AvgPool2dBackward(op) => Some(&op.x),
                    ModuleOperationIr::AdaptiveAvgPool1d(op) => Some(&op.x),
                    ModuleOperationIr::AdaptiveAvgPool2d(op) => Some(&op.x),
                    ModuleOperationIr::AdaptiveAvgPool1dBackward(op) => Some(&op.x),
                    ModuleOperationIr::AdaptiveAvgPool2dBackward(op) => Some(&op.x),
                    ModuleOperationIr::MaxPool1d(op) => Some(&op.x),
                    ModuleOperationIr::MaxPool1dWithIndices(op) => Some(&op.x),
                    ModuleOperationIr::MaxPool1dWithIndicesBackward(op) => Some(&op.x),
                    ModuleOperationIr::MaxPool2d(op) => Some(&op.x),
                    ModuleOperationIr::MaxPool2dWithIndices(op) => Some(&op.x),
                    ModuleOperationIr::MaxPool2dWithIndicesBackward(op) => Some(&op.x),
                    ModuleOperationIr::Interpolate(op) => Some(&op.x),
                    ModuleOperationIr::InterpolateBackward(op) => Some(&op.x),
                    _ => None,
                };

                if let Some(tensor_ir) = op {
                    self.op = Some(operation.clone());
                    self.fuser.output_nhwc_layout(tensor_ir);
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
        let relayout =
            NHWCRelayoutOptimization::new(trace, client, self.device.clone(), self.len());
        CubeOptimization::NHWCRelayout(relayout)
    }

    fn reset(&mut self) {
        self.fuser.reset();
        self.op = None;
        self.status = self.fuser.status();
    }

    fn status(&self) -> FuserStatus {
        self.status
    }

    fn properties(&self) -> FuserProperties {
        let mut properties = self.fuser.properties();
        properties.score += match &self.op {
            Some(_) => 100, // TODO : proper score calculation
            None => 0,
        };
        properties.ready = properties.ready && self.op.is_some();
        properties
    }

    fn len(&self) -> usize {
        self.fuser.len()
            + match &self.op {
                Some(_) => 1,
                None => 0,
            }
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
