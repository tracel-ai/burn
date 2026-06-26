use crate::{
    engine::{
        fuser::TraceOperationFuser,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::{CubeOptimization, nhwc_relayout::optimization::NHWCRelayoutOptimization},
};
use burn_fusion::{FuserProperties, FuserStatus, OperationFuser};
use burn_ir::{ModuleOperationIr, OperationIr, TensorIr};
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

/// Build the stride relayout permutation for a tensor of the given `rank`.
fn permutation(rank: usize) -> Shape {
    let mut permutation = vec![0usize; rank];

    if rank >= 2 {
        permutation[1] = rank - 1;
        for dim in 2..rank {
            permutation[dim] = dim - 1;
        }
    }

    Shape::from(permutation)
}

/// Collect tensor that should be relayed out to NHWC for a given module operation.
fn nhwc_relayout_tensor(ir: &ModuleOperationIr) -> Option<&TensorIr> {
    use ModuleOperationIr::*;

    match ir {
        AvgPool1d(op) => Some(&op.x),
        AvgPool2d(op) => Some(&op.x),
        AdaptiveAvgPool1d(op) => Some(&op.x),
        AdaptiveAvgPool2d(op) => Some(&op.x),
        MaxPool1d(op) => Some(&op.x),
        MaxPool1dWithIndices(op) => Some(&op.x),
        MaxPool2d(op) => Some(&op.x),
        MaxPool2dWithIndices(op) => Some(&op.x),
        Interpolate(op) => Some(&op.x),

        _ => None,
    }
}

impl<R: Runtime> NHWCRelayoutFuser<R> {
    pub fn shape_id(&self) -> Shape {
        self.fuser.current_output_shape.clone()
    }

    fn settings() -> FuseSettings {
        FuseSettings {
            broadcast: false,
            output_shape_updates: false,
            inplace: false,
            vectorization: VectorizationSetting::Deactivated,
            ref_layout: RefLayoutSetting::Any,
        }
    }

    pub fn new(device: R::Device) -> Self {
        let client = R::client(&device);
        let max_bindings = client.properties().hardware.max_bindings;
        let fuser = TraceOperationFuser::new(max_bindings, Self::settings());

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
            OperationIr::Module(ir) if let Some(tensor) = nhwc_relayout_tensor(ir) => {
                self.op = Some(operation.clone());
                self.fuser
                    .output_nhwc_layout(tensor, permutation(tensor.shape.num_dims()));
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
