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

#[derive(Debug)]
pub enum RelayoutKind {
    NHWC,
    NCHW,
}

impl RelayoutKind {
    fn permutation(&self) -> Shape {
        match self {
            RelayoutKind::NHWC => Shape::new([0, 3, 1, 2]),
            RelayoutKind::NCHW => Shape::new([0, 1, 2, 3]),
        }
    }
}

macro_rules! with_layout {
    ($layout:expr, [$($tensor:expr),* $(,)?]) => {
        vec![$( ($tensor, $layout) ),*]
    };
}

/// Collect tensors that should be relayed out to NHWC for a given module operation.
///
/// Returns a non-empty vec if the operation internally permutes its inputs from NCHW to NHWC,
/// meaning the relayout can be fused into the preceding element-wise operations.
fn nhwc_relayout_tensors(ir: &ModuleOperationIr) -> Vec<(&TensorIr, RelayoutKind)> {
    use ModuleOperationIr::*;
    use RelayoutKind::{NCHW, NHWC};

    match ir {
        // Pooling ops – only `x` is permuted to NHWC.
        AvgPool1d(op) => with_layout!(NHWC, [&op.x]),
        AvgPool2d(op) => with_layout!(NHWC, [&op.x]),
        AvgPool1dBackward(op) => with_layout!(NHWC, [&op.x]),
        AvgPool2dBackward(op) => with_layout!(NHWC, [&op.x]),
        AdaptiveAvgPool1d(op) => with_layout!(NHWC, [&op.x]),
        AdaptiveAvgPool2d(op) => with_layout!(NHWC, [&op.x]),
        AdaptiveAvgPool1dBackward(op) => with_layout!(NHWC, [&op.x]),
        AdaptiveAvgPool2dBackward(op) => with_layout!(NHWC, [&op.x]),
        MaxPool1d(op) => with_layout!(NHWC, [&op.x]),
        MaxPool1dWithIndices(op) => with_layout!(NHWC, [&op.x]),
        MaxPool1dWithIndicesBackward(op) => with_layout!(NHWC, [&op.x]),
        MaxPool2d(op) => with_layout!(NHWC, [&op.x]),
        MaxPool2dWithIndices(op) => with_layout!(NHWC, [&op.x]),
        MaxPool2dWithIndicesBackward(op) => with_layout!(NHWC, [&op.x]),

        // Interpolation – only `x` is permuted.
        Interpolate(op) => with_layout!(NHWC, [&op.x]),
        InterpolateBackward(op) => with_layout!(NHWC, [&op.x]),

        // Conv forward – both input and weight are permuted to NHWC.
        Conv1d(op) => with_layout!(NHWC, [&op.x, &op.weight]),
        Conv2d(op) => with_layout!(NHWC, [&op.x, &op.weight]),
        Conv3d(op) => with_layout!(NHWC, [&op.x, &op.weight]),

        // Conv X backward – output_grad and weight are permuted.
        Conv1dXBackward(op) => with_layout!(NHWC, [&op.output_grad, &op.weight]),
        Conv2dXBackward(op) => with_layout!(NHWC, [&op.output_grad, &op.weight]),
        Conv3dXBackward(op) => with_layout!(NHWC, [&op.output_grad, &op.weight]),

        // Conv weight backward – input and output_grad are permuted.
        Conv1dWeightBackward(op) => with_layout!(NHWC, [&op.x, &op.output_grad]),
        Conv2dWeightBackward(op) => with_layout!(NHWC, [&op.x, &op.output_grad]),
        Conv3dWeightBackward(op) => with_layout!(NHWC, [&op.x, &op.output_grad]),

        Conv1dBiasBackward(op) => with_layout!(NCHW, [&op.output_grad]),
        Conv2dBiasBackward(op) => with_layout!(NCHW, [&op.output_grad]),
        Conv3dBiasBackward(op) => with_layout!(NCHW, [&op.output_grad]),
        ConvTranspose1d(op) => with_layout!(NCHW, [&op.x, &op.weight]),
        ConvTranspose2d(op) => with_layout!(NCHW, [&op.x, &op.weight]),
        ConvTranspose3d(op) => with_layout!(NCHW, [&op.x, &op.weight]),

        // Deformable conv2d – input, offset, and weight are all made contiguous.
        DeformableConv2d(op) => {
            let mut tensors = with_layout!(NCHW, [&op.x, &op.offset, &op.weight]);
            if let Some(mask) = &op.mask {
                tensors.push((mask, NCHW));
            }
            tensors
        }
        DeformableConv2dBackward(op) => {
            let mut tensors = with_layout!(NCHW, [&op.x, &op.offset, &op.weight, &op.out_grad]);
            if let Some(mask) = &op.mask {
                tensors.push((mask, NCHW));
            }
            tensors
        }
        _ => vec![],
    }
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
                let tensors = nhwc_relayout_tensors(ir);

                if !tensors.is_empty() {
                    self.op = Some(operation.clone());
                    for (tensor_ir, shape) in tensors {
                        self.fuser
                            .output_nhwc_layout(tensor_ir, shape.permutation());
                    }
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
