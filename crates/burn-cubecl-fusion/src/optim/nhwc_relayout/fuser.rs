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
    fusers: Vec<TraceOperationFuser>,
    op: Option<OperationIr>,
    status: FuserStatus,
    device: R::Device,
    max_bindings: u32,
}

#[derive(Debug)]
pub enum RelayoutKind {
    NHWC,
    NCHW,
}

impl RelayoutKind {
    /// Build the stride relayout permutation for a tensor of the given `rank`.
    ///
    /// `permutation[original_dim]` is the physical position the dimension should occupy in the
    /// resulting memory layout (`0` = outermost, `rank - 1` = innermost / contiguous).
    ///
    /// This works for any rank, e.g. 3D (`N, C, L`), 4D (`N, C, H, W`) and 5D (`N, C, D, H, W`)
    /// tensors produced by 1D/2D/3D pooling and convolution operations.
    fn permutation(&self, rank: usize) -> Shape {
        let mut permutation = vec![0usize; rank];

        match self {
            // Channels-last: batch stays outermost, channels move to the innermost position and
            // the spatial dimensions shift one position towards the front.
            RelayoutKind::NHWC => {
                if rank >= 2 {
                    permutation[1] = rank - 1;
                    for dim in 2..rank {
                        permutation[dim] = dim - 1;
                    }
                }
            }
            // Channels-first / contiguous: identity permutation.
            RelayoutKind::NCHW => {
                for (dim, slot) in permutation.iter_mut().enumerate() {
                    *slot = dim;
                }
            }
        }

        Shape::from(permutation)
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
        // Pooling ops – `x`, `grad`, and `indices` are permuted to NHWC.
        AvgPool1d(op) => with_layout!(NHWC, [&op.x]),
        AvgPool2d(op) => with_layout!(NHWC, [&op.x]),
        AvgPool1dBackward(op) => with_layout!(NHWC, [&op.x, &op.grad]),
        AvgPool2dBackward(op) => with_layout!(NHWC, [&op.x, &op.grad]),
        AdaptiveAvgPool1d(op) => with_layout!(NHWC, [&op.x]),
        AdaptiveAvgPool2d(op) => with_layout!(NHWC, [&op.x]),
        AdaptiveAvgPool1dBackward(op) => with_layout!(NHWC, [&op.x, &op.grad]),
        AdaptiveAvgPool2dBackward(op) => with_layout!(NHWC, [&op.x, &op.grad]),
        MaxPool1d(op) => with_layout!(NHWC, [&op.x]),
        MaxPool1dWithIndices(op) => with_layout!(NHWC, [&op.x]),
        MaxPool1dWithIndicesBackward(op) => with_layout!(NHWC, [&op.x, &op.grad, &op.indices]),
        MaxPool2d(op) => with_layout!(NHWC, [&op.x]),
        MaxPool2dWithIndices(op) => with_layout!(NHWC, [&op.x]),
        MaxPool2dWithIndicesBackward(op) => with_layout!(NHWC, [&op.x, &op.grad, &op.indices]),

        // Interpolation – `x` and `grad` are permuted.
        Interpolate(op) => with_layout!(NHWC, [&op.x]),
        InterpolateBackward(op) => with_layout!(NHWC, [&op.x, &op.grad]),

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
        self.fusers
            .last()
            .map(|f| f.current_output_shape.clone())
            .unwrap_or_else(|| Shape::new([]))
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
            fusers: vec![fuser],
            device,
            max_bindings,
        }
    }

    /// Tries to fuse the given operation into an existing trace, or starts a new one if none fits.
    fn try_fuse(&mut self, operation: &OperationIr) {
        for trace in self.fusers.iter_mut() {
            if trace.can_fuse(operation) {
                trace.fuse(operation);
                self.status = trace.status();
                return;
            }
        }

        let mut new_trace = TraceOperationFuser::new(self.max_bindings, Self::settings());
        new_trace.fuse(operation);

        if let FuserStatus::Closed = new_trace.status() {
            self.status = FuserStatus::Closed;
        } else {
            self.status = new_trace.status();
            self.fusers.push(new_trace);
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
                        // Apply the relayout only to the trace that actually produces the tensor.
                        if let Some(trace) = self
                            .fusers
                            .iter_mut()
                            .find(|trace| trace.produces_output(tensor_ir.id))
                        {
                            trace.output_nhwc_layout(
                                tensor_ir,
                                shape.permutation(tensor_ir.shape.num_dims()),
                            );
                        }
                    }
                    self.status = FuserStatus::Closed;
                } else {
                    self.try_fuse(operation);
                }
            }
            _ => {
                self.try_fuse(operation);
            }
        };
    }

    fn finish(&mut self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let traces = self.fusers.iter_mut().map(|f| f.finish()).collect();
        let relayout =
            NHWCRelayoutOptimization::new(traces, client, self.device.clone(), self.len());
        CubeOptimization::NHWCRelayout(relayout)
    }

    fn reset(&mut self) {
        self.fusers.truncate(1);
        self.fusers[0].reset();
        self.op = None;
        self.status = self.fusers[0].status();
    }

    fn status(&self) -> FuserStatus {
        self.status
    }

    fn properties(&self) -> FuserProperties {
        let mut properties = FuserProperties::default();
        properties.ready = true;
        for trace in &self.fusers {
            let p = trace.properties();
            properties.score += p.score;
            properties.ready &= p.ready;
        }
        properties.score += match &self.op {
            Some(_) => 100, // TODO : proper score calculation
            None => 0,
        };
        properties.ready = properties.ready && self.op.is_some();
        properties
    }

    fn len(&self) -> usize {
        self.fusers.iter().map(|f| f.len()).sum::<usize>()
            + match &self.op {
                Some(_) => 1,
                None => 0,
            }
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
