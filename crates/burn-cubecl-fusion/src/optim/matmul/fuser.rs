use super::optimization::{FusedMatmul, MatmulOptimization};
use crate::{
    engine::{fuser::TraceOperationFuser, settings::FuseSettings},
    optim::CubeOptimization,
    optim::matmul::args::MatmulArg,
};
use burn_fusion::{FuserStatus, OperationFuser};
use burn_ir::{FloatOperationIr, OperationIr, TensorIr};
use burn_std::DType;
use cubecl::Runtime;

/// Fused element wise operations that are normally memory bound.
pub struct MatmulFuser<R: Runtime> {
    fuser: TraceOperationFuser,
    fuser_fallback: TraceOperationFuser,
    device: R::Device,
    matmul: Option<FusedMatmul>,
}

impl<R: Runtime> Clone for MatmulFuser<R> {
    fn clone(&self) -> Self {
        Self {
            fuser: self.fuser.clone(),
            fuser_fallback: self.fuser_fallback.clone(),
            device: self.device.clone(),
            matmul: self.matmul.clone(),
        }
    }
}

impl<R: Runtime> MatmulFuser<R> {
    pub fn new(device: R::Device) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware.max_bindings;
        let settings_matmul = FuseSettings {
            output_shape_updates: false,
            ..Default::default()
        };
        let settings_fallback = FuseSettings::default();

        Self {
            fuser: TraceOperationFuser::new(max_bindings, settings_matmul),
            fuser_fallback: TraceOperationFuser::new(max_bindings, settings_fallback),
            device,
            matmul: None,
        }
    }

    /// Register a matmul operand, or return `None` when it can't be read by a fused kernel.
    fn matmul_arg(&mut self, tensor: &TensorIr, out: DType) -> Option<MatmulArg> {
        match tensor.dtype {
            DType::QFloat(scheme) => {
                let (data, scales) = self.fuser.input_quantized_unhandled(tensor)?;
                Some(MatmulArg::Quantized {
                    data,
                    scales,
                    precision: out.into(),
                    scheme,
                })
            }
            _ => Some(MatmulArg::Normal(self.fuser.input_unhandled(tensor))),
        }
    }
}

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for MatmulFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        if let FuserStatus::Closed = self.fuser.status() {
            return;
        }

        if self.matmul.is_none() {
            if let OperationIr::Float(_, FloatOperationIr::Matmul(op)) = operation {
                // Precision shouldn't be hardcoded but I don't know how to get float precision of the backend
                let lhs = self.matmul_arg(&op.lhs, op.out.dtype);
                let rhs = self.matmul_arg(&op.rhs, op.out.dtype);

                if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                    let out = self.fuser.output_unhandled(&op.out);

                    self.matmul = Some(FusedMatmul::new(
                        lhs,
                        rhs,
                        out,
                        op.clone().into(),
                        Default::default(),
                    ));
                } else {
                    self.fuser.close();
                    self.fuser_fallback.close();
                }
            } else {
                self.fuser.close();
                self.fuser_fallback.close();
            }
        } else {
            let can_register =
                self.fuser.can_fuse(operation) && self.fuser_fallback.can_fuse(operation);

            if can_register {
                self.fuser.fuse(operation);
                self.fuser_fallback.fuse(operation);
            } else {
                self.fuser.close();
                self.fuser_fallback.close();
            }
        }
    }

    fn finish(&mut self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let trace = self.fuser.finish();
        let trace_fallback = self.fuser_fallback.finish();

        let matmul = MatmulOptimization::new(
            trace,
            trace_fallback,
            client,
            self.device.clone(),
            self.len(),
            self.matmul.as_ref().unwrap().clone(),
        );

        CubeOptimization::new(matmul)
    }

    fn reset(&mut self) {
        self.fuser.reset();
        self.fuser_fallback.reset();
        self.matmul = None;
    }

    fn status(&self) -> burn_fusion::FuserStatus {
        self.fuser.status()
    }

    fn properties(&self) -> burn_fusion::FuserProperties {
        self.fuser.properties()
    }

    fn len(&self) -> usize {
        // Matmul operation isn't registered in the fuser
        self.fuser.len() + 1
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
