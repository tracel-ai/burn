use super::optimization::{FusedMatmul, MatmulOptimization};
use crate::{
    engine::{fuser::TraceOperationFuser, settings::FuseSettings},
    optim::CubeOptimization,
    optim::matmul::args::MatmulArg,
};
use burn_fusion::{FuserStatus, OperationFuser};
use burn_ir::{BaseOperationIr, BinaryOpIr, FloatOperationIr, OperationIr, TensorIr, TensorStatus};
use burn_std::DType;

/// Fused element wise operations that are normally memory bound.
#[derive(Clone)]
pub struct MatmulFuser {
    fuser: TraceOperationFuser,
    fuser_fallback: TraceOperationFuser,
    device: cubecl::Device,
    matmul: Option<Product>,
}

/// The matmul the fuser anchored on, before and after its output takes its slot in the trace.
#[derive(Clone)]
enum Product {
    /// Its output not yet registered: a reshape of it may still change the tensor the routine
    /// writes, and the operations the fuser absorbed that way.
    Open {
        lhs: MatmulArg,
        rhs: MatmulArg,
        op: BinaryOpIr,
        output: TensorIr,
        reshapes: usize,
    },
    /// Its output registered, as the tensor the routine writes.
    Slotted(FusedMatmul),
}

impl Product {
    /// The operations the fuser claimed for the product: the matmul, then the reshapes of its
    /// output it absorbed.
    fn len(&self) -> usize {
        match self {
            Self::Open { reshapes, .. } => 1 + reshapes,
            Self::Slotted(matmul) => 1 + matmul.reshapes,
        }
    }
}

impl MatmulFuser {
    pub fn new(device: cubecl::Device) -> Self {
        let client = device.client();
        let props = client.properties();
        let max_bindings = props.hardware.max_bindings;
        let settings_matmul = FuseSettings {
            output_shape_updates: false,
            // An output aliasing an input reuses that input's buffer argument in the compiled
            // kernel, found by its position among the kernel's arguments. The routine's kernel
            // declares its own operands before the trace's, so the alias would name another
            // argument and the output would never be written.
            inplace: false,
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

    /// Absorb `operation` where it reshapes the product's output before the output takes its
    /// slot: the routine then writes the reshaped tensor. That is exact, since the routine writes
    /// its product at linear positions and a reshape moves no element's.
    ///
    /// Only where the reshape is the output's last use: an output something else still reads
    /// has to be written as it is. The fallback runs the reshape on its own, after the matmul.
    fn reshapes_the_output(&mut self, operation: &OperationIr) -> bool {
        let (
            Some(Product::Open {
                output, reshapes, ..
            }),
            OperationIr::BaseFloat(BaseOperationIr::Reshape(reshape)),
        ) = (self.matmul.as_mut(), operation)
        else {
            return false;
        };
        if reshape.input.id != output.id || reshape.input.status != TensorStatus::ReadWrite {
            return false;
        }
        *output = reshape.out.clone();
        *reshapes += 1;
        true
    }

    /// Register the product's output, as its last reshape left it, before the first operation
    /// of its tail. Idempotent.
    fn slot_output(&mut self) {
        self.matmul = match self.matmul.take() {
            Some(Product::Open {
                lhs,
                rhs,
                op,
                output,
                reshapes,
            }) => {
                let out = self.fuser.output_unhandled(&output);
                Some(Product::Slotted(FusedMatmul::new(
                    lhs, rhs, out, op, output, reshapes,
                )))
            }
            slotted => slotted,
        };
    }

    /// The fused matmul, once its output has its slot.
    fn fused_matmul(&self) -> &FusedMatmul {
        match &self.matmul {
            Some(Product::Slotted(matmul)) => matmul,
            _ => panic!("MatmulFuser::finish called before the tail claimed the product's output"),
        }
    }
}

impl OperationFuser<CubeOptimization> for MatmulFuser {
    fn fuse(&mut self, operation: &OperationIr) {
        if let FuserStatus::Closed = self.fuser.status() {
            return;
        }

        if self.matmul.is_none() {
            if let OperationIr::Float(_, FloatOperationIr::Matmul(op)) = operation {
                // Precision shouldn't be hardcoded but I don't know how to get float precision of the backend
                let lhs = self.matmul_arg(&op.lhs, op.out.dtype);
                let rhs = self.matmul_arg(&op.rhs, op.out.dtype);

                match (lhs, rhs) {
                    (Some(lhs), Some(rhs)) => {
                        self.matmul = Some(Product::Open {
                            lhs,
                            rhs,
                            op: op.clone().into(),
                            output: op.out.clone(),
                            reshapes: 0,
                        });
                    }
                    _ => {
                        self.fuser.close();
                        self.fuser_fallback.close();
                    }
                }
            } else {
                self.fuser.close();
                self.fuser_fallback.close();
            }
        } else {
            if self.reshapes_the_output(operation) {
                return;
            }
            self.slot_output();
            let can_register =
                self.fuser.can_fuse(operation) && self.fuser_fallback.can_fuse(operation);

            match can_register {
                true => {
                    self.fuser.fuse(operation);
                    self.fuser_fallback.fuse(operation);
                }
                false => {
                    self.fuser.close();
                    self.fuser_fallback.close();
                }
            };
        }
    }

    fn finish(&mut self) -> CubeOptimization {
        let client = self.device.client();
        let trace = self.fuser.finish();
        let trace_fallback = self.fuser_fallback.finish();

        let matmul = MatmulOptimization::new(
            trace,
            trace_fallback,
            client,
            self.device.clone(),
            self.len(),
            self.fused_matmul().clone(),
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
        // The matmul and the reshapes of its output aren't registered in the fused trace.
        self.fuser.len() + self.matmul.as_ref().map_or(1, Product::len)
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization>> {
        Box::new(self.clone())
    }
}
