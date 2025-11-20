use burn_fusion::{OptimizationBuilder, OptimizationStatus};
use burn_ir::{FloatOperationIr, OperationIr};
use burn_tensor::DType;
use cubecl::Runtime;

use crate::{
    CubeOptimization,
    matmul::args::MatmulArg,
    shared::{builder::FuseOptimizationBuilder, ir::FuseType, settings::FuseSettings},
};

use super::optimization::{FusedMatmul, MatmulOptimization};

/// Fused element wise operations that are normally memory bound.
pub struct MatmulBuilder<R: Runtime> {
    builder: FuseOptimizationBuilder,
    builder_fallback: FuseOptimizationBuilder,
    device: R::Device,
    matmul: Option<FusedMatmul>,
}

impl<R: Runtime> Clone for MatmulBuilder<R> {
    fn clone(&self) -> Self {
        Self {
            builder: self.builder.clone(),
            builder_fallback: self.builder_fallback.clone(),
            device: self.device.clone(),
            matmul: self.matmul.clone(),
        }
    }
}

impl<R: Runtime> MatmulBuilder<R> {
    pub fn new(device: R::Device, bool_precision: FuseType) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware.max_bindings;
        let settings_matmul = FuseSettings {
            output_shape_updates: false,
            ..Default::default()
        };
        let settings_fallback = FuseSettings::default();

        Self {
            builder: FuseOptimizationBuilder::new(max_bindings, bool_precision, settings_matmul),
            builder_fallback: FuseOptimizationBuilder::new(
                max_bindings,
                bool_precision,
                settings_fallback,
            ),
            device,
            matmul: None,
        }
    }
}

impl<R: Runtime> OptimizationBuilder<CubeOptimization<R>> for MatmulBuilder<R> {
    fn register(&mut self, operation: &OperationIr) {
        if let OptimizationStatus::Closed = self.builder.status() {
            return;
        }

        if self.matmul.is_none() {
            if let OperationIr::Float(_, FloatOperationIr::Matmul(op)) = operation {
                // Precision shouldn't be hardcoded but I don't know how to get float precision of the backend
                let lhs = match op.lhs.dtype {
                    DType::QFloat(scheme) => {
                        let (data, scales) =
                            self.builder.input_quantized_unhandled(&op.lhs).unwrap();
                        MatmulArg::Quantized {
                            data,
                            scales,
                            precision: op.out.dtype.into(),
                            scheme,
                        }
                    }
                    _ => MatmulArg::Normal(self.builder.input_unhandled(&op.lhs)),
                };
                let rhs = match op.rhs.dtype {
                    DType::QFloat(scheme) => {
                        let (data, scales) =
                            self.builder.input_quantized_unhandled(&op.rhs).unwrap();
                        MatmulArg::Quantized {
                            data,
                            scales,
                            precision: op.out.dtype.into(),
                            scheme,
                        }
                    }
                    _ => MatmulArg::Normal(self.builder.input_unhandled(&op.rhs)),
                };

                let out = self.builder.output_unhandled(&op.out);

                self.matmul = Some(FusedMatmul::new(
                    lhs,
                    rhs,
                    out,
                    op.clone().into(),
                    Default::default(),
                ));
            } else {
                self.builder.close();
                self.builder_fallback.close();
            }
        } else {
            let can_register = self.builder.can_register(operation)
                && self.builder_fallback.can_register(operation);

            match can_register {
                true => {
                    self.builder.register(operation);
                    self.builder_fallback.register(operation);
                }
                false => {
                    self.builder.close();
                    self.builder_fallback.close();
                }
            };
        }
    }

    fn build(&self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let trace = self.builder.build();
        let trace_fallback = self.builder_fallback.build();

        let matmul = MatmulOptimization::<R>::new(
            trace,
            trace_fallback,
            client,
            self.device.clone(),
            self.len(),
            self.matmul.as_ref().unwrap().clone(),
        );

        CubeOptimization::Matmul(matmul)
    }

    fn reset(&mut self) {
        self.builder.reset();
        self.builder_fallback.reset();
        self.matmul = None;
    }

    fn status(&self) -> burn_fusion::OptimizationStatus {
        self.builder.status()
    }

    fn properties(&self) -> burn_fusion::OptimizationProperties {
        let mut properties = self.builder.properties();
        properties.score += 1;
        properties
    }

    fn len(&self) -> usize {
        // Matmul operation isn't registered in the builder
        self.builder.len() + 1
    }

    fn clone_dyn(&self) -> Box<dyn OptimizationBuilder<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
