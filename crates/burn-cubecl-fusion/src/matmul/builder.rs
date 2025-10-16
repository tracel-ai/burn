use burn_fusion::{OptimizationBuilder, OptimizationStatus};
use burn_ir::{FloatOperationIr, OperationIr};
use cubecl::Runtime;

use crate::{
    CubeOptimization,
    matmul::args::MatmulArg,
    shared::{
        builder::FuseOptimizationBuilder,
        ir::FusePrecision,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
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
    pub fn new(device: R::Device, bool_precision: FusePrecision) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware.max_bindings;
        let settings = FuseSettings {
            broadcast: true,
            output_shape_updates: false,
            inplace: true,
            vectorization: VectorizationSetting::Activated,
            ref_layout: RefLayoutSetting::Any,
        };

        Self {
            builder: FuseOptimizationBuilder::new(max_bindings, bool_precision, settings),
            builder_fallback: FuseOptimizationBuilder::new(max_bindings, bool_precision, settings),
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
                let lhs = self.builder.input_unhandled(&op.lhs);
                let rhs = self.builder.input_unhandled(&op.rhs);
                let out = self.builder.output_unhandled(&op.out);

                self.matmul = Some(FusedMatmul::new(
                    MatmulArg::Normal(lhs),
                    MatmulArg::Normal(rhs),
                    out,
                    op.clone(),
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
