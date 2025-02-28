use std::sync::Arc;

use super::MatmulFallbackFn;
use burn_fusion::{OptimizationBuilder, OptimizationStatus};
use burn_ir::{FloatOperationIr, OperationIr};
use cubecl::Runtime;

use crate::{
    on_write::{builder::FuseOnWriteBuilder, ir::ElemwisePrecision, settings::FuseSettings},
    CubeOptimization,
};

use super::optimization::{FusedMatmul, MatmulOptimization};

/// Fused element wise operations that are normally memory bound.
pub struct MatmulBuilder<R: Runtime> {
    builder: FuseOnWriteBuilder,
    builder_fallback: FuseOnWriteBuilder,
    device: R::Device,
    matmul: Option<FusedMatmul>,
    fallback: Arc<dyn MatmulFallbackFn<R>>,
}

impl<R: Runtime> MatmulBuilder<R> {
    pub fn new(
        device: R::Device,
        bool_precision: ElemwisePrecision,
        fallback: Arc<dyn MatmulFallbackFn<R>>,
    ) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware_properties().max_bindings;
        let settings = FuseSettings {
            broadcast: true,
            output_shape_updates: false,
            inplace: true,
        };

        Self {
            builder: FuseOnWriteBuilder::new(max_bindings, bool_precision, settings),
            builder_fallback: FuseOnWriteBuilder::new(max_bindings, bool_precision, settings),
            device,
            matmul: None,
            fallback,
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
                    lhs,
                    rhs,
                    out,
                    op.clone(),
                    Default::default(),
                ));
            } else {
                self.builder.close();
                self.builder_fallback.close();
            }
        } else {
            self.builder.register(operation);

            // We might not be able to accept an operation because of unhandled tensors in the
            // fused matmul builder. To keep both builders in sync we have to check their length.
            if self.builder_fallback.len() < self.builder.len() {
                self.builder_fallback.register(operation);
            }
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
            self.fallback.clone(),
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
}
