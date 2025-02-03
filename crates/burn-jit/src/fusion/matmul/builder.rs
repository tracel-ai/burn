use burn_fusion::{OptimizationBuilder, OptimizationStatus};
use burn_tensor::repr::{FloatOperationDescription, OperationDescription};

use crate::{
    fusion::{
        on_write::{builder::FuseOnWriteBuilder, ir::ElemwisePrecision, settings::FuseSettings},
        JitOptimization,
    },
    JitRuntime,
};

use super::optimization::{FusedMatmul, MatmulOptimization};

/// Fused element wise operations that are normally memory bound.
pub(crate) struct MatmulBuilder<R: JitRuntime> {
    builder: FuseOnWriteBuilder,
    builder_fallback: FuseOnWriteBuilder,
    device: R::Device,
    matmul: Option<FusedMatmul>,
}

impl<R: JitRuntime> MatmulBuilder<R> {
    pub fn new(device: R::Device, bool_precision: ElemwisePrecision) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware_properties().max_bindings;
        let settings = FuseSettings {
            broadcast: true,
            output_shape_updates: false,
            mix_vectorization: true,
            inplace: true,
        };

        Self {
            builder: FuseOnWriteBuilder::new(max_bindings, bool_precision, settings),
            builder_fallback: FuseOnWriteBuilder::new(max_bindings, bool_precision, settings),
            device,
            matmul: None,
        }
    }
}

impl<R: JitRuntime> OptimizationBuilder<JitOptimization<R>> for MatmulBuilder<R> {
    fn register(&mut self, operation: &OperationDescription) {
        if let OptimizationStatus::Closed = self.builder.status() {
            return;
        }

        if self.matmul.is_none() {
            if let OperationDescription::Float(_, FloatOperationDescription::Matmul(op)) = operation
            {
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
            self.builder_fallback.register(operation);
        }
    }

    fn build(&self) -> JitOptimization<R> {
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

        JitOptimization::Matmul(matmul)
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
