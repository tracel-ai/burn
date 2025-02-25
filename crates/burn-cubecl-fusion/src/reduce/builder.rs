use burn_fusion::{OptimizationBuilder, OptimizationStatus};
use burn_ir::{NumericOperationIr, OperationIr};
use cubecl::Runtime;

use crate::{
    shared::{builder::FuseBuilder, ir::ElemwisePrecision, settings::FuseSettings},
    CubeOptimization,
};

use super::optimization::{FusedReduce, ReduceOptimization};

/// Fused element wise operations that are normally memory bound.
pub struct ReduceBuilder<R: Runtime> {
    builder: FuseBuilder,
    builder_fallback: FuseBuilder,
    device: R::Device,
    reduce: Option<FusedReduce>,
}

impl<R: Runtime> ReduceBuilder<R> {
    pub fn new(device: R::Device, bool_precision: ElemwisePrecision) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware_properties().max_bindings;
        let settings = FuseSettings {
            broadcast: true,
            output_shape_updates: false,
            inplace: true,
        };

        Self {
            builder: FuseBuilder::new(max_bindings, bool_precision, settings),
            builder_fallback: FuseBuilder::new(max_bindings, bool_precision, settings),
            device,
            reduce: None,
        }
    }
}

impl<R: Runtime> OptimizationBuilder<CubeOptimization<R>> for ReduceBuilder<R> {
    fn register(&mut self, operation: &OperationIr) {
        if let OptimizationStatus::Closed = self.builder.status() {
            return;
        }

        if self.reduce.is_none() {
            if let OperationIr::NumericFloat(_, NumericOperationIr::SumDim(op)) = operation {
                let input = self.builder.input(&op.input);
                let output = self.builder.output_manual(&op.out);
                let axis = op.axis;

                self.reduce = Some(FusedReduce::new(input, output, axis, op.clone()));
                self.builder.close();
                self.builder_fallback.close();
            } else {
                self.builder.register(operation);

                if self.builder_fallback.len() < self.builder.len() {
                    self.builder_fallback.register(operation);
                }
            }
        } else {
            panic!("Should not happen");
        }
    }

    fn build(&self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let trace = self.builder.build();
        let trace_fallback = self.builder_fallback.build();

        let reduce = ReduceOptimization::<R>::new(
            trace,
            trace_fallback,
            client,
            self.device.clone(),
            self.len(),
            self.reduce.as_ref().unwrap().clone(),
        );

        CubeOptimization::Reduce(reduce)
    }

    fn reset(&mut self) {
        self.builder.reset();
        self.builder_fallback.reset();
        self.reduce = None;
    }

    fn status(&self) -> burn_fusion::OptimizationStatus {
        self.builder.status()
    }

    fn properties(&self) -> burn_fusion::OptimizationProperties {
        let mut properties = self.builder.properties();
        if self.reduce.is_some() {
            properties.score += 1;
            properties
        } else {
            properties.ready = false;
            properties
        }
    }

    fn len(&self) -> usize {
        // Reduce operation isn't registered in the builder
        self.builder.len() + self.reduce.as_ref().map(|_| 1).unwrap_or(0)
    }
}
