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
    builder_read: FuseBuilder,
    builder_write: FuseBuilder,
    builder_read_fallback: FuseBuilder,
    builder_write_fallback: FuseBuilder,
    device: R::Device,
    reduce: Option<FusedReduce>,
    status: OptimizationStatus,
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
            builder_read: FuseBuilder::new(max_bindings, bool_precision, settings),
            builder_write: FuseBuilder::new(max_bindings, bool_precision, settings),
            builder_read_fallback: FuseBuilder::new(max_bindings, bool_precision, settings),
            builder_write_fallback: FuseBuilder::new(max_bindings, bool_precision, settings),
            device,
            reduce: None,
            status: OptimizationStatus::Open,
        }
    }
}

impl<R: Runtime> OptimizationBuilder<CubeOptimization<R>> for ReduceBuilder<R> {
    fn register(&mut self, operation: &OperationIr) {
        if let OptimizationStatus::Closed = self.status {
            return;
        }

        if self.reduce.is_none() {
            if let OperationIr::NumericFloat(_, NumericOperationIr::SumDim(op)) = operation {
                let input = self.builder_read.input(&op.input);
                self.builder_read.not_output(&op.input);

                let output = self.builder_write.output_unhandled(&op.out);
                let axis = op.axis;

                self.reduce = Some(FusedReduce::new(input, output, axis, op.clone()));
                self.builder_read.close();
                self.builder_read_fallback.close();
                self.status = OptimizationStatus::Closed;
            } else {
                self.builder_read.register(operation);

                if self.builder_read_fallback.len() < self.builder_read.len() {
                    self.builder_read_fallback.register(operation);
                }

                self.status = self.builder_read.status();
            }
        } else {
            panic!("Should not happen");
        }
    }

    fn build(&self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let trace_read = self.builder_read.build();
        let trace_write = self.builder_write.build();
        let trace_read_fallback = self.builder_read_fallback.build();
        let trace_write_fallback = self.builder_write_fallback.build();

        let reduce = ReduceOptimization::<R>::new(
            trace_read,
            trace_write,
            trace_read_fallback,
            trace_write_fallback,
            client,
            self.device.clone(),
            self.len(),
            self.reduce.as_ref().unwrap().clone(),
        );

        CubeOptimization::Reduce(reduce)
    }

    fn reset(&mut self) {
        self.builder_read.reset();
        self.builder_write.reset();
        self.builder_read_fallback.reset();
        self.builder_write_fallback.reset();
        self.reduce = None;
        self.status = OptimizationStatus::Open;
    }

    fn status(&self) -> burn_fusion::OptimizationStatus {
        self.status
    }

    fn properties(&self) -> burn_fusion::OptimizationProperties {
        let mut properties = self.builder_read.properties();
        properties.ready = false;

        if self.reduce.is_some() {
            let properties_write = self.builder_write.properties();
            properties.score += properties_write.score + 10;
            properties.ready = true;
            properties
        } else {
            properties.ready = false;
            properties
        }
    }

    fn len(&self) -> usize {
        self.builder_read.len()
            + self.builder_write.len()
            + self.reduce.as_ref().map(|_| 1).unwrap_or(0)
    }
}
