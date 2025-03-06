use std::sync::Arc;

use super::optimization::ReduceInstruction;
use burn_fusion::{OptimizationBuilder, OptimizationStatus};
use burn_ir::{NumericOperationIr, OperationIr, ReduceDimOpIr};
use cubecl::{reduce::ReduceStrategy, Runtime};

use crate::{
    shared::{builder::FuseBuilder, ir::ElemwisePrecision, settings::FuseSettings},
    CubeOptimization,
};

use super::optimization::{FusedReduce, ReduceFallbackFn, ReduceOptimization};

/// Fused element wise operations that are normally memory bound.
pub struct ReduceBuilder<R: Runtime> {
    builder_read: FuseBuilder,
    builder_write: FuseBuilder,
    builder_read_fallback: FuseBuilder,
    builder_write_fallback: FuseBuilder,
    device: R::Device,
    reduce: Option<FusedReduce>,
    status: OptimizationStatus,
    fallback: Arc<dyn ReduceFallbackFn<R>>,
}

impl<R: Runtime> ReduceBuilder<R> {
    pub fn new(
        device: R::Device,
        bool_precision: ElemwisePrecision,
        fallback: Arc<dyn ReduceFallbackFn<R>>,
    ) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware_properties().max_bindings;
        let settings_read = FuseSettings {
            broadcast: true,
            output_shape_updates: true,
            inplace: true,
            vectorization: true,
        };
        let settings_write = FuseSettings {
            broadcast: false,
            output_shape_updates: false,
            inplace: false,
            vectorization: false,
        };

        Self {
            builder_read: FuseBuilder::new(max_bindings, bool_precision, settings_read),
            builder_write: FuseBuilder::new(max_bindings, bool_precision, settings_write),
            builder_read_fallback: FuseBuilder::new(max_bindings, bool_precision, settings_read),
            builder_write_fallback: FuseBuilder::new(max_bindings, bool_precision, settings_write),
            device,
            reduce: None,
            status: OptimizationStatus::Open,
            fallback,
        }
    }
    fn on_reduce(&mut self, op: &ReduceDimOpIr, inst: ReduceInstruction) {
        if self.builder_read.current_output_shape != op.input.shape {
            self.builder_read.close();
            self.builder_read_fallback.close();
            self.status = OptimizationStatus::Closed;
            return;
        }

        let input = self.builder_read.input(&op.input);
        self.builder_read.not_output(&op.input);

        let output = self.builder_write.output_unhandled(&op.out);
        let axis = op.axis;

        self.reduce = Some(FusedReduce::new(
            input,
            output,
            axis,
            op.clone(),
            ReduceStrategy {
                shared: false,
                use_planes: false,
            },
            inst,
        ));
        self.builder_read.close();
        self.builder_read_fallback.close();
        self.status = OptimizationStatus::Closed;
    }

    fn on_elemwise(&mut self, operation: &OperationIr) {
        self.builder_read.register(operation);

        if self.builder_read_fallback.len() < self.builder_read.len() {
            self.builder_read_fallback.register(operation);
        }

        self.status = self.builder_read.status();
    }
}

impl<R: Runtime> OptimizationBuilder<CubeOptimization<R>> for ReduceBuilder<R> {
    fn register(&mut self, operation: &OperationIr) {
        if let OptimizationStatus::Closed = self.status {
            return;
        }

        if self.reduce.is_none() {
            if let OperationIr::NumericFloat(_, op) = operation {
                match op {
                    NumericOperationIr::SumDim(op) => {
                        self.on_reduce(op, ReduceInstruction::Sum);
                    }
                    NumericOperationIr::MeanDim(op) => {
                        self.on_reduce(op, ReduceInstruction::Mean);
                    }
                    NumericOperationIr::ProdDim(op) => {
                        self.on_reduce(op, ReduceInstruction::Prod);
                    }
                    NumericOperationIr::ArgMax(op) => {
                        self.on_reduce(op, ReduceInstruction::ArgMax);
                    }
                    NumericOperationIr::ArgMin(op) => {
                        self.on_reduce(op, ReduceInstruction::ArgMin);
                    }
                    _ => {
                        self.on_elemwise(operation);
                    }
                };
            } else if let OperationIr::NumericInt(_, op) = operation {
                match op {
                    NumericOperationIr::SumDim(op) => {
                        self.on_reduce(op, ReduceInstruction::Sum);
                    }
                    NumericOperationIr::MeanDim(op) => {
                        self.on_reduce(op, ReduceInstruction::Mean);
                    }
                    NumericOperationIr::ProdDim(op) => {
                        self.on_reduce(op, ReduceInstruction::Prod);
                    }
                    NumericOperationIr::ArgMax(op) => {
                        self.on_reduce(op, ReduceInstruction::ArgMax);
                    }
                    NumericOperationIr::ArgMin(op) => {
                        self.on_reduce(op, ReduceInstruction::ArgMin);
                    }
                    _ => {
                        self.on_elemwise(operation);
                    }
                };
            } else {
                self.on_elemwise(operation);
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
        let fuse_reduce = self.reduce.as_ref().unwrap();

        let reduce = ReduceOptimization::<R>::new(
            trace_read,
            trace_write,
            trace_read_fallback,
            trace_write_fallback,
            client,
            self.device.clone(),
            self.len(),
            fuse_reduce.clone(),
            self.fallback.clone(),
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
            properties.score += properties_write.score + 1;
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
