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
    builder: FuseBuilder,
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
            broadcast: true,
            output_shape_updates: false,
            inplace: true,
            vectorization: true,
        };

        Self {
            builder: FuseBuilder::new(max_bindings, bool_precision, settings_read),
            builder_read_fallback: FuseBuilder::new(max_bindings, bool_precision, settings_read),
            builder_write_fallback: FuseBuilder::new(max_bindings, bool_precision, settings_write),
            device,
            reduce: None,
            status: OptimizationStatus::Open,
            fallback,
        }
    }

    fn on_reduce(&mut self, op: &ReduceDimOpIr, inst: ReduceInstruction) {
        if self.builder.current_output_shape != op.input.shape {
            self.builder.close();
            self.builder_read_fallback.close();
            self.status = OptimizationStatus::Closed;
            return;
        }

        let input = self.builder.input(&op.input);
        self.builder
            .next_block(&[&op.input], self.builder_write_fallback.settings);

        let output = self.builder.output_unhandled(&op.out);
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
        self.builder.close();
        self.builder_read_fallback.close();
        self.status = OptimizationStatus::Open;
    }

    fn on_elemwise(&mut self, operation: &OperationIr) {
        self.builder.register(operation);

        if self.builder_read_fallback.len() < self.builder.len() {
            self.builder_read_fallback.register(operation);
        }

        self.status = self.builder.status();
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
                // On fuse on read.
                self.on_elemwise(operation);
            }
        } else {
            // On fuse on write.
            self.on_elemwise(operation);
        }
    }

    fn build(&self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let trace = self.builder.build();
        let trace_read_fallback = self.builder_read_fallback.build();
        let trace_write_fallback = self.builder_write_fallback.build();
        let fuse_reduce = self.reduce.as_ref().unwrap();

        let reduce = ReduceOptimization::<R>::new(
            trace,
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
        self.builder.reset();
        self.builder_read_fallback.reset();
        self.builder_write_fallback.reset();
        self.reduce = None;
        self.status = OptimizationStatus::Open;
    }

    fn status(&self) -> burn_fusion::OptimizationStatus {
        self.status
    }

    fn properties(&self) -> burn_fusion::OptimizationProperties {
        let mut properties = self.builder.properties();

        if self.reduce.is_some() {
            properties.ready = true;
            properties.score += 1;
        } else {
            properties.ready = false;
        };

        properties
    }

    fn len(&self) -> usize {
        self.builder.len() + self.reduce.as_ref().map(|_| 1).unwrap_or(0)
    }
}
