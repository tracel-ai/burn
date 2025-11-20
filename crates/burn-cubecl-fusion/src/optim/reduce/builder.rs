use super::optimization::{FusedReduce, ReduceInstruction, ReduceOptimization};
use crate::{
    engine::{
        builder::FuseTraceCompiler,
        ir::FuseType,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::CubeOptimization,
};
use burn_fusion::{OperationFuser, OptimizationStatus};
use burn_ir::{NumericOperationIr, OperationIr, ReduceDimOpIr};
use cubecl::{Runtime, reduce::ReduceStrategy};

/// Fused element wise operations that are normally memory bound.
pub struct ReduceBuilder<R: Runtime> {
    builder: FuseTraceCompiler,
    builder_read_fallback: FuseTraceCompiler,
    builder_write_fallback: FuseTraceCompiler,
    settings_write: FuseSettings,
    device: R::Device,
    reduce: Option<FusedReduce>,
}

impl<R: Runtime> Clone for ReduceBuilder<R> {
    fn clone(&self) -> Self {
        Self {
            builder: self.builder.clone(),
            builder_read_fallback: self.builder_read_fallback.clone(),
            builder_write_fallback: self.builder_write_fallback.clone(),
            settings_write: self.settings_write,
            device: self.device.clone(),
            reduce: self.reduce.clone(),
        }
    }
}

impl<R: Runtime> ReduceBuilder<R> {
    pub fn new(device: R::Device, bool_precision: FuseType) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware.max_bindings;
        let settings_read = FuseSettings {
            inplace: false,
            ref_layout: RefLayoutSetting::OnlyContiguous,
            ..Default::default()
        };
        let settings_write = FuseSettings {
            output_shape_updates: false,
            vectorization: VectorizationSetting::SmallerOrEqualThanPreviousBlock,
            ..Default::default()
        };
        let settings_fallback = FuseSettings::default();

        Self {
            builder: FuseTraceCompiler::new(max_bindings, bool_precision, settings_read),
            builder_read_fallback: FuseTraceCompiler::new(
                max_bindings,
                bool_precision,
                settings_fallback,
            ),
            builder_write_fallback: FuseTraceCompiler::new(
                max_bindings,
                bool_precision,
                settings_fallback,
            ),
            settings_write,
            device,
            reduce: None,
        }
    }

    fn on_reduce(&mut self, op: &ReduceDimOpIr, inst: ReduceInstruction) {
        if self.builder.current_output_shape != op.input.shape.dims {
            self.builder.close();
            self.builder_read_fallback.close();
            return;
        }

        let Some([input]) = self.builder.next_block([&op.input], self.settings_write) else {
            self.builder.close();
            self.builder_read_fallback.close();
            return;
        };

        let output = self.builder.output_unhandled(&op.out);
        let axis = op.axis;

        // We only activate fuse-on-write when the reduction isn't on the last dimension, otherwise
        // vectorization is impossible. Only [LineMode::Perpendicular] supports vectorization.
        //
        // We could still fuse some output operations, but it would probably lead to worse performance.
        let fuse_on_write_activated = axis != op.input.shape.rank() - 1;

        if !fuse_on_write_activated {
            self.builder.close();
        }

        let acc = match inst {
            ReduceInstruction::Mean | ReduceInstruction::Prod | ReduceInstruction::Sum => {
                match input.precision() {
                    FuseType::F16 | FuseType::BF16 => FuseType::F32,
                    FuseType::I16 | FuseType::I8 => FuseType::I32,
                    FuseType::U16 | FuseType::U8 => FuseType::U32,
                    _ => input.precision(),
                }
            }
            _ => input.precision(),
        };

        self.reduce = Some(FusedReduce::new(
            input,
            output,
            acc,
            axis,
            op.clone(),
            ReduceStrategy {
                shared: false,
                use_planes: false,
            },
            inst,
        ));
        self.builder_read_fallback.close();
    }

    fn on_elemwise_read(&mut self, operation: &OperationIr) {
        let can_register = self.builder.can_register(operation)
            && self.builder_read_fallback.can_register(operation);

        match can_register {
            true => {
                self.builder.fuse(operation);
                self.builder_read_fallback.fuse(operation);
            }
            false => {
                self.builder.close();
                self.builder_read_fallback.close();
            }
        };
    }

    fn on_elemwise_write(&mut self, operation: &OperationIr) {
        let can_register = self.builder.can_register(operation)
            && self.builder_write_fallback.can_register(operation);

        match can_register {
            true => {
                self.builder.fuse(operation);
                self.builder_write_fallback.fuse(operation);
            }
            false => {
                self.builder.close();
                self.builder_write_fallback.close();
            }
        };
    }
}

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for ReduceBuilder<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        if let OptimizationStatus::Closed = self.builder.status() {
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
                    NumericOperationIr::MinDim(op) => {
                        self.on_reduce(op, ReduceInstruction::Min);
                    }
                    NumericOperationIr::MaxDim(op) => {
                        self.on_reduce(op, ReduceInstruction::Max);
                    }
                    NumericOperationIr::MaxAbsDim(op) => {
                        self.on_reduce(op, ReduceInstruction::MaxAbs);
                    }
                    _ => {
                        self.on_elemwise_read(operation);
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
                    NumericOperationIr::MinDim(op) => {
                        self.on_reduce(op, ReduceInstruction::Min);
                    }
                    NumericOperationIr::MaxDim(op) => {
                        self.on_reduce(op, ReduceInstruction::Max);
                    }
                    NumericOperationIr::MaxAbsDim(op) => {
                        self.on_reduce(op, ReduceInstruction::MaxAbs);
                    }
                    _ => {
                        self.on_elemwise_read(operation);
                    }
                };
            } else {
                self.on_elemwise_read(operation);
            }
        } else {
            self.on_elemwise_write(operation);
        }
    }

    fn finish(&self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let trace = self.builder.finish();
        let trace_read_fallback = self.builder_read_fallback.finish();
        let trace_write_fallback = self.builder_write_fallback.finish();
        let fuse_reduce = self.reduce.as_ref().unwrap();

        let reduce = ReduceOptimization::<R>::new(
            trace,
            trace_read_fallback,
            trace_write_fallback,
            client,
            self.device.clone(),
            self.len(),
            self.builder_read_fallback.len(),
            fuse_reduce.clone(),
        );

        CubeOptimization::Reduce(reduce)
    }

    fn reset(&mut self) {
        self.builder.reset();
        self.builder_read_fallback.reset();
        self.builder_write_fallback.reset();
        self.reduce = None;
    }

    fn status(&self) -> burn_fusion::OptimizationStatus {
        self.builder.status()
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
        self.builder.len() + if self.reduce.is_some() { 1 } else { 0 }
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
