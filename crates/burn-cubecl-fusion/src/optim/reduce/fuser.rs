use super::{
    ReduceSettings,
    optimization::{FusedReduce, ReduceInstruction, ReduceOptimization},
};
use crate::{
    engine::{
        codegen::ir::FuseType,
        fuser::TraceOperationFuser,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::CubeOptimization,
};
use burn_fusion::{FuserStatus, OperationFuser};
use burn_ir::{NumericOperationIr, OperationIr, ReduceDimOpIr};
use cubecl::Runtime;

/// Fuses element wise operations around a reduce operation.
pub struct ReduceFuser<R: Runtime> {
    pub(crate) fuser: TraceOperationFuser,
    pub(crate) fuser_read_fallback: TraceOperationFuser,
    fuser_write_fallback: TraceOperationFuser,
    settings_write: FuseSettings,
    pub(crate) device: R::Device,
    pub(crate) reduce: Option<FusedReduce>,
    settings: ReduceSettings,
}

impl<R: Runtime> Clone for ReduceFuser<R> {
    fn clone(&self) -> Self {
        Self {
            fuser: self.fuser.clone(),
            fuser_read_fallback: self.fuser_read_fallback.clone(),
            fuser_write_fallback: self.fuser_write_fallback.clone(),
            settings_write: self.settings_write,
            device: self.device.clone(),
            reduce: self.reduce.clone(),
            settings: self.settings,
        }
    }
}

#[derive(Debug)]
pub enum ReduceFuserInfo {
    FusedReduce {
        shape_input_id: Vec<usize>,
        axis: usize,
    },
    FusedElemwise {
        shape_id: Vec<usize>,
    },
}

impl<R: Runtime> ReduceFuser<R> {
    pub fn new(device: R::Device, bool_precision: FuseType, settings: ReduceSettings) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware.max_bindings;
        let settings_read = FuseSettings {
            inplace: false,
            ref_layout: RefLayoutSetting::OnlyContiguous,
            broadcast: true,
            output_shape_updates: true,
            vectorization: VectorizationSetting::Activated,
        };
        let settings_write = FuseSettings {
            inplace: false,
            output_shape_updates: false,
            // TODO: Fusion axis should be on the reduce_axis - 1.
            // vectorization: VectorizationSetting::SmallerOrEqualThanPreviousBlock { block_pos: 0 },
            vectorization: VectorizationSetting::Deactivated,
            broadcast: false,
            ref_layout: RefLayoutSetting::OnlyContiguous,
        };
        let settings_fallback = FuseSettings::default();

        Self {
            fuser: TraceOperationFuser::new(max_bindings, bool_precision, settings_read),
            fuser_read_fallback: TraceOperationFuser::new(
                max_bindings,
                bool_precision,
                settings_fallback,
            ),
            fuser_write_fallback: TraceOperationFuser::new(
                max_bindings,
                bool_precision,
                settings_fallback,
            ),
            settings_write,
            device,
            reduce: None,
            settings,
        }
    }

    pub fn reduce_info(&self) -> ReduceFuserInfo {
        match &self.reduce {
            Some(reduce) => {
                let shape_input_id = reduce.op.input.shape.dims.clone();
                let axis = reduce.axis;

                ReduceFuserInfo::FusedReduce {
                    shape_input_id,
                    axis,
                }
            }
            None => {
                let shape_id = self.fuser_read_fallback.current_output_shape.clone();
                ReduceFuserInfo::FusedElemwise { shape_id }
            }
        }
    }
    fn on_reduce(&mut self, op: &ReduceDimOpIr, inst: ReduceInstruction) {
        // TODO: Fix: we need to hava fuse-on-read with an identity block.
        if self.fuser.num_ops == 0 && false {
            self.fuser.current_output_shape = op.input.shape.dims.clone();
        } else if self.fuser.current_output_shape != op.input.shape.dims {
            self.fuser.close();
            self.fuser_read_fallback.close();
            return;
        }

        let [input] = self
            .fuser
            .next_block([&op.input], self.settings_write, false);

        let output = self.fuser.output_unhandled(&op.out);
        let axis = op.axis;

        let fuse_on_write_activated = match self.settings {
            ReduceSettings::Always => true,
            // We only activate fuse-on-write when the reduction isn't on the last dimension, otherwise
            // vectorization is impossible. Only [LineMode::Perpendicular] supports vectorization.
            //
            // We could still fuse some output operations, but it would probably lead to worse performance.
            ReduceSettings::OnlyParallel => axis != op.input.shape.rank() - 1,
            ReduceSettings::Never => false,
        };

        if !fuse_on_write_activated {
            self.fuser.close();
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

        self.reduce = Some(FusedReduce {
            input,
            output,
            acc,
            axis,
            op: op.clone(),
            use_planes: false,
            shared: false,
            inst,
        });

        self.fuser_read_fallback.close();
    }

    fn on_elemwise_read(&mut self, operation: &OperationIr) {
        let can_register =
            self.fuser.can_fuse(operation) && self.fuser_read_fallback.can_fuse(operation);

        match can_register {
            true => {
                self.fuser.fuse(operation);
                self.fuser_read_fallback.fuse(operation);
            }
            false => {
                self.fuser.close();
                self.fuser_read_fallback.close();
            }
        };
    }

    fn on_elemwise_write(&mut self, operation: &OperationIr) {
        let can_register =
            self.fuser.can_fuse(operation) && self.fuser_write_fallback.can_fuse(operation);

        match can_register {
            true => {
                self.fuser.fuse(operation);
                self.fuser_write_fallback.fuse(operation);
            }
            false => {
                self.fuser.close();
                self.fuser_write_fallback.close();
            }
        };
    }
}

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for ReduceFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        if let FuserStatus::Closed = self.fuser.status() {
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

    fn finish(&mut self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let trace = self.fuser.finish();
        let trace_read_fallback = self.fuser_read_fallback.finish();
        let trace_write_fallback = self.fuser_write_fallback.finish();
        let fuse_reduce = self.reduce.as_ref().unwrap();

        let reduce = ReduceOptimization::new(
            trace,
            trace_read_fallback,
            trace_write_fallback,
            client,
            self.device.clone(),
            self.len(),
            self.fuser_read_fallback.len(),
            fuse_reduce.clone(),
            self.settings.clone(),
        );

        CubeOptimization::Reduce(reduce)
    }

    fn reset(&mut self) {
        self.fuser.reset();
        self.fuser_read_fallback.reset();
        self.fuser_write_fallback.reset();
        self.reduce = None;
    }

    fn status(&self) -> burn_fusion::FuserStatus {
        self.fuser.status()
    }

    fn properties(&self) -> burn_fusion::FuserProperties {
        let mut properties = self.fuser.properties();

        if self.reduce.is_some() {
            properties.ready = true;
            properties.score += 1;
        } else {
            properties.ready = false;
        };

        properties
    }

    fn len(&self) -> usize {
        self.fuser.len() + if self.reduce.is_some() { 1 } else { 0 }
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
