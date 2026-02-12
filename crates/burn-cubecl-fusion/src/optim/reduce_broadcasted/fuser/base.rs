use crate::{
    engine::codegen::ir::FuseType,
    optim::{
        CubeOptimization,
        reduce::{ReduceFuser, ReduceFuserInfo, ReduceSettings},
        reduce_broadcasted::{
            ReduceBroadcastedOptimization, ReduceBroadcastedOptimizationInfo,
            fuser::{
                block::{ReduceBlockFuser, ReduceBlockFusionAnalysis, ReduceBroadcastedStatus},
                full::ReduceBroadcastedFullFuser,
                full_analyzer::FullFuserAnalyzer,
            },
        },
    },
};
use burn_fusion::{FuserProperties, FuserStatus, OperationFuser};
use burn_ir::OperationIr;
use cubecl::Runtime;
use std::sync::Arc;

/// Fuses element wise operations around a reduce operation.
pub struct ReduceBroadcastedFuser<R: Runtime> {
    blocks: Vec<ReduceBlockFuser<R>>,
    fuser_default: ReduceFuser<R>,
    num_ops: usize,
    state: ReduceBroadcastedStatus,
    max_bindings: u32,
    bool_precision: FuseType,
}

impl<R: Runtime> Clone for ReduceBroadcastedFuser<R> {
    fn clone(&self) -> Self {
        Self {
            blocks: self.blocks.clone(),
            fuser_default: self.fuser_default.clone(),
            num_ops: self.num_ops,
            state: self.state.clone(),
            max_bindings: self.max_bindings,
            bool_precision: self.bool_precision,
        }
    }
}

impl<R: Runtime> ReduceBroadcastedFuser<R> {
    pub fn new(device: R::Device, bool_precision: FuseType) -> Self {
        let fuser = ReduceFuser::new(device, bool_precision, ReduceSettings::Always);
        let max_bindings = fuser.fuser.max_bindings;
        let block = ReduceBlockFuser::new(fuser.clone());

        Self {
            blocks: vec![block],
            fuser_default: fuser,
            num_ops: 0,
            state: ReduceBroadcastedStatus::Starting,
            max_bindings,
            bool_precision,
        }
    }
}

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for ReduceBroadcastedFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        if matches!(
            &self.state,
            ReduceBroadcastedStatus::Closed | ReduceBroadcastedStatus::Abort
        ) {
            return;
        }

        let block = self.blocks.last_mut().unwrap();
        let analyze = block.analyze(operation, &self.state, &self.fuser_default);

        let info = match analyze {
            ReduceBlockFusionAnalysis::Accept => {
                block.fuse(operation);
                self.num_ops += 1;
                block.fuser.reduce_info()
            }
            ReduceBlockFusionAnalysis::Refuse => {
                self.state = ReduceBroadcastedStatus::Closed;
                return;
            }
            ReduceBlockFusionAnalysis::NewBlockRequired => {
                let info = block.fuser.reduce_info();
                let mut block = ReduceBlockFuser::new(self.fuser_default.clone());
                block.fuse(operation);
                self.num_ops += 1;
                self.blocks.push(block);
                info
            }
        };

        match info {
            ReduceFuserInfo::FusedReduce {
                shape_input_id,
                axis,
            } => {
                // Only support last axis for now.
                if axis != shape_input_id.len() - 1 {
                    self.state = ReduceBroadcastedStatus::Abort;
                } else {
                    self.state = ReduceBroadcastedStatus::Init {
                        shape_id: shape_input_id,
                        axis,
                    };
                }
            }
            ReduceFuserInfo::FusedElemwise { .. } => {}
        }
    }

    fn finish(&mut self) -> CubeOptimization<R> {
        let analyzer = FullFuserAnalyzer::new(&self.blocks);
        let mut full =
            ReduceBroadcastedFullFuser::new(self.max_bindings, self.bool_precision, analyzer);
        let mut num_ops = 0;
        let fallbacks = self
            .blocks
            .iter_mut()
            .map(|block| block.finish(&mut num_ops, &mut full))
            .collect::<Vec<_>>();

        let broadcasted = Arc::new(full.finish());
        let info = Arc::new(ReduceBroadcastedOptimizationInfo {
            fallbacks,
            broadcasted,
        });
        CubeOptimization::ReduceBroadcasted(ReduceBroadcastedOptimization { info, num_ops })
    }

    fn reset(&mut self) {
        let block = ReduceBlockFuser::new(self.fuser_default.clone());
        self.blocks = vec![block];
        self.num_ops = 0;
        self.state = ReduceBroadcastedStatus::Starting;
    }

    fn status(&self) -> FuserStatus {
        match self.state {
            ReduceBroadcastedStatus::Closed | ReduceBroadcastedStatus::Abort => {
                return FuserStatus::Closed;
            }
            _ => {}
        };

        let fuser = self.blocks.last().unwrap();
        fuser.fuser.status()
    }

    fn properties(&self) -> FuserProperties {
        let ready = match self.state {
            ReduceBroadcastedStatus::Starting | ReduceBroadcastedStatus::Abort => false,
            ReduceBroadcastedStatus::Closed => {
                if self.blocks.len() == 1 {
                    !self.blocks[0].is_elemwise()
                } else {
                    true
                }
            }
            _ => true,
        };
        let mut props = FuserProperties { score: 0, ready };
        for block in self.blocks.iter() {
            let p = block.properties();
            props.score += p.score;
            props.ready = p.ready && props.ready;
        }
        props
    }

    fn len(&self) -> usize {
        self.num_ops
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use burn_ir::{
        BaseOperationIr, BinaryOpIr, CreationOpIr, ReduceDimOpIr, TensorId, TensorIr, TensorStatus,
    };
    use burn_std::{DType, Shape};

    use super::*;

    type Run = cubecl::TestRuntime;

    #[test]
    fn reduce_broadcast_workflow_1() {
        let device: <Run as Runtime>::Device = Default::default();
        let mut fuser = ReduceBroadcastedFuser::<Run>::new(device, FuseType::I32);
        let (tensor1_out, tensor1) = tensor(0, vec![1, 2], TensorStatus::ReadWrite);
        let (tensor2_out, tensor2) = tensor(1, vec![1, 0], TensorStatus::ReadWrite);

        fuser.fuse(&OperationIr::BaseFloat(BaseOperationIr::Ones(
            CreationOpIr { out: tensor1_out },
        )));
        fuser.fuse(&OperationIr::NumericFloat(
            DType::F32,
            burn_ir::NumericOperationIr::SumDim(ReduceDimOpIr {
                input: tensor1,
                out: tensor2_out,
                axis: 1,
            }),
        ));

        let status = fuser.status();
        assert_eq!(2, fuser.len());
        assert_eq!(status, FuserStatus::Open);
        assert!(fuser.properties().ready,);

        // An existing tensor
        let (_tensor3_out, tensor3) = tensor(2, vec![1, 0], TensorStatus::ReadWrite);
        // A new tensor
        let (tensor4_out, tensor4) = tensor(3, vec![1, 0], TensorStatus::ReadWrite);
        fuser.fuse(&OperationIr::NumericFloat(
            DType::F32,
            burn_ir::NumericOperationIr::Add(BinaryOpIr {
                lhs: tensor2,
                rhs: tensor3,
                out: tensor4_out,
            }),
        ));

        let status = fuser.status();
        assert_eq!(3, fuser.len());
        assert_eq!(status, FuserStatus::Open);
        assert!(fuser.properties().ready,);

        // An existing tensor
        let (_tensor5_out, tensor5) = tensor(4, vec![1, 2], TensorStatus::ReadWrite);
        // A new tensor
        let (tensor6_out, tensor6) = tensor(5, vec![1, 2], TensorStatus::ReadWrite);
        fuser.fuse(&OperationIr::NumericFloat(
            DType::F32,
            burn_ir::NumericOperationIr::Add(BinaryOpIr {
                lhs: tensor4,
                rhs: tensor5,
                out: tensor6_out,
            }),
        ));

        let status = fuser.status();
        assert_eq!(4, fuser.len());
        assert_eq!(status, FuserStatus::Open);
        assert!(fuser.properties().ready,);

        let (tensor7_out, _tensor7) = tensor(6, vec![1, 0], TensorStatus::ReadWrite);
        fuser.fuse(&OperationIr::NumericFloat(
            DType::F32,
            burn_ir::NumericOperationIr::SumDim(ReduceDimOpIr {
                input: tensor6,
                out: tensor7_out,
                axis: 1,
            }),
        ));
        assert_eq!(5, fuser.len());
        assert_eq!(status, FuserStatus::Open);
        assert!(fuser.properties().ready,);

        let _optimization = fuser.finish();
    }

    #[test]
    fn reduce_broadcast_workflow_2() {
        let device: <Run as Runtime>::Device = Default::default();
        let mut fuser = ReduceBroadcastedFuser::<Run>::new(device, FuseType::I32);
        let (tensor1_out, tensor1) = tensor(0, vec![1, 2], TensorStatus::ReadWrite);
        // An existing tensor
        let (_tensor2_out, mut tensor2) = tensor(2, vec![1, 2], TensorStatus::ReadOnly);
        let (tensor3_out, tensor3) = tensor(3, vec![1, 2], TensorStatus::ReadWrite);

        // First reduce output
        let (tensor4_out, tensor4) = tensor(1, vec![1, 0], TensorStatus::ReadWrite);

        fuser.fuse(&OperationIr::BaseFloat(BaseOperationIr::Ones(
            CreationOpIr { out: tensor1_out },
        )));

        fuser.fuse(&OperationIr::NumericFloat(
            DType::F32,
            burn_ir::NumericOperationIr::Add(BinaryOpIr {
                lhs: tensor1,
                rhs: tensor2.clone(),
                out: tensor3_out,
            }),
        ));

        fuser.fuse(&OperationIr::NumericFloat(
            DType::F32,
            burn_ir::NumericOperationIr::SumDim(ReduceDimOpIr {
                input: tensor3,
                out: tensor4_out,
                axis: 1,
            }),
        ));

        let status = fuser.status();
        assert_eq!(3, fuser.len());
        assert_eq!(status, FuserStatus::Open);
        assert!(fuser.properties().ready,);

        // A new tensor
        let (tensor5_out, _tensor5) = tensor(5, vec![1, 2], TensorStatus::ReadWrite);
        // Last time we use tensor2.
        tensor2.status = TensorStatus::ReadWrite;
        fuser.fuse(&OperationIr::NumericFloat(
            DType::F32,
            burn_ir::NumericOperationIr::Add(BinaryOpIr {
                lhs: tensor4,
                rhs: tensor2,
                out: tensor5_out,
            }),
        ));

        let status = fuser.status();
        assert_eq!(4, fuser.len());
        assert_eq!(status, FuserStatus::Open);
        assert!(fuser.properties().ready,);

        let _optimization = fuser.finish();
    }

    fn tensor(id: u64, shape: Vec<usize>, status: TensorStatus) -> (TensorIr, TensorIr) {
        let tensor = TensorIr {
            id: TensorId::new(id),
            shape: Shape { dims: shape },
            status: TensorStatus::NotInit,
            dtype: DType::F32,
        };
        let mut tensor_init = tensor.clone();
        tensor_init.status = status;

        (tensor, tensor_init)
    }
}
