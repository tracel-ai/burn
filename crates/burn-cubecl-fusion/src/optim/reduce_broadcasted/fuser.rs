use crate::{
    engine::codegen::ir::FuseType,
    optim::{
        CubeOptimization,
        elemwise::ElemwiseOptimization,
        reduce::{ReduceFuser, ReduceFuserInfo, ReduceSettings},
        reduce_broadcasted::{
            ReduceBlockOptimInfo, ReduceBroadcastedOptimization, ReduceBroadcastedOptimizationInfo,
        },
    },
};
use burn_fusion::{FuserProperties, FuserStatus, OperationFuser};
use burn_ir::OperationIr;
use cubecl::Runtime;
use std::sync::Arc;

/// Fuses element wise operations around a reduce operation.
pub struct ReduceBroadcastedFuser<R: Runtime> {
    blocks: Vec<FuserBlock<R>>,
    pub(crate) fuser: ReduceFuser<R>,
    fuser_default: ReduceFuser<R>,
    num_ops: usize,
    state: State,
}

struct FuserBlock<R: Runtime> {
    fuser: ReduceFuser<R>,
    ops: Vec<OperationIr>,
}

impl<R: Runtime> Clone for FuserBlock<R> {
    fn clone(&self) -> Self {
        Self {
            fuser: self.fuser.clone(),
            ops: self.ops.clone(),
        }
    }
}

enum FuserNodeStatus {
    Elemwise,
    Reduce,
}

#[derive(Clone, Copy, Debug)]
enum FuserNodeAnalyse {
    Accept,
    Refuse,
    NewBlockRequired,
}

impl<R: Runtime> FuserBlock<R> {
    pub fn finish(&self, num_ops: &mut usize) -> ReduceBlockOptimInfo<R> {
        match self.status() {
            FuserNodeStatus::Elemwise => {
                let len = self.fuser.fuser_read_fallback.len();
                let device = self.fuser.device.clone();
                *num_ops += len;
                let trace = self.fuser.fuser_read_fallback.finish();
                let client = R::client(&device);
                let elementwise = ElemwiseOptimization::new(trace, client, device, len);
                ReduceBlockOptimInfo::Elemwise(Arc::new(elementwise))
            }
            FuserNodeStatus::Reduce => {
                *num_ops += self.fuser.len();
                let optim = self.fuser.finish();
                let optim = match optim {
                    CubeOptimization::Reduce(optim) => optim.info,
                    _ => unreachable!(),
                };
                ReduceBlockOptimInfo::Reduce(optim)
            }
        }
    }

    pub fn status(&self) -> FuserNodeStatus {
        match self.fuser.properties().ready {
            true => FuserNodeStatus::Reduce,
            false => FuserNodeStatus::Elemwise,
        }
    }

    pub fn fuse(&mut self, op: &OperationIr) {
        self.fuser.fuse(op);
        self.ops.push(op.clone());
    }

    pub fn properties(&self) -> FuserProperties {
        let mut properties = self.fuser.properties();
        match self.status() {
            FuserNodeStatus::Elemwise => properties.ready = true,
            FuserNodeStatus::Reduce => {}
        }
        properties
    }

    pub fn analyze_fusion(
        &self,
        op: &OperationIr,
        status: &State,
        default_node: &ReduceFuser<R>,
    ) -> FuserNodeAnalyse {
        let mut fuser_try = self.fuser.clone();
        let before = fuser_try.len();
        fuser_try.fuse(op);
        let after = fuser_try.len();

        if after > before {
            return FuserNodeAnalyse::Accept;
        }

        let mut fuser_try = default_node.clone();

        let before = fuser_try.len();
        fuser_try.fuse(op);
        let after = fuser_try.len();

        if after > before {
            let info = fuser_try.reduce_info();

            return match (info, status) {
                (
                    ReduceFuserInfo::FusedReduce {
                        shape_input_id,
                        axis,
                    },
                    State::Init {
                        shape_id,
                        axis: axis_init,
                    },
                ) => {
                    if shape_id == &shape_input_id && axis_init == &axis {
                        FuserNodeAnalyse::NewBlockRequired
                    } else {
                        FuserNodeAnalyse::Refuse
                    }
                }
                (
                    ReduceFuserInfo::FusedElemwise { shape_id },
                    State::Init {
                        shape_id: shape_init,
                        ..
                    },
                ) => {
                    if &shape_id == shape_init {
                        FuserNodeAnalyse::NewBlockRequired
                    } else {
                        FuserNodeAnalyse::Refuse
                    }
                }
                _ => FuserNodeAnalyse::Refuse,
            };
        }

        FuserNodeAnalyse::Refuse
    }
}

enum State {
    Starting,
    Init { shape_id: Vec<usize>, axis: usize },
    Closed { num_ops: usize },
}

impl<R: Runtime> Clone for ReduceBroadcastedFuser<R> {
    fn clone(&self) -> Self {
        Self {
            fuser: self.fuser.clone(),
            blocks: self.blocks.clone(),
            fuser_default: self.fuser_default.clone(),
            num_ops: 0,
            state: State::Starting,
        }
    }
}

impl<R: Runtime> ReduceBroadcastedFuser<R> {
    pub fn new(device: R::Device, bool_precision: FuseType) -> Self {
        let fuser = ReduceFuser::new(device, bool_precision, ReduceSettings::Always);
        let block = FuserBlock {
            fuser: fuser.clone(),
            ops: Vec::new(),
        };

        Self {
            blocks: vec![block],
            fuser_default: fuser.clone(),
            fuser,
            num_ops: 0,
            state: State::Starting,
        }
    }
}

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for ReduceBroadcastedFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        if let State::Closed { .. } = &self.state {
            return;
        }

        if let FuserStatus::Closed = self.fuser.status() {
            return;
        }

        let block = self.blocks.last_mut().unwrap();

        let analyze = block.analyze_fusion(operation, &self.state, &self.fuser);
        println!("{analyze:?}");
        let info = match analyze {
            FuserNodeAnalyse::Accept => {
                block.fuse(operation);
                self.num_ops += 1;
                block.fuser.reduce_info()
            }
            FuserNodeAnalyse::Refuse => {
                self.state = State::Closed {
                    num_ops: self.num_ops,
                };
                return;
            }
            FuserNodeAnalyse::NewBlockRequired => {
                let mut block = FuserBlock {
                    fuser: self.fuser.clone(),
                    ops: Vec::new(),
                };
                block.fuse(operation);
                self.num_ops += 1;
                let info = block.fuser.reduce_info();
                self.blocks.push(block);
                info
            }
        };

        match info {
            ReduceFuserInfo::FusedReduce {
                shape_input_id,
                axis,
            } => {
                self.state = State::Init {
                    shape_id: shape_input_id,
                    axis,
                };
            }
            ReduceFuserInfo::FusedElemwise { .. } => {}
        }
    }

    fn finish(&self) -> CubeOptimization<R> {
        let mut num_ops = 0;
        let fallbacks = self
            .blocks
            .iter()
            .map(|block| block.finish(&mut num_ops))
            .collect::<Vec<_>>();

        let info = Arc::new(ReduceBroadcastedOptimizationInfo { fallbacks });
        CubeOptimization::ReduceBroadcasted(ReduceBroadcastedOptimization { info, num_ops })
    }

    fn reset(&mut self) {
        let block = FuserBlock {
            fuser: self.fuser_default.clone(),
            ops: Vec::new(),
        };
        self.blocks = vec![block];
        self.num_ops = 0;
        self.state = State::Starting;
    }

    fn status(&self) -> FuserStatus {
        let fuser = self.blocks.last().unwrap();
        fuser.fuser.status()
    }

    fn properties(&self) -> FuserProperties {
        let mut props = FuserProperties {
            score: 0,
            ready: true,
        };
        for block in self.blocks.iter() {
            let p = block.properties();
            println!("{p:?}");
            props.score += p.score;
            props.ready = p.ready & props.ready;
        }
        println!("----- {props:?}");
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
        assert_eq!(
            fuser.properties(),
            FuserProperties {
                score: 2,
                ready: true
            }
        );

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
        assert_eq!(
            fuser.properties(),
            FuserProperties {
                score: 3,
                ready: true
            }
        );

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
        assert_eq!(
            fuser.properties(),
            FuserProperties {
                score: 4,
                ready: true
            }
        );

        let (tensor7_out, tensor7) = tensor(6, vec![1, 0], TensorStatus::ReadWrite);
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
        assert_eq!(
            fuser.properties(),
            FuserProperties {
                score: 5,
                ready: true
            }
        );
    }

    fn tensor(id: u64, shape: Vec<usize>, status: TensorStatus) -> (TensorIr, TensorIr) {
        let tensor = TensorIr {
            id: TensorId::new(id),
            shape: Shape { dims: shape },
            status: TensorStatus::NotInit,
            dtype: DType::F32,
        };
        let mut tensor_init = tensor.clone();
        tensor_init.status = TensorStatus::ReadWrite;

        (tensor, tensor_init)
    }
}
