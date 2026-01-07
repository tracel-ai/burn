use crate::{
    engine::codegen::ir::FuseType,
    optim::{
        CubeOptimization,
        reduce::{ReduceFuser, ReduceSettings},
    },
};
use burn_fusion::{FuserProperties, FuserStatus, OperationFuser};
use burn_ir::OperationIr;
use cubecl::Runtime;

/// Fuses element wise operations around a reduce operation.
pub struct ReduceBroadcastedFuser<R: Runtime> {
    fusers: Vec<ReduceFuser<R>>,
    fuser_default: ReduceFuser<R>,
    num_ops: usize,
    state: State,
}

enum State {
    Starting,
    Init { shape_id: usize, axis: usize },
    Closed { num_ops: usize },
}

impl<R: Runtime> Clone for ReduceBroadcastedFuser<R> {
    fn clone(&self) -> Self {
        Self {
            fusers: self.fusers.clone(),
            fuser_default: self.fuser_default.clone(),
            num_ops: 0,
            state: State::Starting,
        }
    }
}

impl<R: Runtime> ReduceBroadcastedFuser<R> {
    pub fn new(device: R::Device, bool_precision: FuseType) -> Self {
        let fuser = ReduceFuser::new(device, bool_precision, ReduceSettings::Always);

        Self {
            fusers: vec![fuser.clone()],
            fuser_default: fuser,
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

        let fuser = self.fusers.last_mut().unwrap();

        if let FuserStatus::Closed = fuser.status() {
            return;
        }

        println!("Fuse {operation:?}");

        let num_ops_before = fuser.len();
        fuser.fuse(operation);
        let num_ops_after = fuser.len();
        let added = num_ops_after - num_ops_before;
        self.num_ops += added;

        if added == 0 {
            self.fusers.push(self.fuser_default.clone());
            let fuser = self.fusers.last_mut().unwrap();
            let num_ops_before = fuser.len();
            fuser.fuse(operation);
            let num_ops_after = fuser.len();
            let added = num_ops_after - num_ops_before;
            self.num_ops += added;
            println!("New fusers: {}", self.fusers.len());
        };

        let fuser = self.fusers.last().unwrap();
        match &self.state {
            State::Starting => match fuser.reduce_info() {
                Some((shape_id, axis)) => self.state = State::Init { shape_id, axis },
                None => {}
            },
            State::Init { shape_id, axis } => match fuser.reduce_info() {
                Some((shape_id_new, axis_new)) => {
                    if shape_id_new != *shape_id || axis_new != *axis {
                        let removed = self.fusers.pop().unwrap();
                        let num_ops = self.num_ops - removed.len();
                        self.state = State::Closed { num_ops };
                    }
                }
                None => {}
            },
            State::Closed { .. } => {}
        }
    }

    fn finish(&self) -> CubeOptimization<R> {
        let fuser = self.fusers.last().unwrap();

        todo!();
    }

    fn reset(&mut self) {
        let mut fuser = self.fusers.remove(0);
        fuser.reset();
        let fusers = vec![fuser];
        self.num_ops = 0;
        self.state = State::Starting;
    }

    fn status(&self) -> FuserStatus {
        let fuser = self.fusers.last().unwrap();
        fuser.status()
    }

    fn properties(&self) -> FuserProperties {
        todo!()
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
    fn basic() {
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
