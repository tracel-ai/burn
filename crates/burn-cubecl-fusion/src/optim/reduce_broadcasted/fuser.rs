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
    num_ops: usize,
}

impl<R: Runtime> Clone for ReduceBroadcastedFuser<R> {
    fn clone(&self) -> Self {
        Self {
            fusers: self.fusers.clone(),
            num_ops: 0,
        }
    }
}

impl<R: Runtime> ReduceBroadcastedFuser<R> {
    pub fn new(device: R::Device, bool_precision: FuseType) -> Self {
        let fuser = ReduceFuser::new(device, bool_precision, ReduceSettings::Always);

        Self {
            fusers: vec![fuser],
            num_ops: 0,
        }
    }
}

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for ReduceBroadcastedFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        let fuser = self.fusers.last_mut().unwrap();

        if let FuserStatus::Closed = fuser.status() {
            return;
        }

        let num_ops_before = fuser.len();
        println!("Fuse {operation:?}");
        fuser.fuse(operation);
        let num_ops_after = fuser.len();
        self.num_ops += num_ops_after - num_ops_before;
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
    use burn_ir::{BaseOperationIr, BinaryOpIr, CreationOpIr, TensorId, TensorIr, TensorStatus};
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
            burn_ir::NumericOperationIr::SumDim(burn_ir::ReduceDimOpIr {
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
