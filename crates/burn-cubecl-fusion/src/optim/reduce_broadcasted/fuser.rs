use crate::{
    engine::codegen::ir::FuseType,
    optim::{CubeOptimization, reduce::ReduceFuser},
};
use burn_fusion::{FuserStatus, OperationFuser};
use burn_ir::OperationIr;
use cubecl::Runtime;

/// Fuses element wise operations around a reduce operation.
pub struct ReduceBroadcastedFuser<R: Runtime> {
    fusers: Vec<ReduceFuser<R>>,
}

impl<R: Runtime> Clone for ReduceBroadcastedFuser<R> {
    fn clone(&self) -> Self {
        Self {
            fusers: self.fusers.clone(),
        }
    }
}

impl<R: Runtime> ReduceBroadcastedFuser<R> {
    pub fn new(device: R::Device, bool_precision: FuseType) -> Self {
        let fuser = ReduceFuser::new(device, bool_precision);

        Self {
            fusers: vec![fuser],
        }
    }
}

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for ReduceBroadcastedFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        let fuser = self.fusers.last_mut().unwrap();

        if let FuserStatus::Closed = fuser.status() {
            return;
        }

        fuser.fuse(operation);
    }

    fn finish(&self) -> CubeOptimization<R> {
        let fuser = self.fusers.last().unwrap();

        todo!();
    }

    fn reset(&mut self) {
        let mut fuser = self.fusers.remove(0);
        fuser.reset();
        let fusers = vec![fuser];
    }

    fn status(&self) -> burn_fusion::FuserStatus {
        let fuser = self.fusers.last().unwrap();
        fuser.status()
    }

    fn properties(&self) -> burn_fusion::FuserProperties {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use burn_ir::{BaseOperationIr, CreationOpIr, TensorId, TensorIr, TensorStatus};
    use burn_std::{DType, Shape};

    use super::*;

    type Run = cubecl::TestRuntime;

    #[test]
    fn basic() {
        let device: <Run as Runtime>::Device = Default::default();
        let mut fuser = ReduceBroadcastedFuser::<Run>::new(device, FuseType::I32);
        let tensor1 = TensorIr {
            id: TensorId::new(0),
            shape: Shape { dims: vec![1, 2] },
            status: TensorStatus::NotInit,
            dtype: DType::F32,
        };
        let mut tensor1_init = tensor1.clone();
        tensor1_init.status = TensorStatus::ReadWrite;

        let tensor2 = TensorIr {
            id: TensorId::new(1),
            shape: Shape { dims: vec![1, 0] },
            status: TensorStatus::NotInit,
            dtype: DType::F32,
        };

        fuser.fuse(&OperationIr::BaseFloat(BaseOperationIr::Ones(
            CreationOpIr { out: tensor1 },
        )));
        fuser.fuse(&OperationIr::NumericFloat(
            DType::F32,
            burn_ir::NumericOperationIr::SumDim(burn_ir::ReduceDimOpIr {
                input: tensor1_init,
                out: tensor2,
                axis: 1,
            }),
        ));
    }

    fn tensor(id: u64, shape: Vec<usize>, status: TensorStatus) -> (TensorIr, TensorIr) {}
        let tensor = TensorIr {
            id: TensorId::new(id),
            shape: Shape { dims: shape },
            status: TensorStatus::NotInit,
            dtype: DType::F32,
        };
        let mut tensor_init = tensor1.clone();
        tensor_init.status = TensorStatus::ReadWrite;

        (tensor, tensor_init)


}
