use alloc::vec::Vec;

use burn_ir::{
    BaseOperationIr, BinaryOpIr, BoolOperationIr, CatOpIr, ExpandOpIr, FlipOpIr, InitOperationIr,
    OperationIr, PermuteOpIr, RepeatDimOpIr, SliceAssignOpIr, SliceOpIr, SwapDimsOpIr, UnaryOpIr,
};
use burn_tensor::ops::{BoolTensor, BoolTensorOps, FloatElem, FloatTensor, IntElem, IntTensor};
use burn_tensor::{Device, Element, Shape, TensorData, TensorMetadata};

use crate::{get_client, BackendRouter, RunnerChannel, RunnerClient};

impl<R: RunnerChannel> BoolTensorOps<Self> for BackendRouter<R> {
    fn bool_empty(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let out = client.register_empty_tensor(shape.into(), R::BoolElem::dtype());

        client.register(OperationIr::BaseBool(BaseOperationIr::Empty(
            out.to_ir_out(),
        )));

        out
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> TensorData {
        tensor.into_data().await
    }

    fn bool_from_data(data: TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        let client = get_client::<R>(device);
        let out = client.register_tensor_data(data);
        let desc = InitOperationIr {
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Init(desc));

        out
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), IntElem::<Self>::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Bool(BoolOperationIr::IntoInt(desc)));

        out
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), FloatElem::<Self>::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Bool(BoolOperationIr::IntoFloat(desc)));

        out
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> Device<Self> {
        tensor.client.device()
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &Device<Self>) -> BoolTensor<Self> {
        if &tensor.client.device() == device {
            return tensor;
        }
        R::change_client_backend(tensor, device)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(shape.into(), tensor.dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseBool(BaseOperationIr::Reshape(desc)));

        out
    }

    fn bool_slice(
        tensor: BoolTensor<Self>,
        ranges: &[core::ops::Range<usize>],
    ) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let ndims = tensor.shape().num_dims();
        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..ndims {
            shape.push(tensor.shape[i]);
        }

        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = SliceOpIr {
            tensor: tensor.into_ir(),
            ranges: ranges.to_vec(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseBool(BaseOperationIr::Slice(desc)));

        out
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        ranges: &[core::ops::Range<usize>],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = SliceAssignOpIr {
            tensor: tensor.into_ir(),
            ranges: ranges.to_vec(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseBool(BaseOperationIr::SliceAssign(desc)));

        out
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseBool(BaseOperationIr::Equal(desc)));

        out
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Bool(BoolOperationIr::Not(desc)));

        out
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Bool(BoolOperationIr::And(desc)));

        out
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Bool(BoolOperationIr::Or(desc)));

        out
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = SwapDimsOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            dim1,
            dim2,
        };

        client.register(OperationIr::BaseBool(BaseOperationIr::SwapDims(desc)));

        out
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        // Change the shape of the tensor to match the new axes
        let shape = axes.iter().map(|x| tensor.shape[*x]).collect();
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = PermuteOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationIr::BaseBool(BaseOperationIr::Permute(desc)));

        out
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = FlipOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationIr::BaseBool(BaseOperationIr::Flip(desc)));

        out
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let shape: Vec<_> = shape.into();
        let out = client.register_empty_tensor(shape.clone(), tensor.dtype);

        let desc = ExpandOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            shape,
        };

        client.register(OperationIr::BaseBool(BaseOperationIr::Expand(desc)));

        out
    }

    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        let tensor_first = tensors.first().unwrap();
        let client = tensor_first.client.clone();
        let dtype = tensor_first.dtype;

        // Calculate the output shape
        let mut shape = tensor_first.shape.clone();
        shape[dim] = 0;
        for tensor in tensors.iter() {
            shape[dim] += tensor.shape[dim];
        }
        let out = client.register_empty_tensor(shape, dtype);

        let desc = CatOpIr {
            tensors: tensors.into_iter().map(|t| t.into_ir()).collect(),
            dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseBool(BaseOperationIr::Cat(desc)));

        out
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let mut shape = tensor.shape.clone();
        shape[dim] *= times;
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = RepeatDimOpIr {
            tensor: tensor.into_ir(),
            dim,
            times,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseBool(BaseOperationIr::RepeatDim(desc)));

        out
    }
}
