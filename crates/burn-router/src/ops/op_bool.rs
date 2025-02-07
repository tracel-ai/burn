use alloc::vec::Vec;

use burn_ir::{
    BaseOperationRepr, BinaryOpRepr, BoolOperationRepr, CatOpRepr, ExpandOpRepr, FlipOpRepr,
    InitOperationRepr, OperationRepr, PermuteOpRepr, RepeatDimOpRepr, SliceAssignOpRepr,
    SliceOpRepr, SwapDimsOpRepr, UnaryOpRepr,
};
use burn_tensor::ops::{BoolTensor, BoolTensorOps, FloatElem, FloatTensor, IntElem, IntTensor};
use burn_tensor::{DType, Device, Element, Shape, TensorData, TensorMetadata};

use crate::{get_client, BackendRouter, RunnerChannel, RunnerClient};

impl<R: RunnerChannel> BoolTensorOps<Self> for BackendRouter<R> {
    fn bool_empty(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let out = client.register_empty_tensor(shape.into(), DType::Bool);

        client.register(OperationRepr::BaseBool(BaseOperationRepr::Empty(
            out.to_tensor_ir_out(),
        )));

        out
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> TensorData {
        tensor.into_data().await
    }

    fn bool_from_data(data: TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        let client = get_client::<R>(device);
        let out = client.register_tensor_data(data);
        let desc = InitOperationRepr {
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Init(desc));

        out
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), IntElem::<Self>::dtype());

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Bool(BoolOperationRepr::IntoInt(desc)));

        out
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), FloatElem::<Self>::dtype());

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Bool(BoolOperationRepr::IntoFloat(desc)));

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

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseBool(BaseOperationRepr::Reshape(desc)));

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

        let desc = SliceOpRepr {
            tensor: tensor.into_tensor_ir(),
            ranges: ranges.to_vec(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseBool(BaseOperationRepr::Slice(desc)));

        out
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        ranges: &[core::ops::Range<usize>],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = SliceAssignOpRepr {
            tensor: tensor.into_tensor_ir(),
            ranges: ranges.to_vec(),
            value: value.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseBool(BaseOperationRepr::SliceAssign(
            desc,
        )));

        out
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseBool(BaseOperationRepr::Equal(desc)));

        out
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Bool(BoolOperationRepr::Not(desc)));

        out
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = SwapDimsOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
            dim1,
            dim2,
        };

        client.register(OperationRepr::BaseBool(BaseOperationRepr::SwapDims(desc)));

        out
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        // Change the shape of the tensor to match the new axes
        let shape = axes.iter().map(|x| tensor.shape[*x]).collect();
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = PermuteOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationRepr::BaseBool(BaseOperationRepr::Permute(desc)));

        out
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = FlipOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationRepr::BaseBool(BaseOperationRepr::Flip(desc)));

        out
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let shape: Vec<_> = shape.into();
        let out = client.register_empty_tensor(shape.clone(), tensor.dtype);

        let desc = ExpandOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
            shape,
        };

        client.register(OperationRepr::BaseBool(BaseOperationRepr::Expand(desc)));

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

        let desc = CatOpRepr {
            tensors: tensors.into_iter().map(|t| t.into_tensor_ir()).collect(),
            dim,
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseBool(BaseOperationRepr::Cat(desc)));

        out
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let mut shape = tensor.shape.clone();
        shape[dim] *= times;
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = RepeatDimOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            times,
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseBool(BaseOperationRepr::RepeatDim(desc)));

        out
    }
}
