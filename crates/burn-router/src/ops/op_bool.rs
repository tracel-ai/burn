use alloc::vec::Vec;

use crate::{BackendRouter, OperationOutput, RunnerChannel, RunnerClient, get_client};
use burn_ir::{
    BaseOperationIr, BinaryOpIr, BoolOperationIr, CastOpIr, CatOpIr, CreationOpIr, FlipOpIr,
    InitOperationIr, OperationIr, PermuteOpIr, RepeatDimOpIr, ShapeOpIr, SliceAssignOpIr,
    SliceOpIr, SwapDimsOpIr, UnaryOpIr, UnfoldOpIr,
};
use burn_tensor::ops::{BoolTensor, BoolTensorOps, FloatElem, FloatTensor, IntElem, IntTensor};
use burn_tensor::{Device, Element, Shape, Slice, TensorData};

impl<R: RunnerChannel> BoolTensorOps<Self> for BackendRouter<R> {
    fn bool_empty(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        let client = get_client::<R>(device);
        let desc =
            CreationOpIr::create(shape, R::BoolElem::dtype(), || client.create_empty_handle());

        client
            .register(OperationIr::BaseBool(BaseOperationIr::Empty(desc)))
            .output()
    }

    fn bool_zeros(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        let client = get_client::<R>(device);
        let desc =
            CreationOpIr::create(shape, R::BoolElem::dtype(), || client.create_empty_handle());

        client
            .register(OperationIr::BaseBool(BaseOperationIr::Zeros(desc)))
            .output()
    }

    fn bool_ones(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        let client = get_client::<R>(device);
        let desc =
            CreationOpIr::create(shape, R::BoolElem::dtype(), || client.create_empty_handle());

        client
            .register(OperationIr::BaseBool(BaseOperationIr::Ones(desc)))
            .output()
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

        client.register(OperationIr::Init(desc)).output()
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = CastOpIr::create(tensor.into_ir(), IntElem::<Self>::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Bool(BoolOperationIr::IntoInt(desc)))
            .output()
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = CastOpIr::create(tensor.into_ir(), FloatElem::<Self>::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Bool(BoolOperationIr::IntoFloat(desc)))
            .output()
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
        let desc = ShapeOpIr::reshape(tensor.into_ir(), shape, || client.create_empty_handle());

        client
            .register(OperationIr::BaseBool(BaseOperationIr::Reshape(desc)))
            .output()
    }

    fn bool_slice(tensor: BoolTensor<Self>, slices: &[Slice]) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let desc = SliceOpIr::create(tensor.into_ir(), slices.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseBool(BaseOperationIr::Slice(desc)))
            .output()
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        slices: &[burn_tensor::Slice],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let desc =
            SliceAssignOpIr::create(tensor.into_ir(), slices.into(), value.into_ir(), || {
                client.create_empty_handle()
            });

        client
            .register(OperationIr::BaseBool(BaseOperationIr::SliceAssign(desc)))
            .output()
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseBool(BaseOperationIr::Equal(desc)))
            .output()
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Bool(BoolOperationIr::Not(desc)))
            .output()
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Bool(BoolOperationIr::And(desc)))
            .output()
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Bool(BoolOperationIr::Or(desc)))
            .output()
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let desc = SwapDimsOpIr::create(tensor.into_ir(), dim1, dim2, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseBool(BaseOperationIr::SwapDims(desc)))
            .output()
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let desc = PermuteOpIr::create(tensor.into_ir(), axes.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseBool(BaseOperationIr::Permute(desc)))
            .output()
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let desc = FlipOpIr::create(tensor.into_ir(), axes.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseBool(BaseOperationIr::Flip(desc)))
            .output()
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let desc = ShapeOpIr::expand(tensor.into_ir(), shape, || client.create_empty_handle());

        client
            .register(OperationIr::BaseBool(BaseOperationIr::Expand(desc)))
            .output()
    }

    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        let client = tensors.first().unwrap().client.clone();
        let tensors = tensors.into_iter().map(|t| t.into_ir()).collect();
        let desc = CatOpIr::create(tensors, dim, || client.create_empty_handle());

        client
            .register(OperationIr::BaseBool(BaseOperationIr::Cat(desc)))
            .output()
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let desc = RepeatDimOpIr::create(tensor.into_ir(), dim, times, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseBool(BaseOperationIr::RepeatDim(desc)))
            .output()
    }

    fn bool_unfold(
        tensor: BoolTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> BoolTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnfoldOpIr::create(tensor.into_ir(), dim, size, step, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseBool(BaseOperationIr::Unfold(desc)))
            .output()
    }
}
