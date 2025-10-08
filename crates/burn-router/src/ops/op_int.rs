use alloc::vec::Vec;
use burn_tensor::backend::Backend;

use crate::{BackendRouter, RunnerChannel, RunnerClient, get_client};
use burn_ir::{
    BaseOperationIr, BinaryOpIr, CatOpIr, ClampOpIr, DimOpIr, ExpandOpIr, FlipOpIr, GatherOpIr,
    InitOperationIr, IntOperationIr, MaskFillOpIr, MaskWhereOpIr, NumericOperationIr, OperationIr,
    PermuteOpIr, RandomOpIr, ReduceDimOpIr, ReduceDimWithIndicesOpIr, RepeatDimOpIr, ScalarIr,
    ScalarOpIr, ScatterOpIr, SelectAssignOpIr, SelectOpIr, SliceAssignOpIr, SliceOpIr,
    SwapDimsOpIr, UnaryOpIr, UnfoldOpIr,
};
use burn_tensor::ops::unfold::calculate_unfold_shape;
use burn_tensor::ops::{BoolTensor, FloatElem, FloatTensor, IntElem, IntTensor, IntTensorOps};
use burn_tensor::{
    Device, Distribution, Element, IntDType, Shape, Slice, TensorData, TensorMetadata,
};

impl<R: RunnerChannel> IntTensorOps<Self> for BackendRouter<R> {
    fn int_empty(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let out = client.register_empty_tensor(shape, dtype.into());

        client.register(OperationIr::BaseInt(BaseOperationIr::Empty(
            out.to_ir_out(),
        )));

        out
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> TensorData {
        tensor
            .into_data()
            .await
            // Since underlying backends can have different data types, we convert to the current elem
            .convert::<<Self as Backend>::IntElem>()
    }

    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self> {
        let client = get_client::<R>(device);
        let out = client.register_tensor_data(data);
        let desc = InitOperationIr {
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Init(desc));

        out
    }

    fn int_device(tensor: &IntTensor<Self>) -> Device<Self> {
        tensor.client.device()
    }

    fn int_to_device(tensor: IntTensor<Self>, device: &Device<Self>) -> IntTensor<Self> {
        if &tensor.client.device() == device {
            return tensor;
        }
        R::change_client_backend(tensor, device)
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::Reshape(desc)));

        out
    }

    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;

        let shape = tensor.shape.clone().slice(slices).unwrap();
        let out = client.register_empty_tensor(shape, dtype);

        let desc = SliceOpIr {
            tensor: tensor.into_ir(),
            ranges: slices.to_vec(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::Slice(desc)));

        out
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        slices: &[burn_tensor::Slice],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = SliceAssignOpIr {
            tensor: tensor.into_ir(),
            ranges: slices.to_vec(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::SliceAssign(desc)));

        out
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;

        let mut shape = lhs.shape.broadcast(&rhs.shape).unwrap();
        let ndims = lhs.shape.num_dims();

        shape[ndims - 2] = lhs.shape[ndims - 2];
        shape[ndims - 1] = rhs.shape[ndims - 1];
        let out = client.register_empty_tensor(shape, dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::Matmul(desc)));

        out
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.broadcast(&mask.shape).unwrap(), dtype);

        let desc = MaskWhereOpIr {
            tensor: tensor.into_ir(),
            mask: mask.into_ir(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::MaskWhere(desc),
        ));

        out
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntElem<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = MaskFillOpIr {
            tensor: tensor.into_ir(),
            mask: mask.into_ir(),
            value: ScalarIr::with_dtype(value, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::MaskFill(desc),
        ));

        out
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(indices.shape.clone(), dtype);

        let desc = GatherOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Gather(desc),
        ));

        out
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = ScatterOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Scatter(desc),
        ));

        out
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = indices.shape[0];
        let out = client.register_empty_tensor(shape, dtype);

        let desc = SelectOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Select(desc),
        ));

        out
    }

    fn int_select_assign(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = SelectAssignOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::SelectAssign(desc),
        ));

        out
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        let tensor_first = tensors.first().unwrap();
        let client = tensor_first.client.clone();
        let dtype = tensor_first.dtype;

        // Calculate the output shape
        let shape = Shape::cat(tensors.iter().map(|t| &t.shape), dim).unwrap();
        let out = client.register_empty_tensor(shape, dtype);

        let desc = CatOpIr {
            tensors: tensors.into_iter().map(|t| t.into_ir()).collect(),
            dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::Cat(desc)));

        out
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let out = client.register_empty_tensor(
            lhs.shape.broadcast(&rhs.shape).unwrap(),
            R::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::Equal(desc)));

        out
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::EqualElem(desc),
        ));

        out
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(
            lhs.shape.broadcast(&rhs.shape).unwrap(),
            R::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Greater(desc),
        ));

        out
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::GreaterElem(desc),
        ));

        out
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(
            lhs.shape.broadcast(&rhs.shape).unwrap(),
            R::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::GreaterEqual(desc),
        ));

        out
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::GreaterEqualElem(desc),
        ));

        out
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(
            lhs.shape.broadcast(&rhs.shape).unwrap(),
            R::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Lower(desc),
        ));

        out
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::LowerElem(desc),
        ));

        out
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(
            lhs.shape.broadcast(&rhs.shape).unwrap(),
            R::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::LowerEqual(desc),
        ));

        out
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::LowerEqualElem(desc),
        ));

        out
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.broadcast(&rhs.shape).unwrap(), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Add(desc),
        ));

        out
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::AddScalar(desc),
        ));

        out
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.broadcast(&rhs.shape).unwrap(), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Sub(desc),
        ));

        out
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::SubScalar(desc),
        ));

        out
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.broadcast(&rhs.shape).unwrap(), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Mul(desc),
        ));

        out
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::MulScalar(desc),
        ));

        out
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.broadcast(&rhs.shape).unwrap(), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Div(desc),
        ));

        out
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::DivScalar(desc),
        ));

        out
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.broadcast(&rhs.shape).unwrap(), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Rem(desc),
        ));

        out
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::RemScalar(desc),
        ));

        out
    }

    fn int_zeros(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = dtype.into();
        let out = client.register_empty_tensor(shape, dtype);

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Zeros(out.to_ir_out()),
        ));

        out
    }

    fn int_ones(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = dtype.into();
        let out = client.register_empty_tensor(shape, dtype);

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Ones(out.to_ir_out()),
        ));

        out
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(Shape::new([1]), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Sum(desc),
        ));

        out
    }

    fn int_sum_dim(tensor: IntTensor<Self>, axis: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[axis] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            axis,
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::SumDim(desc),
        ));

        out
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(Shape::new([1]), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Prod(desc),
        ));

        out
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::ProdDim(desc),
        ));

        out
    }

    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(Shape::new([1]), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Mean(desc),
        ));

        out
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::MeanDim(desc),
        ));

        out
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let shape = tensor.shape.clone();
        let out = client.register_empty_tensor(shape, dtype);

        let desc = DimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::CumSum(desc)));

        out
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, IntElem::<Self>::dtype());

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::ArgMax(desc),
        ));

        out
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, IntElem::<Self>::dtype());

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::ArgMin(desc),
        ));

        out
    }

    fn int_clamp(
        tensor: IntTensor<Self>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = ClampOpIr {
            tensor: tensor.into_ir(),
            min: ScalarIr::with_dtype(min, &dtype),
            max: ScalarIr::with_dtype(max, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Clamp(desc),
        ));

        out
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Abs(desc),
        ));

        out
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), FloatElem::<Self>::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::IntoFloat(desc)));

        out
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let shape = tensor.shape.clone().swap(dim1, dim2).unwrap();
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = SwapDimsOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            dim1,
            dim2,
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::SwapDims(desc)));

        out
    }

    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(Shape::new([1]), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Max(desc),
        ));

        out
    }

    fn int_max_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::MaxDim(desc),
        ));

        out
    }

    fn int_max_dim_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape.clone(), dtype);
        let out_indices = client.register_empty_tensor(shape, IntElem::<Self>::dtype());

        let desc = ReduceDimWithIndicesOpIr {
            tensor: tensor.into_ir(),
            dim,
            out: out.to_ir_out(),
            out_indices: out_indices.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::MaxDimWithIndices(desc),
        ));

        (out, out_indices)
    }

    fn int_max_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(Shape::new([1]), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::MaxAbs(desc),
        ));

        out
    }

    fn int_max_abs_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::MaxAbsDim(desc),
        ));

        out
    }

    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(Shape::new([1]), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::Min(desc),
        ));

        out
    }

    fn int_min_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::MinDim(desc),
        ));

        out
    }

    fn int_min_dim_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape.clone(), dtype);
        let out_indices = client.register_empty_tensor(shape, IntElem::<Self>::dtype());

        let desc = ReduceDimWithIndicesOpIr {
            tensor: tensor.into_ir(),
            dim,
            out: out.to_ir_out(),
            out_indices: out_indices.to_ir_out(),
        };

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::MinDimWithIndices(desc),
        ));

        (out, out_indices)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = IntElem::<Self>::dtype();
        let out = client.register_empty_tensor(shape, dtype);

        client.register(OperationIr::NumericInt(
            dtype,
            NumericOperationIr::IntRandom(RandomOpIr {
                out: out.to_ir_out(),
                distribution,
            }),
        ));

        out
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        let client = tensor.client.clone();
        // Change the shape of the tensor to match the new axes
        let shape = tensor.shape.clone().permute(axes).unwrap();
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = PermuteOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::Permute(desc)));

        out
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(shape.clone(), tensor.dtype);

        let desc = ExpandOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            shape,
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::Expand(desc)));

        out
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = FlipOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::Flip(desc)));

        out
    }

    fn int_repeat_dim(tensor: IntTensor<Self>, dim: usize, times: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let shape = tensor.shape.clone().repeat(dim, times);
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = RepeatDimOpIr {
            tensor: tensor.into_ir(),
            dim,
            times,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::RepeatDim(desc)));

        out
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.broadcast(&rhs.shape).unwrap(), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::BitwiseAnd(desc)));

        out
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.broadcast(&rhs.shape).unwrap(), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::BitwiseOr(desc)));

        out
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.broadcast(&rhs.shape).unwrap(), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::BitwiseXor(desc)));

        out
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::BitwiseNot(desc)));

        out
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::BitwiseAndScalar(desc)));

        out
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::BitwiseOrScalar(desc)));

        out
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::BitwiseXorScalar(desc)));

        out
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.broadcast(&rhs.shape).unwrap(), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::BitwiseLeftShift(desc)));

        out
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::BitwiseLeftShiftScalar(
            desc,
        )));

        out
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.broadcast(&rhs.shape).unwrap(), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::BitwiseRightShift(desc)));

        out
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Int(IntOperationIr::BitwiseRightShiftScalar(
            desc,
        )));

        out
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: burn_tensor::IntDType) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype.into());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::Cast(desc)));

        out
    }

    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();

        let shape = calculate_unfold_shape(tensor.shape(), dim, size, step);
        let out = client.register_empty_tensor(Shape::from(shape), tensor.dtype);

        let desc = UnfoldOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            dim,
            size,
            step,
        };

        client.register(OperationIr::BaseInt(BaseOperationIr::Unfold(desc)));

        out
    }
}
