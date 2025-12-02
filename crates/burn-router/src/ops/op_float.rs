use alloc::vec::Vec;
use burn_tensor::backend::{Backend, ExecutionError};

use crate::{BackendRouter, RunnerChannel, RunnerClient, get_client};
use burn_ir::{
    BaseOperationIr, BinaryOpIr, CastOpIr, CatOpIr, ClampOpIr, CreationOpIr, CrossOpIr, DimOpIr,
    FlipOpIr, FloatOperationIr, FullOpIr, GatherOpIr, InitOperationIr, MaskFillOpIr, MaskWhereOpIr,
    MatmulOpIr, NumericOperationIr, OperationIr, OperationOutput, PermuteOpIr, RandomOpIr,
    ReduceDimOpIr, ReduceDimWithIndicesOpIr, ReduceOpIr, RepeatDimOpIr, ScalarIr, ScalarOpIr,
    ScatterOpIr, SelectAssignOpIr, SelectOpIr, ShapeOpIr, SliceAssignOpIr, SliceOpIr, SwapDimsOpIr,
    UnaryOpIr, UnfoldOpIr,
};
use burn_tensor::ops::{BoolTensor, FloatElem, FloatTensor, FloatTensorOps, IntElem, IntTensor};
use burn_tensor::{
    Device, Distribution, Element, FloatDType, Shape, Slice, TensorData, IndexingUpdateOp,
};

impl<R: RunnerChannel> FloatTensorOps<Self> for BackendRouter<R> {
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        let client = get_client::<R>(device);
        let out = client.register_tensor_data(data);
        let desc = InitOperationIr {
            out: out.to_ir_out(),
        };

        // Call register op when output is already initialized
        client.register_op(OperationIr::Init(desc));

        out
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self> {
        let client = get_client::<R>(device);
        let dtype = FloatElem::<Self>::dtype();
        let desc = RandomOpIr::create(shape, dtype, distribution, || client.create_empty_handle());

        client
            .register(OperationIr::Float(dtype, FloatOperationIr::Random(desc)))
            .output()
    }

    fn float_zeros(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let client = get_client::<R>(device);
        let desc = CreationOpIr::create(shape, dtype.into(), || client.create_empty_handle());

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::Zeros(desc)))
            .output()
    }

    fn float_ones(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let client = get_client::<R>(device);
        let desc = CreationOpIr::create(shape, dtype.into(), || client.create_empty_handle());

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::Ones(desc)))
            .output()
    }

    fn float_full(
        shape: Shape,
        fill_value: FloatElem<Self>,
        device: &Device<Self>,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        let client = get_client::<R>(device);
        let dtype = dtype.into();
        let value = ScalarIr::with_dtype(fill_value, &dtype);
        let desc = FullOpIr::create(shape, dtype, value, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Full(desc),
            ))
            .output()
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> Result<TensorData, ExecutionError> {
        Ok(tensor
            .into_data()
            .await?
            // Since underlying backends can have different data types, we convert to the current elem
            .convert::<<Self as Backend>::FloatElem>())
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device<Self> {
        tensor.client.device()
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self> {
        if &tensor.client.device() == device {
            return tensor;
        }
        R::change_client_backend(tensor, device)
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = CastOpIr::create(tensor.into_ir(), IntElem::<Self>::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Float(
                desc.input.dtype,
                FloatOperationIr::IntoInt(desc),
            ))
            .output()
    }

    fn float_empty(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let client = get_client::<R>(device);
        let desc = CreationOpIr::create(shape, dtype.into(), || client.create_empty_handle());

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::Empty(desc)))
            .output()
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Add(desc),
            ))
            .output()
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::AddScalar(desc),
            ))
            .output()
    }

    fn float_clamp(
        tensor: FloatTensor<Self>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let min = ScalarIr::with_dtype(min, &tensor.dtype);
        let max = ScalarIr::with_dtype(max, &tensor.dtype);
        let desc = ClampOpIr::create(tensor.into_ir(), min, max, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Clamp(desc),
            ))
            .output()
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Sub(desc),
            ))
            .output()
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::SubScalar(desc),
            ))
            .output()
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Mul(desc),
            ))
            .output()
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::MulScalar(desc),
            ))
            .output()
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Div(desc),
            ))
            .output()
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::DivScalar(desc),
            ))
            .output()
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Rem(desc),
            ))
            .output()
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::RemScalar(desc),
            ))
            .output()
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let desc = MatmulOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Matmul(desc),
            ))
            .output()
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let desc = CrossOpIr::create(lhs.into_ir(), rhs.into_ir(), dim, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Cross(desc),
            ))
            .output()
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = SwapDimsOpIr::create(tensor.into_ir(), dim1, dim2, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::SwapDims(desc)))
            .output()
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = ShapeOpIr::reshape(tensor.into_ir(), shape, || client.create_empty_handle());

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::Reshape(desc)))
            .output()
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = GatherOpIr::create(tensor.into_ir(), dim, indices.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Gather(desc),
            ))
            .output()
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = ScatterOpIr::create(
            tensor.into_ir(),
            dim,
            indices.into_ir(),
            value.into_ir(),
            IndexingUpdateOp::Add,
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Scatter(desc),
            ))
            .output()
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = SelectOpIr::create(tensor.into_ir(), dim, indices.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Select(desc),
            ))
            .output()
    }

    fn float_select_add(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = SelectAssignOpIr::create(
            tensor.into_ir(),
            dim,
            indices.into_ir(),
            value.into_ir(),
            IndexingUpdateOp::Add,
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::SelectAssign(desc),
            ))
            .output()
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = SliceOpIr::create(tensor.into_ir(), slices.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::Slice(desc)))
            .output()
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[burn_tensor::Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc =
            SliceAssignOpIr::create(tensor.into_ir(), slices.into(), value.into_ir(), || {
                client.create_empty_handle()
            });

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::SliceAssign(desc)))
            .output()
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = MaskWhereOpIr::create(tensor.into_ir(), mask.into_ir(), value.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::MaskWhere(desc),
            ))
            .output()
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let value = ScalarIr::with_dtype(value, &tensor.dtype);
        let desc = MaskFillOpIr::create(tensor.into_ir(), mask.into_ir(), value, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::MaskFill(desc),
            ))
            .output()
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            R::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::Equal(desc)))
            .output()
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, R::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.lhs.dtype,
                NumericOperationIr::EqualElem(desc),
            ))
            .output()
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            R::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericFloat(
                desc.lhs.dtype,
                NumericOperationIr::Greater(desc),
            ))
            .output()
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, R::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.lhs.dtype,
                NumericOperationIr::GreaterElem(desc),
            ))
            .output()
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            R::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericFloat(
                desc.lhs.dtype,
                NumericOperationIr::GreaterEqual(desc),
            ))
            .output()
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, R::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.lhs.dtype,
                NumericOperationIr::GreaterEqualElem(desc),
            ))
            .output()
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            R::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericFloat(
                desc.lhs.dtype,
                NumericOperationIr::Lower(desc),
            ))
            .output()
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, R::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.lhs.dtype,
                NumericOperationIr::LowerElem(desc),
            ))
            .output()
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            R::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericFloat(
                desc.lhs.dtype,
                NumericOperationIr::LowerEqual(desc),
            ))
            .output()
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, R::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.lhs.dtype,
                NumericOperationIr::LowerEqualElem(desc),
            ))
            .output()
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Sum(desc),
            ))
            .output()
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, axis: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), axis, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::SumDim(desc),
            ))
            .output()
    }

    fn float_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Prod(desc),
            ))
            .output()
    }

    fn float_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::ProdDim(desc),
            ))
            .output()
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Mean(desc),
            ))
            .output()
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::MeanDim(desc),
            ))
            .output()
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::CumSum(desc),
            ))
            .output()
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::CumProd(desc),
            ))
            .output()
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::CumMin(desc),
            ))
            .output()
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::CumMax(desc),
            ))
            .output()
    }

    fn float_exp(lhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let desc = UnaryOpIr::create(lhs.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Exp(desc),
            ))
            .output()
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Log(desc),
            ))
            .output()
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Log1p(desc),
            ))
            .output()
    }

    fn float_powf_scalar_impl(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::PowfScalar(desc),
            ))
            .output()
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Sqrt(desc),
            ))
            .output()
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Abs(desc),
            ))
            .output()
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Cos(desc),
            ))
            .output()
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Sin(desc),
            ))
            .output()
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Tanh(desc),
            ))
            .output()
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Round(desc),
            ))
            .output()
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Floor(desc),
            ))
            .output()
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Ceil(desc),
            ))
            .output()
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Trunc(desc),
            ))
            .output()
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Recip(desc),
            ))
            .output()
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Float(
                desc.out.dtype,
                FloatOperationIr::Erf(desc),
            ))
            .output()
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        let client = tensors.first().unwrap().client.clone();
        let tensors = tensors.into_iter().map(|t| t.into_ir()).collect();
        let desc = CatOpIr::create(tensors, dim, || client.create_empty_handle());

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::Cat(desc)))
            .output()
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc =
            ReduceDimOpIr::create_arg(tensor.into_ir(), dim, IntElem::<Self>::dtype(), || {
                client.create_empty_handle()
            });

        client
            .register(OperationIr::NumericFloat(
                desc.input.dtype,
                NumericOperationIr::ArgMax(desc),
            ))
            .output()
    }

    fn float_repeat_dim(tensor: FloatTensor<Self>, dim: usize, times: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = RepeatDimOpIr::create(tensor.into_ir(), dim, times, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::RepeatDim(desc)))
            .output()
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc =
            ReduceDimOpIr::create_arg(tensor.into_ir(), dim, IntElem::<Self>::dtype(), || {
                client.create_empty_handle()
            });

        client
            .register(OperationIr::NumericFloat(
                desc.input.dtype,
                NumericOperationIr::ArgMin(desc),
            ))
            .output()
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Max(desc),
            ))
            .output()
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::MaxDim(desc),
            ))
            .output()
    }

    fn float_max_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        let client = tensor.client.clone();
        let desc = ReduceDimWithIndicesOpIr::create(
            tensor.into_ir(),
            dim,
            IntElem::<Self>::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericFloat(
                desc.tensor.dtype,
                NumericOperationIr::MaxDimWithIndices(desc),
            ))
            .outputs()
            .into()
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Min(desc),
            ))
            .output()
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::MinDim(desc),
            ))
            .output()
    }

    fn float_min_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        let client = tensor.client.clone();
        let desc = ReduceDimWithIndicesOpIr::create(
            tensor.into_ir(),
            dim,
            IntElem::<Self>::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericFloat(
                desc.tensor.dtype,
                NumericOperationIr::MinDimWithIndices(desc),
            ))
            .outputs()
            .into()
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericFloat(
                desc.out.dtype,
                NumericOperationIr::Powf(desc),
            ))
            .output()
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = PermuteOpIr::create(tensor.into_ir(), axes.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::Permute(desc)))
            .output()
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = ShapeOpIr::expand(tensor.into_ir(), shape, || client.create_empty_handle());

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::Expand(desc)))
            .output()
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = FlipOpIr::create(tensor.into_ir(), axes.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::Flip(desc)))
            .output()
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: burn_tensor::FloatDType) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = CastOpIr::create(tensor.into_ir(), dtype.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::Cast(desc)))
            .output()
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnfoldOpIr::create(tensor.into_ir(), dim, size, step, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseFloat(BaseOperationIr::Unfold(desc)))
            .output()
    }
}
