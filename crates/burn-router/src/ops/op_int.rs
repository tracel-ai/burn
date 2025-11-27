use alloc::vec::Vec;
use burn_tensor::backend::{Backend, DeferedError};

use crate::{BackendRouter, RunnerChannel, RunnerClient, get_client};
use burn_ir::{
    BaseOperationIr, BinaryOpIr, CastOpIr, CatOpIr, ClampOpIr, CreationOpIr, DimOpIr, FlipOpIr,
    GatherOpIr, InitOperationIr, IntOperationIr, MaskFillOpIr, MaskWhereOpIr, MatmulOpIr,
    NumericOperationIr, OperationIr, OperationOutput, PermuteOpIr, RandomOpIr, ReduceDimOpIr,
    ReduceDimWithIndicesOpIr, ReduceOpIr, RepeatDimOpIr, ScalarIr, ScalarOpIr, ScatterOpIr,
    SelectAssignOpIr, SelectOpIr, ShapeOpIr, SliceAssignOpIr, SliceOpIr, SwapDimsOpIr, UnaryOpIr,
    UnfoldOpIr,
};
use burn_tensor::ops::{BoolTensor, FloatElem, FloatTensor, IntElem, IntTensor, IntTensorOps};
use burn_tensor::{Device, Distribution, Element, IntDType, Shape, Slice, TensorData};

impl<R: RunnerChannel> IntTensorOps<Self> for BackendRouter<R> {
    fn int_empty(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let client = get_client::<R>(device);
        let desc = CreationOpIr::create(shape, dtype.into(), || client.create_empty_handle());

        client
            .register(OperationIr::BaseInt(BaseOperationIr::Empty(desc)))
            .output()
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> Result<TensorData, DeferedError> {
        Ok(tensor
            .into_data()
            .await?
            // Since underlying backends can have different data types, we convert to the current elem
            .convert::<<Self as Backend>::IntElem>())
    }

    fn int_from_data(data: TensorData, device: &Device<Self>) -> IntTensor<Self> {
        let client = get_client::<R>(device);
        let out = client.register_tensor_data(data);
        let desc = InitOperationIr {
            out: out.to_ir_out(),
        };

        // Call register op when output is already initialized
        client.register_op(OperationIr::Init(desc));

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
        let desc = ShapeOpIr::reshape(tensor.into_ir(), shape, || client.create_empty_handle());

        client
            .register(OperationIr::BaseInt(BaseOperationIr::Reshape(desc)))
            .output()
    }

    fn int_slice(tensor: IntTensor<Self>, slices: &[Slice]) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = SliceOpIr::create(tensor.into_ir(), slices.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseInt(BaseOperationIr::Slice(desc)))
            .output()
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        slices: &[burn_tensor::Slice],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc =
            SliceAssignOpIr::create(tensor.into_ir(), slices.into(), value.into_ir(), || {
                client.create_empty_handle()
            });

        client
            .register(OperationIr::BaseInt(BaseOperationIr::SliceAssign(desc)))
            .output()
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let desc = MatmulOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Int(IntOperationIr::Matmul(desc)))
            .output()
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = MaskWhereOpIr::create(tensor.into_ir(), mask.into_ir(), value.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::MaskWhere(desc),
            ))
            .output()
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntElem<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let value = ScalarIr::with_dtype(value, &tensor.dtype);
        let desc = MaskFillOpIr::create(tensor.into_ir(), mask.into_ir(), value, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::MaskFill(desc),
            ))
            .output()
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = GatherOpIr::create(tensor.into_ir(), dim, indices.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Gather(desc),
            ))
            .output()
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ScatterOpIr::create(
            tensor.into_ir(),
            dim,
            indices.into_ir(),
            value.into_ir(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Scatter(desc),
            ))
            .output()
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = SelectOpIr::create(tensor.into_ir(), dim, indices.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Select(desc),
            ))
            .output()
    }

    fn int_select_assign(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = SelectAssignOpIr::create(
            tensor.into_ir(),
            dim,
            indices.into_ir(),
            value.into_ir(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::SelectAssign(desc),
            ))
            .output()
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        let client = tensors.first().unwrap().client.clone();
        let tensors = tensors.into_iter().map(|t| t.into_ir()).collect();
        let desc = CatOpIr::create(tensors, dim, || client.create_empty_handle());

        client
            .register(OperationIr::BaseInt(BaseOperationIr::Cat(desc)))
            .output()
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            R::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::BaseInt(BaseOperationIr::Equal(desc)))
            .output()
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, R::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.lhs.dtype,
                NumericOperationIr::EqualElem(desc),
            ))
            .output()
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            R::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericInt(
                desc.lhs.dtype,
                NumericOperationIr::Greater(desc),
            ))
            .output()
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, R::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.lhs.dtype,
                NumericOperationIr::GreaterElem(desc),
            ))
            .output()
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            R::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericInt(
                desc.lhs.dtype,
                NumericOperationIr::GreaterEqual(desc),
            ))
            .output()
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, R::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.lhs.dtype,
                NumericOperationIr::GreaterEqualElem(desc),
            ))
            .output()
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            R::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericInt(
                desc.lhs.dtype,
                NumericOperationIr::Lower(desc),
            ))
            .output()
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, R::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.lhs.dtype,
                NumericOperationIr::LowerElem(desc),
            ))
            .output()
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            R::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericInt(
                desc.lhs.dtype,
                NumericOperationIr::LowerEqual(desc),
            ))
            .output()
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, R::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.lhs.dtype,
                NumericOperationIr::LowerEqualElem(desc),
            ))
            .output()
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Add(desc),
            ))
            .output()
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::AddScalar(desc),
            ))
            .output()
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Sub(desc),
            ))
            .output()
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::SubScalar(desc),
            ))
            .output()
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Mul(desc),
            ))
            .output()
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::MulScalar(desc),
            ))
            .output()
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Div(desc),
            ))
            .output()
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::DivScalar(desc),
            ))
            .output()
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Rem(desc),
            ))
            .output()
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::RemScalar(desc),
            ))
            .output()
    }

    fn int_zeros(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let client = get_client::<R>(device);
        let desc = CreationOpIr::create(shape, dtype.into(), || client.create_empty_handle());

        client
            .register(OperationIr::BaseInt(BaseOperationIr::Zeros(desc)))
            .output()
    }

    fn int_ones(shape: Shape, device: &Device<Self>, dtype: IntDType) -> IntTensor<Self> {
        let client = get_client::<R>(device);
        let desc = CreationOpIr::create(shape, dtype.into(), || client.create_empty_handle());

        client
            .register(OperationIr::BaseInt(BaseOperationIr::Ones(desc)))
            .output()
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Sum(desc),
            ))
            .output()
    }

    fn int_sum_dim(tensor: IntTensor<Self>, axis: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), axis, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::SumDim(desc),
            ))
            .output()
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Prod(desc),
            ))
            .output()
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::ProdDim(desc),
            ))
            .output()
    }

    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Mean(desc),
            ))
            .output()
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::MeanDim(desc),
            ))
            .output()
    }

    fn int_cumsum(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::CumSum(desc),
            ))
            .output()
    }

    fn int_cumprod(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::CumProd(desc),
            ))
            .output()
    }

    fn int_cummin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::CumMin(desc),
            ))
            .output()
    }

    fn int_cummax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::CumMax(desc),
            ))
            .output()
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::ArgMax(desc),
            ))
            .output()
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::ArgMin(desc),
            ))
            .output()
    }

    fn int_clamp(
        tensor: IntTensor<Self>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let min = ScalarIr::with_dtype(min, &tensor.dtype);
        let max = ScalarIr::with_dtype(max, &tensor.dtype);
        let desc = ClampOpIr::create(tensor.into_ir(), min, max, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Clamp(desc),
            ))
            .output()
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Abs(desc),
            ))
            .output()
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let desc = CastOpIr::create(tensor.into_ir(), FloatElem::<Self>::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Int(IntOperationIr::IntoFloat(desc)))
            .output()
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = SwapDimsOpIr::create(tensor.into_ir(), dim1, dim2, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseInt(BaseOperationIr::SwapDims(desc)))
            .output()
    }

    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Max(desc),
            ))
            .output()
    }

    fn int_max_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::MaxDim(desc),
            ))
            .output()
    }

    fn int_max_dim_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        let client = tensor.client.clone();
        let desc = ReduceDimWithIndicesOpIr::create(
            tensor.into_ir(),
            dim,
            IntElem::<Self>::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericInt(
                desc.tensor.dtype,
                NumericOperationIr::MaxDimWithIndices(desc),
            ))
            .outputs()
            .into()
    }

    fn int_max_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::MaxAbs(desc),
            ))
            .output()
    }

    fn int_max_abs_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::MaxAbsDim(desc),
            ))
            .output()
    }

    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::Min(desc),
            ))
            .output()
    }

    fn int_min_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::MinDim(desc),
            ))
            .output()
    }

    fn int_min_dim_with_indices(
        tensor: IntTensor<Self>,
        dim: usize,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        let client = tensor.client.clone();
        let desc = ReduceDimWithIndicesOpIr::create(
            tensor.into_ir(),
            dim,
            IntElem::<Self>::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(OperationIr::NumericInt(
                desc.out.dtype,
                NumericOperationIr::MinDimWithIndices(desc),
            ))
            .outputs()
            .into()
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self> {
        let client = get_client::<R>(device);
        let dtype = IntElem::<Self>::dtype();
        let desc = RandomOpIr::create(shape, dtype, distribution, || client.create_empty_handle());

        client
            .register(OperationIr::NumericInt(
                dtype,
                NumericOperationIr::IntRandom(desc),
            ))
            .output()
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = PermuteOpIr::create(tensor.into_ir(), axes.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseInt(BaseOperationIr::Permute(desc)))
            .output()
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = ShapeOpIr::expand(tensor.into_ir(), shape, || client.create_empty_handle());

        client
            .register(OperationIr::BaseInt(BaseOperationIr::Expand(desc)))
            .output()
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = FlipOpIr::create(tensor.into_ir(), axes.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseInt(BaseOperationIr::Flip(desc)))
            .output()
    }

    fn int_repeat_dim(tensor: IntTensor<Self>, dim: usize, times: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = RepeatDimOpIr::create(tensor.into_ir(), dim, times, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseInt(BaseOperationIr::RepeatDim(desc)))
            .output()
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Int(IntOperationIr::BitwiseAnd(desc)))
            .output()
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Int(IntOperationIr::BitwiseOr(desc)))
            .output()
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Int(IntOperationIr::BitwiseXor(desc)))
            .output()
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(OperationIr::Int(IntOperationIr::BitwiseNot(desc)))
            .output()
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::Int(IntOperationIr::BitwiseAndScalar(desc)))
            .output()
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::Int(IntOperationIr::BitwiseOrScalar(desc)))
            .output()
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::Int(IntOperationIr::BitwiseXorScalar(desc)))
            .output()
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Int(IntOperationIr::BitwiseLeftShift(desc)))
            .output()
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::Int(IntOperationIr::BitwiseLeftShiftScalar(
                desc,
            )))
            .output()
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::Int(IntOperationIr::BitwiseRightShift(desc)))
            .output()
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(OperationIr::Int(IntOperationIr::BitwiseRightShiftScalar(
                desc,
            )))
            .output()
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: burn_tensor::IntDType) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = CastOpIr::create(tensor.into_ir(), dtype.into(), || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseInt(BaseOperationIr::Cast(desc)))
            .output()
    }

    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let desc = UnfoldOpIr::create(tensor.into_ir(), dim, size, step, || {
            client.create_empty_handle()
        });

        client
            .register(OperationIr::BaseInt(BaseOperationIr::Unfold(desc)))
            .output()
    }
}
