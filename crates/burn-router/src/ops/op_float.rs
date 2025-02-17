use alloc::{vec, vec::Vec};
use burn_tensor::backend::Backend;
use core::ops::Range;

use burn_ir::{
    BaseOperationIr, BinaryOpIr, CatOpIr, ClampOpIr, ExpandOpIr, FlipOpIr, FloatOperationIr,
    GatherOpIr, InitOperationIr, MaskFillOpIr, MaskWhereOpIr, NumericOperationIr, OperationIr,
    PermuteOpIr, RandomOpIr, ReduceDimWithIndicesOpIr, RepeatDimOpIr, ScalarOpIr, ScatterOpIr,
    SelectAssignOpIr, SelectOpIr, SliceAssignOpIr, SliceOpIr, SwapDimsOpIr, UnaryOpIr,
};
use burn_tensor::ops::{
    binary_ops_shape, BoolTensor, FloatElem, FloatTensor, FloatTensorOps, IntElem, IntTensor,
};
use burn_tensor::{
    Device, Distribution, Element, ElementConversion, Shape, TensorData, TensorMetadata,
};

use crate::{get_client, BackendRouter, RunnerChannel, RunnerClient};

impl<R: RunnerChannel> FloatTensorOps<Self> for BackendRouter<R> {
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        let client = get_client::<R>(device);
        let out = client.register_tensor_data(data);
        let desc = InitOperationIr {
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Init(desc));

        out
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = FloatElem::<Self>::dtype();
        let out = client.register_empty_tensor(shape.into(), dtype);

        client.register(OperationIr::Float(
            dtype,
            FloatOperationIr::Random(RandomOpIr {
                out: out.to_ir_out(),
                distribution,
            }),
        ));

        out
    }

    fn float_zeros(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = FloatElem::<Self>::dtype();
        let out = client.register_empty_tensor(shape.into(), dtype);

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Zeros(out.to_ir_out()),
        ));

        out
    }

    fn float_ones(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = FloatElem::<Self>::dtype();
        let out = client.register_empty_tensor(shape.into(), dtype);

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Ones(out.to_ir_out()),
        ));

        out
    }

    fn float_full(
        shape: Shape,
        fill_value: FloatElem<Self>,
        device: &Device<Self>,
    ) -> FloatTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = FloatElem::<Self>::dtype();
        let out = client.register_empty_tensor(shape.into(), dtype);

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Full((out.to_ir_out(), fill_value.elem())),
        ));

        out
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> TensorData {
        tensor
            .into_data()
            .await
            // Since underlying backends can have different data types, we convert to the current elem
            .convert::<<Self as Backend>::FloatElem>()
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
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), IntElem::<Self>::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::IntoInt(desc)));

        out
    }

    fn float_empty(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let out = client.register_empty_tensor(shape.into(), FloatElem::<Self>::dtype());

        client.register(OperationIr::BaseFloat(BaseOperationIr::Empty(
            out.to_ir_out(),
        )));

        out
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Add(desc),
        ));

        out
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::AddScalar(desc),
        ));

        out
    }

    fn float_clamp(
        tensor: FloatTensor<Self>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = ClampOpIr {
            tensor: tensor.into_ir(),
            min: min.elem(),
            max: max.elem(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Clamp(desc),
        ));

        out
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Sub(desc),
        ));

        out
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::SubScalar(desc),
        ));

        out
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Mul(desc),
        ));

        out
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::MulScalar(desc),
        ));

        out
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Div(desc),
        ));

        out
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::DivScalar(desc),
        ));

        out
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Rem(desc),
        ));

        out
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::RemScalar(desc),
        ));

        out
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;

        let mut shape = binary_ops_shape(&lhs.shape, &rhs.shape);
        let ndims = lhs.shape().num_dims();

        shape[ndims - 2] = lhs.shape[ndims - 2];
        shape[ndims - 1] = rhs.shape[ndims - 1];
        let out = client.register_empty_tensor(shape, dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Matmul(desc)));

        out
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
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

        client.register(OperationIr::BaseFloat(BaseOperationIr::SwapDims(desc)));

        out
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(shape.into(), tensor.dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseFloat(BaseOperationIr::Reshape(desc)));

        out
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(indices.shape.clone(), dtype);

        let desc = GatherOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Gather(desc),
        ));

        out
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
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

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Scatter(desc),
        ));

        out
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
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

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Select(desc),
        ));

        out
    }

    fn float_select_assign(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
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

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::SelectAssign(desc),
        ));

        out
    }

    fn float_slice(tensor: FloatTensor<Self>, ranges: &[Range<usize>]) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;

        let ndims = tensor.shape().num_dims();
        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..ndims {
            shape.push(tensor.shape[i]);
        }

        let out = client.register_empty_tensor(shape, dtype);

        let desc = SliceOpIr {
            tensor: tensor.into_ir(),
            ranges: ranges.to_vec(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseFloat(BaseOperationIr::Slice(desc)));

        out
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        ranges: &[Range<usize>],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = SliceAssignOpIr {
            tensor: tensor.into_ir(),
            ranges: ranges.to_vec(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseFloat(BaseOperationIr::SliceAssign(desc)));

        out
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let shape = binary_ops_shape(&tensor.shape, &mask.shape);
        let out = client.register_empty_tensor(shape, dtype);

        let desc = MaskWhereOpIr {
            tensor: tensor.into_ir(),
            mask: mask.into_ir(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::MaskWhere(desc),
        ));

        out
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = MaskFillOpIr {
            tensor: tensor.into_ir(),
            mask: mask.into_ir(),
            value: value.elem(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::MaskFill(desc),
        ));

        out
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let out = client.register_empty_tensor(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            R::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseFloat(BaseOperationIr::Equal(desc)));

        out
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::EqualElem(desc),
        ));

        out
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            R::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Greater(desc),
        ));

        out
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::GreaterElem(desc),
        ));

        out
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            R::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::GreaterEqual(desc),
        ));

        out
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::GreaterEqualElem(desc),
        ));

        out
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            R::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Lower(desc),
        ));

        out
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::LowerElem(desc),
        ));

        out
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            R::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::LowerEqual(desc),
        ));

        out
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), R::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.elem(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::LowerEqualElem(desc),
        ));

        out
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Sum(desc),
        ));

        out
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::SumDim(desc),
        ));

        out
    }

    fn float_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Prod(desc),
        ));

        out
    }

    fn float_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::ProdDim(desc),
        ));

        out
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Mean(desc),
        ));

        out
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::MeanDim(desc),
        ));

        out
    }

    fn float_exp(lhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: lhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Exp(desc)));

        out
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Log(desc)));

        out
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Log1p(desc)));

        out
    }

    fn float_powf_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(
            dtype,
            FloatOperationIr::PowfScalar(desc),
        ));

        out
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Sqrt(desc)));

        out
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Abs(desc),
        ));

        out
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Cos(desc)));

        out
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Sin(desc)));

        out
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Tanh(desc)));

        out
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Round(desc)));

        out
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Floor(desc)));

        out
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Ceil(desc)));

        out
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Recip(desc)));

        out
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::Float(dtype, FloatOperationIr::Erf(desc)));

        out
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        let tensor_first = tensors.first().unwrap();
        let client = tensor_first.client.clone();

        // Calculate the output shape
        let mut shape = tensor_first.shape.clone();
        shape[dim] = 0;
        for tensor in tensors.iter() {
            shape[dim] += tensor.shape[dim];
        }
        let out = client.register_empty_tensor(shape, tensor_first.dtype);

        let desc = CatOpIr {
            tensors: tensors.into_iter().map(|tensor| tensor.into_ir()).collect(),
            dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseFloat(BaseOperationIr::Cat(desc)));

        out
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, IntElem::<Self>::dtype());

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::ArgMax(desc),
        ));

        out
    }

    fn float_repeat_dim(tensor: FloatTensor<Self>, dim: usize, times: usize) -> FloatTensor<Self> {
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

        client.register(OperationIr::BaseFloat(BaseOperationIr::RepeatDim(desc)));

        out
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, IntElem::<Self>::dtype());

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::ArgMin(desc),
        ));

        out
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Max(desc),
        ));

        out
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::MaxDim(desc),
        ));

        out
    }

    fn float_max_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
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

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::MaxDimWithIndices(desc),
        ));

        (out, out_indices)
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Min(desc),
        ));

        out
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOpIr {
            lhs: tensor.into_ir(),
            rhs: dim,
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::MinDim(desc),
        ));

        out
    }

    fn float_min_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
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

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::MinDimWithIndices(desc),
        ));

        (out, out_indices)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::NumericFloat(
            dtype,
            NumericOperationIr::Powf(desc),
        ));

        out
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        // Change the shape of the tensor to match the new axes
        let shape = axes.iter().map(|x| tensor.shape[*x]).collect();
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = PermuteOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationIr::BaseFloat(BaseOperationIr::Permute(desc)));

        out
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let shape: Vec<_> = shape.into();
        let out = client.register_empty_tensor(shape.clone(), tensor.dtype);

        let desc = ExpandOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            shape,
        };

        client.register(OperationIr::BaseFloat(BaseOperationIr::Expand(desc)));

        out
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = FlipOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationIr::BaseFloat(BaseOperationIr::Flip(desc)));

        out
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: burn_tensor::FloatDType) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_float_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        client.register(OperationIr::BaseFloat(BaseOperationIr::Cast(desc)));

        out
    }
}
