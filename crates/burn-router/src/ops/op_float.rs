use alloc::{vec, vec::Vec};
use burn_tensor::backend::Backend;
use core::ops::Range;

use burn_ir::{
    BaseOperationRepr, BinaryOpRepr, CatOpRepr, ClampOpRepr, ExpandOpRepr, FlipOpRepr,
    FloatOperationRepr, GatherOpRepr, InitOperationRepr, MaskFillOpRepr, MaskWhereOpRepr,
    NumericOperationRepr, OperationRepr, PermuteOpRepr, RandomOpRepr, ReduceDimWithIndicesOpRepr,
    RepeatDimOpRepr, ScalarOpRepr, ScatterOpRepr, SelectAssignOpRepr, SelectOpRepr,
    SliceAssignOpRepr, SliceOpRepr, SwapDimsOpRepr, UnaryOpRepr,
};
use burn_tensor::ops::{
    binary_ops_shape, BoolTensor, FloatElem, FloatTensor, FloatTensorOps, IntElem, IntTensor,
};
use burn_tensor::{
    DType, Device, Distribution, Element, ElementConversion, Shape, TensorData, TensorMetadata,
};

use crate::{get_client, BackendRouter, RunnerChannel, RunnerClient};

impl<R: RunnerChannel> FloatTensorOps<Self> for BackendRouter<R> {
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        let client = get_client::<R>(device);
        let out = client.register_tensor_data(data);
        let desc = InitOperationRepr {
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Init(desc));

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

        client.register(OperationRepr::Float(
            dtype,
            FloatOperationRepr::Random(RandomOpRepr {
                out: out.to_tensor_ir_out(),
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

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Zeros(out.to_tensor_ir_out()),
        ));

        out
    }

    fn float_ones(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = FloatElem::<Self>::dtype();
        let out = client.register_empty_tensor(shape.into(), dtype);

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Ones(out.to_tensor_ir_out()),
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

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Full((out.to_tensor_ir_out(), fill_value.elem())),
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

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(
            dtype,
            FloatOperationRepr::IntoInt(desc),
        ));

        out
    }

    fn float_empty(shape: Shape, device: &Device<Self>) -> FloatTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let out = client.register_empty_tensor(shape.into(), FloatElem::<Self>::dtype());

        client.register(OperationRepr::BaseFloat(BaseOperationRepr::Empty(
            out.to_tensor_ir_out(),
        )));

        out
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Add(desc),
        ));

        out
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::AddScalar(desc),
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

        let desc = ClampOpRepr {
            tensor: tensor.into_tensor_ir(),
            min: min.elem(),
            max: max.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Clamp(desc),
        ));

        out
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Sub(desc),
        ));

        out
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::SubScalar(desc),
        ));

        out
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Mul(desc),
        ));

        out
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::MulScalar(desc),
        ));

        out
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Div(desc),
        ));

        out
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::DivScalar(desc),
        ));

        out
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Rem(desc),
        ));

        out
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::RemScalar(desc),
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

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(
            dtype,
            FloatOperationRepr::Matmul(desc),
        ));

        out
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
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

        client.register(OperationRepr::BaseFloat(BaseOperationRepr::SwapDims(desc)));

        out
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(shape.into(), tensor.dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseFloat(BaseOperationRepr::Reshape(desc)));

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

        let desc = GatherOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            indices: indices.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Gather(desc),
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

        let desc = ScatterOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            indices: indices.into_tensor_ir(),
            value: value.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Scatter(desc),
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

        let desc = SelectOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            indices: indices.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Select(desc),
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

        let desc = SelectAssignOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            indices: indices.into_tensor_ir(),
            value: value.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::SelectAssign(desc),
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

        let desc = SliceOpRepr {
            tensor: tensor.into_tensor_ir(),
            ranges: ranges.to_vec(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseFloat(BaseOperationRepr::Slice(desc)));

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

        let desc = SliceAssignOpRepr {
            tensor: tensor.into_tensor_ir(),
            ranges: ranges.to_vec(),
            value: value.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseFloat(BaseOperationRepr::SliceAssign(
            desc,
        )));

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

        let desc = MaskWhereOpRepr {
            tensor: tensor.into_tensor_ir(),
            mask: mask.into_tensor_ir(),
            value: value.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::MaskWhere(desc),
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

        let desc = MaskFillOpRepr {
            tensor: tensor.into_tensor_ir(),
            mask: mask.into_tensor_ir(),
            value: value.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::MaskFill(desc),
        ));

        out
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseFloat(BaseOperationRepr::Equal(desc)));

        out
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::EqualElem(desc),
        ));

        out
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Greater(desc),
        ));

        out
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::GreaterElem(desc),
        ));

        out
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::GreaterEqual(desc),
        ));

        out
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::GreaterEqualElem(desc),
        ));

        out
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Lower(desc),
        ));

        out
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::LowerElem(desc),
        ));

        out
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::LowerEqual(desc),
        ));

        out
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::LowerEqualElem(desc),
        ));

        out
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Sum(desc),
        ));

        out
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOpRepr {
            lhs: tensor.into_tensor_ir(),
            rhs: dim,
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::SumDim(desc),
        ));

        out
    }

    fn float_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Prod(desc),
        ));

        out
    }

    fn float_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOpRepr {
            lhs: tensor.into_tensor_ir(),
            rhs: dim,
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::ProdDim(desc),
        ));

        out
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Mean(desc),
        ));

        out
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOpRepr {
            lhs: tensor.into_tensor_ir(),
            rhs: dim,
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::MeanDim(desc),
        ));

        out
    }

    fn float_exp(lhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: lhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(dtype, FloatOperationRepr::Exp(desc)));

        out
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(dtype, FloatOperationRepr::Log(desc)));

        out
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(dtype, FloatOperationRepr::Log1p(desc)));

        out
    }

    fn float_powf_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs,
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(
            dtype,
            FloatOperationRepr::PowfScalar(desc),
        ));

        out
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(dtype, FloatOperationRepr::Sqrt(desc)));

        out
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Abs(desc),
        ));

        out
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(dtype, FloatOperationRepr::Cos(desc)));

        out
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(dtype, FloatOperationRepr::Sin(desc)));

        out
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(dtype, FloatOperationRepr::Tanh(desc)));

        out
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(dtype, FloatOperationRepr::Round(desc)));

        out
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(dtype, FloatOperationRepr::Floor(desc)));

        out
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(dtype, FloatOperationRepr::Ceil(desc)));

        out
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(dtype, FloatOperationRepr::Recip(desc)));

        out
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Float(dtype, FloatOperationRepr::Erf(desc)));

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

        let desc = CatOpRepr {
            tensors: tensors
                .into_iter()
                .map(|tensor| tensor.into_tensor_ir())
                .collect(),
            dim,
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseFloat(BaseOperationRepr::Cat(desc)));

        out
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, IntElem::<Self>::dtype());

        let desc = ScalarOpRepr {
            lhs: tensor.into_tensor_ir(),
            rhs: dim,
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::ArgMax(desc),
        ));

        out
    }

    fn float_repeat_dim(tensor: FloatTensor<Self>, dim: usize, times: usize) -> FloatTensor<Self> {
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

        client.register(OperationRepr::BaseFloat(BaseOperationRepr::RepeatDim(desc)));

        out
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, IntElem::<Self>::dtype());

        let desc = ScalarOpRepr {
            lhs: tensor.into_tensor_ir(),
            rhs: dim,
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::ArgMin(desc),
        ));

        out
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Max(desc),
        ));

        out
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOpRepr {
            lhs: tensor.into_tensor_ir(),
            rhs: dim,
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::MaxDim(desc),
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

        let desc = ReduceDimWithIndicesOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            out: out.to_tensor_ir_out(),
            out_indices: out_indices.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::MaxDimWithIndices(desc),
        ));

        (out, out_indices)
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Min(desc),
        ));

        out
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOpRepr {
            lhs: tensor.into_tensor_ir(),
            rhs: dim,
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::MinDim(desc),
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

        let desc = ReduceDimWithIndicesOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            out: out.to_tensor_ir_out(),
            out_indices: out_indices.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::MinDimWithIndices(desc),
        ));

        (out, out_indices)
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericFloat(
            dtype,
            NumericOperationRepr::Powf(desc),
        ));

        out
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        // Change the shape of the tensor to match the new axes
        let shape = axes.iter().map(|x| tensor.shape[*x]).collect();
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = PermuteOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationRepr::BaseFloat(BaseOperationRepr::Permute(desc)));

        out
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let shape: Vec<_> = shape.into();
        let out = client.register_empty_tensor(shape.clone(), tensor.dtype);

        let desc = ExpandOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
            shape,
        };

        client.register(OperationRepr::BaseFloat(BaseOperationRepr::Expand(desc)));

        out
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = FlipOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationRepr::BaseFloat(BaseOperationRepr::Flip(desc)));

        out
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: burn_tensor::FloatDType) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_float_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseFloat(BaseOperationRepr::Cast(desc)));

        out
    }
}
