use alloc::{vec, vec::Vec};
use burn_tensor::backend::Backend;
use core::ops::Range;

use burn_ir::{
    BaseOperationRepr, BinaryOpRepr, CatOpRepr, ClampOpRepr, ExpandOpRepr, FlipOpRepr,
    GatherOpRepr, InitOperationRepr, IntOperationRepr, MaskFillOpRepr, MaskWhereOpRepr,
    NumericOperationRepr, OperationRepr, PermuteOpRepr, RandomOpRepr, ReduceDimWithIndicesOpRepr,
    RepeatDimOpRepr, ScalarOpRepr, ScatterOpRepr, SelectAssignOpRepr, SelectOpRepr,
    SliceAssignOpRepr, SliceOpRepr, SwapDimsOpRepr, UnaryOpRepr,
};
use burn_tensor::ops::{
    binary_ops_shape, BoolTensor, FloatElem, FloatTensor, IntElem, IntTensor, IntTensorOps,
};
use burn_tensor::{
    DType, Device, Distribution, Element, ElementConversion, Shape, TensorData, TensorMetadata,
};

use crate::{get_client, BackendRouter, RunnerChannel, RunnerClient};

impl<R: RunnerChannel> IntTensorOps<Self> for BackendRouter<R> {
    fn int_empty(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let out = client.register_empty_tensor(shape.into(), IntElem::<Self>::dtype());

        client.register(OperationRepr::BaseInt(BaseOperationRepr::Empty(
            out.to_tensor_ir_out(),
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
        let desc = InitOperationRepr {
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Init(desc));

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
        let out = client.register_empty_tensor(shape.into(), tensor.dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseInt(BaseOperationRepr::Reshape(desc)));

        out
    }

    fn int_slice(tensor: IntTensor<Self>, ranges: &[Range<usize>]) -> IntTensor<Self> {
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

        client.register(OperationRepr::BaseInt(BaseOperationRepr::Slice(desc)));

        out
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        ranges: &[Range<usize>],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = SliceAssignOpRepr {
            tensor: tensor.into_tensor_ir(),
            ranges: ranges.to_vec(),
            value: value.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseInt(BaseOperationRepr::SliceAssign(desc)));

        out
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: BoolTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
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

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::MaskWhere(desc),
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

        let desc = MaskFillOpRepr {
            tensor: tensor.into_tensor_ir(),
            mask: mask.into_tensor_ir(),
            value: value.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::MaskFill(desc),
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

        let desc = GatherOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            indices: indices.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Gather(desc),
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

        let desc = ScatterOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            indices: indices.into_tensor_ir(),
            value: value.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Scatter(desc),
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

        let desc = SelectOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            indices: indices.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Select(desc),
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

        let desc = SelectAssignOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            indices: indices.into_tensor_ir(),
            value: value.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::SelectAssign(desc),
        ));

        out
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
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

        client.register(OperationRepr::BaseInt(BaseOperationRepr::Cat(desc)));

        out
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::BaseInt(BaseOperationRepr::Equal(desc)));

        out
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::EqualElem(desc),
        ));

        out
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Greater(desc),
        ));

        out
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::GreaterElem(desc),
        ));

        out
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::GreaterEqual(desc),
        ));

        out
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::GreaterEqualElem(desc),
        ));

        out
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Lower(desc),
        ));

        out
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::LowerElem(desc),
        ));

        out
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::LowerEqual(desc),
        ));

        out
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::LowerEqualElem(desc),
        ));

        out
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Add(desc),
        ));

        out
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::AddScalar(desc),
        ));

        out
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Sub(desc),
        ));

        out
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::SubScalar(desc),
        ));

        out
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Mul(desc),
        ));

        out
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::MulScalar(desc),
        ));

        out
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Div(desc),
        ));

        out
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::DivScalar(desc),
        ));

        out
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Rem(desc),
        ));

        out
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::RemScalar(desc),
        ));

        out
    }

    fn int_zeros(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = IntElem::<Self>::dtype();
        let out = client.register_empty_tensor(shape.dims.to_vec(), dtype);

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Zeros(out.to_tensor_ir_out()),
        ));

        out
    }

    fn int_ones(shape: Shape, device: &Device<Self>) -> IntTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = IntElem::<Self>::dtype();
        let out = client.register_empty_tensor(shape.into(), dtype);

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Ones(out.to_tensor_ir_out()),
        ));

        out
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Sum(desc),
        ));

        out
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
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

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::SumDim(desc),
        ));

        out
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Prod(desc),
        ));

        out
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
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

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::ProdDim(desc),
        ));

        out
    }

    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Mean(desc),
        ));

        out
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
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

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::MeanDim(desc),
        ));

        out
    }

    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
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

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::ArgMax(desc),
        ));

        out
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
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

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::ArgMin(desc),
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

        let desc = ClampOpRepr {
            tensor: tensor.into_tensor_ir(),
            min: min.elem(),
            max: max.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Clamp(desc),
        ));

        out
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Abs(desc),
        ));

        out
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), FloatElem::<Self>::dtype());

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Int(IntOperationRepr::IntoFloat(desc)));

        out
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
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

        client.register(OperationRepr::BaseInt(BaseOperationRepr::SwapDims(desc)));

        out
    }

    fn int_max(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Max(desc),
        ));

        out
    }

    fn int_max_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
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

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::MaxDim(desc),
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

        let desc = ReduceDimWithIndicesOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            out: out.to_tensor_ir_out(),
            out_indices: out_indices.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::MaxDimWithIndices(desc),
        ));

        (out, out_indices)
    }

    fn int_min(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::Min(desc),
        ));

        out
    }

    fn int_min_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
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

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::MinDim(desc),
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

        let desc = ReduceDimWithIndicesOpRepr {
            tensor: tensor.into_tensor_ir(),
            dim,
            out: out.to_tensor_ir_out(),
            out_indices: out_indices.to_tensor_ir_out(),
        };

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::MinDimWithIndices(desc),
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
        let out = client.register_empty_tensor(shape.into(), dtype);

        client.register(OperationRepr::NumericInt(
            dtype,
            NumericOperationRepr::IntRandom(RandomOpRepr {
                out: out.to_tensor_ir_out(),
                distribution,
            }),
        ));

        out
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        let client = tensor.client.clone();
        // Change the shape of the tensor to match the new axes
        let shape = axes.iter().map(|x| tensor.shape[*x]).collect();
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = PermuteOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationRepr::BaseInt(BaseOperationRepr::Permute(desc)));

        out
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let shape: Vec<_> = shape.into();
        let out = client.register_empty_tensor(shape.clone(), tensor.dtype);

        let desc = ExpandOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
            shape,
        };

        client.register(OperationRepr::BaseInt(BaseOperationRepr::Expand(desc)));

        out
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = FlipOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationRepr::BaseInt(BaseOperationRepr::Flip(desc)));

        out
    }

    fn int_repeat_dim(tensor: IntTensor<Self>, dim: usize, times: usize) -> IntTensor<Self> {
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

        client.register(OperationRepr::BaseInt(BaseOperationRepr::RepeatDim(desc)));

        out
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Int(IntOperationRepr::BitwiseAnd(desc)));

        out
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Int(IntOperationRepr::BitwiseOr(desc)));

        out
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Int(IntOperationRepr::BitwiseXor(desc)));

        out
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOpRepr {
            input: tensor.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Int(IntOperationRepr::BitwiseNot(desc)));

        out
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Int(IntOperationRepr::BitwiseAndScalar(desc)));

        out
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Int(IntOperationRepr::BitwiseOrScalar(desc)));

        out
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Int(IntOperationRepr::BitwiseXorScalar(desc)));

        out
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Int(IntOperationRepr::BitwiseLeftShift(desc)));

        out
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Int(
            IntOperationRepr::BitwiseLeftShiftScalar(desc),
        ));

        out
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.into_tensor_ir(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Int(IntOperationRepr::BitwiseRightShift(
            desc,
        )));

        out
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: IntElem<Self>) -> IntTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOpRepr {
            lhs: lhs.into_tensor_ir(),
            rhs: rhs.elem(),
            out: out.to_tensor_ir_out(),
        };

        client.register(OperationRepr::Int(
            IntOperationRepr::BitwiseRightShiftScalar(desc),
        ));

        out
    }
}
