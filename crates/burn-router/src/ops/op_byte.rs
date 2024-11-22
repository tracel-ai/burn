use alloc::{vec, vec::Vec};
use burn_tensor::{
    backend::Backend,
    ops::{ByteTensor, ByteTensorOps, IntElem},
    repr::ByteOperationDescription,
};
use core::ops::Range;

use burn_tensor::ops::{binary_ops_shape, BoolTensor, ByteElem, FloatElem, FloatTensor, IntTensor};
use burn_tensor::repr::{
    BaseOperationDescription, BinaryOperationDescription, CatOperationDescription,
    ClampOperationDescription, ExpandOperationDescription, FlipOperationDescription,
    GatherOperationDescription, MaskFillOperationDescription, MaskWhereOperationDescription,
    NumericOperationDescription, OperationDescription, PermuteOperationDescription,
    RandomOperationDescription, ReduceDimWithIndicesDescription, RepeatDimOperationDescription,
    ReshapeDescription, ScalarOperationDescription, ScatterOperationDescription,
    SelectAssignOperationDescription, SelectOperationDescription, SliceAssignOperationDescription,
    SliceOperationDescription, SwapDimsDescription, UnaryOperationDescription,
};
use burn_tensor::{
    DType, Device, Distribution, Element, ElementConversion, Shape, TensorData, TensorMetadata,
};

use crate::{get_client, BackendRouter, RunnerChannel, RunnerClient};

impl<R: RunnerChannel> ByteTensorOps<Self> for BackendRouter<R> {
    fn byte_empty(shape: Shape, device: &Device<Self>) -> ByteTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let out = client.register_empty_tensor(shape.into(), ByteElem::<Self>::dtype());

        client.register(OperationDescription::BaseByte(
            BaseOperationDescription::Empty(out.to_description_out()),
        ));

        out
    }

    async fn byte_into_data(tensor: ByteTensor<Self>) -> TensorData {
        tensor
            .into_data()
            .await
            // Since underlying backends can have different data types, we convert to the current elem
            .convert::<<Self as Backend>::ByteElem>()
    }

    fn byte_from_data(data: TensorData, device: &Device<Self>) -> ByteTensor<Self> {
        let client = get_client::<R>(device);
        client.register_tensor_data(data.convert::<<Self as Backend>::ByteElem>())
    }

    fn byte_device(tensor: &ByteTensor<Self>) -> Device<Self> {
        tensor.client.device()
    }

    fn byte_to_device(tensor: ByteTensor<Self>, device: &Device<Self>) -> ByteTensor<Self> {
        if &tensor.client.device() == device {
            return tensor;
        }
        R::change_client_backend(tensor, device)
    }

    fn byte_reshape(tensor: ByteTensor<Self>, shape: Shape) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(shape.into(), tensor.dtype);

        let desc = ReshapeDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::BaseByte(
            BaseOperationDescription::Reshape(desc),
        ));

        out
    }

    fn byte_slice(tensor: ByteTensor<Self>, ranges: &[Range<usize>]) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;

        let ndims = tensor.shape().num_dims();
        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..ndims {
            shape.push(tensor.shape[i]);
        }

        let out = client.register_empty_tensor(shape, dtype);

        let desc = SliceOperationDescription {
            tensor: tensor.into_description(),
            ranges: ranges.to_vec(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::BaseByte(
            BaseOperationDescription::Slice(desc),
        ));

        out
    }

    fn byte_slice_assign(
        tensor: ByteTensor<Self>,
        ranges: &[Range<usize>],
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = SliceAssignOperationDescription {
            tensor: tensor.into_description(),
            ranges: ranges.to_vec(),
            value: value.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::BaseByte(
            BaseOperationDescription::SliceAssign(desc),
        ));

        out
    }

    fn byte_mask_where(
        tensor: ByteTensor<Self>,
        mask: BoolTensor<Self>,
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let shape = binary_ops_shape(&tensor.shape, &mask.shape);
        let out = client.register_empty_tensor(shape, dtype);

        let desc = MaskWhereOperationDescription {
            tensor: tensor.into_description(),
            mask: mask.into_description(),
            value: value.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::MaskWhere(desc),
        ));

        out
    }

    fn byte_mask_fill(
        tensor: ByteTensor<Self>,
        mask: BoolTensor<Self>,
        value: ByteElem<Self>,
    ) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = MaskFillOperationDescription {
            tensor: tensor.into_description(),
            mask: mask.into_description(),
            value: value.elem(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::MaskFill(desc),
        ));

        out
    }

    fn byte_gather(
        dim: usize,
        tensor: ByteTensor<Self>,
        indices: IntTensor<Self>,
    ) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(indices.shape.clone(), dtype);

        let desc = GatherOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Gather(desc),
        ));

        out
    }

    fn byte_scatter(
        dim: usize,
        tensor: ByteTensor<Self>,
        indices: IntTensor<Self>,
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = ScatterOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            value: value.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Scatter(desc),
        ));

        out
    }

    fn byte_select(
        tensor: ByteTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = indices.shape[0];
        let out = client.register_empty_tensor(shape, dtype);

        let desc = SelectOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Select(desc),
        ));

        out
    }

    fn byte_select_assign(
        tensor: ByteTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: ByteTensor<Self>,
    ) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = SelectAssignOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            value: value.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::SelectAssign(desc),
        ));

        out
    }

    fn byte_cat(tensors: Vec<ByteTensor<Self>>, dim: usize) -> ByteTensor<Self> {
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

        let desc = CatOperationDescription {
            tensors: tensors.into_iter().map(|t| t.into_description()).collect(),
            dim,
            out: out.to_description_out(),
        };

        client.register(OperationDescription::BaseByte(
            BaseOperationDescription::Cat(desc),
        ));

        out
    }

    fn byte_equal(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::BaseByte(
            BaseOperationDescription::Equal(desc),
        ));

        out
    }

    fn byte_equal_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::EqualElem(desc),
        ));

        out
    }

    fn byte_greater(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Greater(desc),
        ));

        out
    }

    fn byte_greater_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::GreaterElem(desc),
        ));

        out
    }

    fn byte_greater_equal(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::GreaterEqual(desc),
        ));

        out
    }

    fn byte_greater_equal_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::GreaterEqualElem(desc),
        ));

        out
    }

    fn byte_lower(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Lower(desc),
        ));

        out
    }

    fn byte_lower_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::LowerElem(desc),
        ));

        out
    }

    fn byte_lower_equal(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out =
            client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::LowerEqual(desc),
        ));

        out
    }

    fn byte_lower_equal_elem(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> BoolTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::LowerEqualElem(desc),
        ));

        out
    }

    fn byte_add(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Add(desc),
        ));

        out
    }

    fn byte_add_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::AddScalar(desc),
        ));

        out
    }

    fn byte_sub(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Sub(desc),
        ));

        out
    }

    fn byte_sub_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::SubScalar(desc),
        ));

        out
    }

    fn byte_mul(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Mul(desc),
        ));

        out
    }

    fn byte_mul_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::MulScalar(desc),
        ));

        out
    }

    fn byte_div(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Div(desc),
        ));

        out
    }

    fn byte_div_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::DivScalar(desc),
        ));

        out
    }

    fn byte_remainder(lhs: ByteTensor<Self>, rhs: ByteTensor<Self>) -> ByteTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(binary_ops_shape(&lhs.shape, &rhs.shape), dtype);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Rem(desc),
        ));

        out
    }

    fn byte_remainder_scalar(lhs: ByteTensor<Self>, rhs: ByteElem<Self>) -> ByteTensor<Self> {
        let client = lhs.client.clone();
        let dtype = lhs.dtype;
        let out = client.register_empty_tensor(lhs.shape.clone(), dtype);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::RemScalar(desc),
        ));

        out
    }

    fn byte_zeros(shape: Shape, device: &Device<Self>) -> ByteTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = ByteElem::<Self>::dtype();
        let out = client.register_empty_tensor(shape.dims.to_vec(), dtype);

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Zeros(out.to_description_out()),
        ));

        out
    }

    fn byte_ones(shape: Shape, device: &Device<Self>) -> ByteTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = ByteElem::<Self>::dtype();
        let out = client.register_empty_tensor(shape.into(), dtype);

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Ones(out.to_description_out()),
        ));

        out
    }

    fn byte_sum(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Sum(desc),
        ));

        out
    }

    fn byte_sum_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::SumDim(desc),
        ));

        out
    }

    fn byte_prod(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Prod(desc),
        ));

        out
    }

    fn byte_prod_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::ProdDim(desc),
        ));

        out
    }

    fn byte_mean(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Mean(desc),
        ));

        out
    }

    fn byte_mean_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::MeanDim(desc),
        ));

        out
    }

    fn byte_argmax(tensor: ByteTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, IntElem::<Self>::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::ArgMax(desc),
        ));

        out
    }

    fn byte_argmin(tensor: ByteTensor<Self>, dim: usize) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, IntElem::<Self>::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::ArgMin(desc),
        ));

        out
    }

    fn byte_clamp(
        tensor: ByteTensor<Self>,
        min: ByteElem<Self>,
        max: ByteElem<Self>,
    ) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = ClampOperationDescription {
            tensor: tensor.into_description(),
            min: min.elem(),
            max: max.elem(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Clamp(desc),
        ));

        out
    }

    fn byte_abs(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(tensor.shape.clone(), dtype);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Abs(desc),
        ));

        out
    }

    fn byte_into_float(tensor: ByteTensor<Self>) -> FloatTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), FloatElem::<Self>::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::Byte(
            ByteOperationDescription::IntoFloat(desc),
        ));

        out
    }

    fn byte_into_int(tensor: ByteTensor<Self>) -> IntTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), ByteElem::<Self>::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::Byte(
            ByteOperationDescription::IntoInt(desc),
        ));

        out
    }

    fn byte_swap_dims(tensor: ByteTensor<Self>, dim1: usize, dim2: usize) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = SwapDimsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
            dim1,
            dim2,
        };

        client.register(OperationDescription::BaseByte(
            BaseOperationDescription::SwapDims(desc),
        ));

        out
    }

    fn byte_max(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Max(desc),
        ));

        out
    }

    fn byte_max_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::MaxDim(desc),
        ));

        out
    }

    fn byte_max_dim_with_indices(
        tensor: ByteTensor<Self>,
        dim: usize,
    ) -> (ByteTensor<Self>, IntTensor<Self>) {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape.clone(), dtype);
        let out_indices = client.register_empty_tensor(shape, IntElem::<Self>::dtype());

        let desc = ReduceDimWithIndicesDescription {
            tensor: tensor.into_description(),
            dim,
            out: out.to_description_out(),
            out_indices: out_indices.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::MaxDimWithIndices(desc),
        ));

        (out, out_indices)
    }

    fn byte_min(tensor: ByteTensor<Self>) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let out = client.register_empty_tensor(vec![1], dtype);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::Min(desc),
        ));

        out
    }

    fn byte_min_dim(tensor: ByteTensor<Self>, dim: usize) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape, dtype);

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::MinDim(desc),
        ));

        out
    }

    fn byte_min_dim_with_indices(
        tensor: ByteTensor<Self>,
        dim: usize,
    ) -> (ByteTensor<Self>, IntTensor<Self>) {
        let client = tensor.client.clone();
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = client.register_empty_tensor(shape.clone(), dtype);
        let out_indices = client.register_empty_tensor(shape, IntElem::<Self>::dtype());

        let desc = ReduceDimWithIndicesDescription {
            tensor: tensor.into_description(),
            dim,
            out: out.to_description_out(),
            out_indices: out_indices.to_description_out(),
        };

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::MinDimWithIndices(desc),
        ));

        (out, out_indices)
    }

    fn byte_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> ByteTensor<Self> {
        // Get the runtime client on which to register the operation for execution.
        let client = get_client::<R>(device);
        let dtype = ByteElem::<Self>::dtype();
        let out = client.register_empty_tensor(shape.into(), dtype);

        client.register(OperationDescription::NumericByte(
            dtype,
            NumericOperationDescription::IntRandom(RandomOperationDescription {
                out: out.to_description_out(),
                distribution,
            }),
        ));

        out
    }

    fn byte_permute(tensor: ByteTensor<Self>, axes: &[usize]) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        // Change the shape of the tensor to match the new axes
        let shape = axes.iter().map(|x| tensor.shape[*x]).collect();
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = PermuteOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationDescription::BaseByte(
            BaseOperationDescription::Permute(desc),
        ));

        out
    }

    fn byte_expand(tensor: ByteTensor<Self>, shape: Shape) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let shape: Vec<_> = shape.into();
        let out = client.register_empty_tensor(shape.clone(), tensor.dtype);

        let desc = ExpandOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
            shape,
        };

        client.register(OperationDescription::BaseByte(
            BaseOperationDescription::Expand(desc),
        ));

        out
    }

    fn byte_flip(tensor: ByteTensor<Self>, axes: &[usize]) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let out = client.register_empty_tensor(tensor.shape.clone(), tensor.dtype);

        let desc = FlipOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
            axes: axes.to_vec(),
        };

        client.register(OperationDescription::BaseByte(
            BaseOperationDescription::Flip(desc),
        ));

        out
    }

    fn byte_repeat_dim(tensor: ByteTensor<Self>, dim: usize, times: usize) -> ByteTensor<Self> {
        let client = tensor.client.clone();
        let mut shape = tensor.shape.clone();
        shape[dim] *= times;
        let out = client.register_empty_tensor(shape, tensor.dtype);

        let desc = RepeatDimOperationDescription {
            tensor: tensor.into_description(),
            dim,
            times,
            out: out.to_description_out(),
        };

        client.register(OperationDescription::BaseByte(
            BaseOperationDescription::RepeatDim(desc),
        ));

        out
    }
}
