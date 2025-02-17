use burn_ir::{
    BaseOperationIr, BinaryOpIr, BoolOperationIr, CatOpIr, ExpandOpIr, FlipOpIr, HandleContainer,
    InitOperationIr, OperationIr, PermuteOpIr, RepeatDimOpIr, SliceAssignOpIr, SliceOpIr,
    SwapDimsOpIr, TensorIr, UnaryOpIr,
};
use burn_tensor::{
    ops::{binary_ops_shape, BoolTensor, BoolTensorOps, FloatTensor, IntTensor},
    Device, Element, Shape, TensorData, TensorMetadata,
};
use std::marker::PhantomData;

use crate::{
    client::FusionClient,
    get_client,
    stream::{execution::Operation, StreamId},
    Fusion, FusionBackend,
};

use super::NoOp;

impl<B: FusionBackend> BoolTensorOps<Self> for Fusion<B> {
    fn bool_empty(shape: Shape, device: &Device<Self>) -> BoolTensor<Self> {
        #[derive(new)]
        struct EmptyOps<B: FusionBackend> {
            desc: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for EmptyOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let output = B::bool_empty(Shape::from(&self.desc.shape), &self.device);
                handles.register_bool_tensor::<B>(&self.desc.id, output);
            }
        }

        let stream = StreamId::current();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape.dims.clone(), B::BoolElem::dtype());

        let desc = out.to_ir_out();

        client.register(
            vec![stream],
            OperationIr::BaseBool(BaseOperationIr::Empty(desc.clone())),
            EmptyOps::<B>::new(desc, device.clone()),
        );

        out
    }

    async fn bool_into_data(tensor: BoolTensor<Self>) -> TensorData {
        tensor.bool_into_data::<B>().await
    }

    fn bool_from_data(data: burn_tensor::TensorData, device: &Device<Self>) -> BoolTensor<Self> {
        let stream = StreamId::current();
        let client = get_client::<B>(&device.clone());
        let tensor = B::bool_from_data(data, device);
        let shape = tensor.shape();

        let handle = B::bool_tensor_handle(tensor);
        let out = client.register_tensor(handle, shape.dims, stream, B::BoolElem::dtype());
        let desc = out.to_ir_out();

        client.register(
            vec![stream],
            OperationIr::Init(InitOperationIr { out: desc }),
            NoOp::<B>::new(),
        );

        out
    }

    fn bool_into_int(tensor: BoolTensor<Self>) -> IntTensor<Self> {
        #[derive(new)]
        struct IntoIntOps<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntoIntOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_into_int(input);
                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::IntElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        out.client.register(
            vec![stream],
            OperationIr::Bool(BoolOperationIr::IntoInt(desc.clone())),
            IntoIntOps::<B>::new(desc),
        );

        out
    }

    fn bool_into_float(tensor: BoolTensor<Self>) -> FloatTensor<Self> {
        #[derive(new)]
        struct IntoFloatOps<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntoFloatOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_into_float(input);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::FloatElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::Bool(BoolOperationIr::IntoFloat(desc.clone())),
            IntoFloatOps::<B>::new(desc),
        );

        out
    }

    fn bool_device(tensor: &BoolTensor<Self>) -> Device<Self> {
        tensor.client.device().clone()
    }

    fn bool_to_device(tensor: BoolTensor<Self>, device: &Device<Self>) -> BoolTensor<Self> {
        let device_original: &B::Device = tensor.client.device();
        let device_target: B::Device = device.clone();

        if device_original == &device_target {
            return tensor;
        }

        let id = tensor.stream;
        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.client.clone();

        client_original
            .clone()
            .change_client_bool::<B>(tensor.into_ir(), client_target, id)
    }

    fn bool_reshape(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        #[derive(new)]
        struct ReshapeDimsOps<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ReshapeDimsOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_reshape(input, Shape::from(&self.desc.out.shape));
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(shape.dims, B::BoolElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::BaseBool(BaseOperationIr::Reshape(desc.clone())),
            ReshapeDimsOps::<B>::new(desc),
        );

        out
    }

    fn bool_slice(tensor: BoolTensor<Self>, ranges: &[std::ops::Range<usize>]) -> BoolTensor<Self> {
        #[derive(new)]
        struct SliceOps<B: FusionBackend> {
            desc: SliceOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_bool_tensor::<B>(&self.desc.tensor);

                let output = B::bool_slice(tensor, self.desc.ranges.as_slice());

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let ndims = burn_tensor::TensorMetadata::shape(&tensor).num_dims();
        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..ndims {
            shape.push(tensor.shape[i]);
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::BoolElem::dtype());

        let desc = SliceOpIr {
            tensor: tensor.into_ir(),
            ranges: ranges.into(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::BaseBool(BaseOperationIr::Slice(desc.clone())),
            SliceOps::<B>::new(desc),
        );

        out
    }

    fn bool_slice_assign(
        tensor: BoolTensor<Self>,
        ranges: &[std::ops::Range<usize>],
        value: BoolTensor<Self>,
    ) -> BoolTensor<Self> {
        #[derive(new)]
        struct SliceAssignOps<B: FusionBackend> {
            desc: SliceAssignOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceAssignOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_bool_tensor::<B>(&self.desc.tensor);
                let value = handles.get_bool_tensor::<B>(&self.desc.value);

                let output = B::bool_slice_assign(tensor, self.desc.ranges.as_slice(), value);

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let stream_1 = tensor.stream;
        let stream_2 = value.stream;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::BoolElem::dtype());

        let desc = SliceAssignOpIr {
            tensor: tensor.into_ir(),
            ranges: ranges.into(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };

        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::BaseBool(BaseOperationIr::SliceAssign(desc.clone())),
            SliceAssignOps::<B>::new(desc),
        );

        out
    }

    fn bool_cat(tensors: Vec<BoolTensor<Self>>, dim: usize) -> BoolTensor<Self> {
        #[derive(new)]
        struct CatOps<B: FusionBackend> {
            desc: CatOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CatOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensors = self
                    .desc
                    .tensors
                    .iter()
                    .map(|tensor| handles.get_bool_tensor::<B>(tensor))
                    .collect();

                let output = B::bool_cat(tensors, self.desc.dim);

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let tensor_first = tensors.first().unwrap();
        let client = tensor_first.client.clone();

        // Calculate the output shape
        let mut shape: Vec<usize> = tensor_first.shape.clone();
        let streams = tensors.iter().map(|t| t.stream).collect::<Vec<_>>();

        shape[dim] = 0;
        for tensor in tensors.iter() {
            shape[dim] += tensor.shape[dim];
        }

        let out = client.tensor_uninitialized(shape, B::BoolElem::dtype());

        let desc = CatOpIr {
            tensors: tensors.into_iter().map(|t| t.into_ir()).collect(),
            dim,
            out: out.to_ir_out(),
        };
        client.register(
            streams,
            OperationIr::BaseBool(BaseOperationIr::Cat(desc.clone())),
            CatOps::<B>::new(desc),
        );

        out
    }

    fn bool_equal(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        #[derive(new)]
        struct EqualOps<B: FusionBackend> {
            desc: BinaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for EqualOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_bool_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_bool_tensor::<B>(&self.desc.rhs);
                let output = B::bool_equal(lhs, rhs);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::BaseBool(BaseOperationIr::Equal(desc.clone())),
            EqualOps::<B>::new(desc),
        );

        out
    }

    fn bool_not(tensor: BoolTensor<Self>) -> BoolTensor<Self> {
        #[derive(new)]
        struct NotOps<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for NotOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_not(input);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::BoolElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        out.client.register(
            vec![stream],
            OperationIr::Bool(BoolOperationIr::Not(desc.clone())),
            NotOps::<B>::new(desc),
        );

        out
    }

    fn bool_and(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        #[derive(new)]
        struct AndOps<B: FusionBackend> {
            desc: BinaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for AndOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_bool_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_bool_tensor::<B>(&self.desc.rhs);
                let output = B::bool_and(lhs, rhs);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::Bool(BoolOperationIr::And(desc.clone())),
            AndOps::<B>::new(desc),
        );

        out
    }

    fn bool_or(lhs: BoolTensor<Self>, rhs: BoolTensor<Self>) -> BoolTensor<Self> {
        #[derive(new)]
        struct OrOps<B: FusionBackend> {
            desc: BinaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for OrOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_bool_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_bool_tensor::<B>(&self.desc.rhs);
                let output = B::bool_or(lhs, rhs);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationIr::Bool(BoolOperationIr::Or(desc.clone())),
            OrOps::<B>::new(desc),
        );

        out
    }

    fn bool_swap_dims(tensor: BoolTensor<Self>, dim1: usize, dim2: usize) -> BoolTensor<Self> {
        #[derive(new)]
        struct SwapDimsOps<B: FusionBackend> {
            desc: SwapDimsOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SwapDimsOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];

        let out = tensor
            .client
            .tensor_uninitialized(shape, B::BoolElem::dtype());

        let desc = SwapDimsOpIr {
            input: tensor.into_ir(),
            dim1,
            dim2,
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::BaseBool(BaseOperationIr::SwapDims(desc.clone())),
            SwapDimsOps::<B>::new(desc),
        );

        out
    }

    fn bool_permute(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        #[derive(new)]
        struct PermuteDimsOps<B: FusionBackend> {
            desc: PermuteOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for PermuteDimsOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_permute(input, self.desc.axes.as_slice());
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        // Change the shape of the tensor to match the new axes
        let shape = axes.iter().map(|x| tensor.shape[*x]).collect();

        let out = tensor
            .client
            .tensor_uninitialized(shape, B::BoolElem::dtype());

        let desc = PermuteOpIr {
            input: tensor.into_ir(),
            axes: axes.to_vec(),
            out: out.to_ir_out(),
        };

        out.client.register(
            vec![stream],
            OperationIr::BaseInt(BaseOperationIr::Permute(desc.clone())),
            PermuteDimsOps::<B>::new(desc),
        );

        out
    }

    fn bool_expand(tensor: BoolTensor<Self>, shape: Shape) -> BoolTensor<Self> {
        #[derive(new)]
        struct ExpandOps<B: FusionBackend> {
            desc: ExpandOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ExpandOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_expand(input, self.desc.shape.into());

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        let out = tensor
            .client
            .tensor_uninitialized(shape.dims.clone(), B::BoolElem::dtype());

        let desc = ExpandOpIr {
            input: tensor.into_ir(),
            shape: shape.dims,
            out: out.to_ir_out(),
        };

        out.client.register(
            vec![stream],
            OperationIr::BaseBool(BaseOperationIr::Expand(desc.clone())),
            ExpandOps::<B>::new(desc),
        );

        out
    }

    fn bool_flip(tensor: BoolTensor<Self>, axes: &[usize]) -> BoolTensor<Self> {
        #[derive(new)]
        struct FlipOps<B: FusionBackend> {
            desc: FlipOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for FlipOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B>(&self.desc.input);
                let output = B::bool_flip(input, self.desc.axes.as_slice());
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::BoolElem::dtype());

        let desc = FlipOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            axes: axes.to_vec(),
        };

        out.client.register(
            vec![stream],
            OperationIr::BaseBool(BaseOperationIr::Flip(desc.clone())),
            FlipOps::<B>::new(desc),
        );

        out
    }

    fn bool_repeat_dim(tensor: BoolTensor<Self>, dim: usize, times: usize) -> BoolTensor<Self> {
        #[derive(new)]
        struct RepeatDimOps<B: FusionBackend> {
            desc: RepeatDimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for RepeatDimOps<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_bool_tensor::<B>(&self.desc.tensor);

                let output = B::bool_repeat_dim(tensor, self.desc.dim, self.desc.times);

                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] *= times;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::BoolElem::dtype());

        let desc = RepeatDimOpIr {
            tensor: tensor.into_ir(),
            dim,
            times,
            out: out.to_ir_out(),
        };
        out.client.register(
            vec![stream],
            OperationIr::BaseBool(BaseOperationIr::RepeatDim(desc.clone())),
            RepeatDimOps::<B>::new(desc),
        );

        out
    }
}
