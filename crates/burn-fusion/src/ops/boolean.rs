use crate::{
    client::FusionClient,
    get_client,
    ops::binary::binary_ops_shape,
    stream::{
        BaseOperationDescription, BinaryOperationDescription, BoolOperationDescription,
        CatOperationDescription, ExpandOperationDescription, FlipOperationDescription, Operation,
        OperationDescription, PermuteOperationDescription, ReshapeDescription,
        SliceAssignOperationDescription, SliceOperationDescription, StreamId, SwapDimsDescription,
        UnaryOperationDescription,
    },
    Fusion, FusionBackend,
};
use burn_tensor::{
    ops::{BoolTensor, BoolTensorOps},
    Device, Shape,
};

impl<B: FusionBackend> BoolTensorOps<Self> for Fusion<B> {
    fn bool_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> BoolTensor<Self, D> {
        let client = get_client::<B>(&device.clone().into());
        let tensor = B::bool_empty(shape.clone(), device);

        client.register_tensor(
            B::bool_tensor_handle(tensor),
            shape.dims.into(),
            StreamId::current(),
        )
    }

    fn bool_shape<const D: usize>(tensor: &BoolTensor<Self, D>) -> Shape<D> {
        tensor.shape()
    }

    fn bool_into_data<const D: usize>(
        tensor: BoolTensor<Self, D>,
    ) -> burn_tensor::Reader<burn_tensor::Data<bool, D>> {
        tensor.bool_into_data()
    }

    fn bool_from_data<const D: usize>(
        data: burn_tensor::Data<bool, D>,
        device: &Device<Self>,
    ) -> BoolTensor<Self, D> {
        let client = get_client::<B>(&device.clone().into());
        let tensor = B::bool_from_data(data, device);
        let shape = B::bool_shape(&tensor);

        client.register_tensor(
            B::bool_tensor_handle(tensor),
            shape.dims.into(),
            StreamId::current(),
        )
    }

    fn bool_into_int<const D: usize>(
        tensor: BoolTensor<Self, D>,
    ) -> burn_tensor::ops::IntTensor<Self, D> {
        #[derive(new)]
        struct IntoIntOps<const D: usize> {
            desc: UnaryOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for IntoIntOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D>(&self.desc.input);
                let output = B::bool_into_int(input);
                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::Bool(BoolOperationDescription::IntoInt(desc.clone())),
            IntoIntOps::<D>::new(desc),
        );

        out
    }

    fn bool_into_float<const D: usize>(
        tensor: BoolTensor<Self, D>,
    ) -> burn_tensor::ops::FloatTensor<Self, D> {
        #[derive(new)]
        struct IntoFloatOps<const D: usize> {
            desc: UnaryOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for IntoFloatOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D>(&self.desc.input);
                let output = B::bool_into_float(input);
                handles.register_float_tensor(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Bool(BoolOperationDescription::IntoFloat(desc.clone())),
            IntoFloatOps::<D>::new(desc),
        );

        out
    }

    fn bool_device<const D: usize>(tensor: &BoolTensor<Self, D>) -> Device<Self> {
        tensor.client.device().clone().into()
    }

    fn bool_to_device<const D: usize>(
        tensor: BoolTensor<Self, D>,
        device: &Device<Self>,
    ) -> BoolTensor<Self, D> {
        let device_original: &B::FusionDevice = tensor.client.device();
        let device_target: B::FusionDevice = device.clone().into();

        if device_original == &device_target {
            return tensor;
        }

        let id = tensor.stream;
        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.client.clone();

        client_original.clone().change_client_bool::<D>(
            tensor.into_description(),
            client_target,
            id,
        )
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<Self, D2> {
        #[derive(new)]
        struct ReshapeDimsOps<const D1: usize, const D2: usize> {
            desc: ReshapeDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B> for ReshapeDimsOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D1>(&self.desc.input);
                let output = B::bool_reshape::<D1, D2>(input, Shape::from(&self.desc.out.shape));
                handles.register_bool_tensor(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let shape: Vec<usize> = shape.dims.into();
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ReshapeDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseBool(BaseOperationDescription::Reshape(desc.clone())),
            ReshapeDimsOps::<D1, D2>::new(desc),
        );

        out
    }

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> BoolTensor<Self, D1> {
        #[derive(new)]
        struct SliceOps<const D1: usize, const D2: usize> {
            desc: SliceOperationDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B> for SliceOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_bool_tensor::<D1>(&self.desc.tensor);

                let output =
                    B::bool_slice::<D1, D2>(tensor, self.desc.ranges.clone().try_into().unwrap());

                handles.register_bool_tensor(&self.desc.out.id, output);
            }
        }

        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..D1 {
            shape.push(tensor.shape[i]);
        }

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = SliceOperationDescription {
            tensor: tensor.into_description(),
            ranges: ranges.into(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseBool(BaseOperationDescription::Slice(desc.clone())),
            SliceOps::<D1, D2>::new(desc),
        );

        out
    }

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: BoolTensor<Self, D1>,
    ) -> BoolTensor<Self, D1> {
        #[derive(new)]
        struct SliceAssignOps<const D1: usize, const D2: usize> {
            desc: SliceAssignOperationDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B> for SliceAssignOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_bool_tensor::<D1>(&self.desc.tensor);
                let value = handles.get_bool_tensor::<D1>(&self.desc.value);

                let output = B::bool_slice_assign::<D1, D2>(
                    tensor,
                    self.desc.ranges.clone().try_into().unwrap(),
                    value,
                );

                handles.register_bool_tensor(&self.desc.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let stream_1 = tensor.stream;
        let stream_2 = value.stream;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = SliceAssignOperationDescription {
            tensor: tensor.into_description(),
            ranges: ranges.into(),
            value: value.into_description(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::BaseBool(BaseOperationDescription::SliceAssign(desc.clone())),
            SliceAssignOps::<D1, D2>::new(desc),
        );

        out
    }

    fn bool_cat<const D: usize>(
        tensors: Vec<BoolTensor<Self, D>>,
        dim: usize,
    ) -> BoolTensor<Self, D> {
        #[derive(new)]
        struct CatOps<const D: usize> {
            desc: CatOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for CatOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensors = self
                    .desc
                    .tensors
                    .iter()
                    .map(|tensor| handles.get_bool_tensor(tensor))
                    .collect();

                let output = B::bool_cat::<D>(tensors, self.desc.dim);

                handles.register_bool_tensor(&self.desc.out.id, output);
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

        let out = client.tensor_uninitialized(shape);

        let desc = CatOperationDescription {
            tensors: tensors.into_iter().map(|t| t.into_description()).collect(),
            dim,
            out: out.to_description_out(),
        };
        client.register(
            streams,
            OperationDescription::BaseBool(BaseOperationDescription::Cat(desc.clone())),
            CatOps::<D>::new(desc),
        );

        out
    }

    fn bool_equal<const D: usize>(
        lhs: BoolTensor<Self, D>,
        rhs: BoolTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        #[derive(new)]
        struct EqualOps<const D: usize> {
            desc: BinaryOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for EqualOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let lhs = handles.get_bool_tensor::<D>(&self.desc.lhs);
                let rhs = handles.get_bool_tensor(&self.desc.rhs);
                let output = B::bool_equal(lhs, rhs);
                handles.register_bool_tensor(&self.desc.out.id, output);
            }
        }

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::BaseBool(BaseOperationDescription::Equal(desc.clone())),
            EqualOps::<D>::new(desc),
        );

        out
    }

    fn bool_not<const D: usize>(tensor: BoolTensor<Self, D>) -> BoolTensor<Self, D> {
        #[derive(new)]
        struct NotOps<const D: usize> {
            desc: UnaryOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for NotOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D>(&self.desc.input);
                let output = B::bool_not(input);
                handles.register_bool_tensor(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::Bool(crate::stream::BoolOperationDescription::Not(desc.clone())),
            NotOps::<D>::new(desc),
        );

        out
    }

    fn bool_swap_dims<const D: usize>(
        tensor: BoolTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> BoolTensor<Self, D> {
        #[derive(new)]
        struct SwapDimsOps<const D: usize> {
            desc: SwapDimsDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for SwapDimsOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D>(&self.desc.input);
                let output = B::bool_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_bool_tensor(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];

        let out = tensor.client.tensor_uninitialized(shape);

        let desc = SwapDimsDescription {
            input: tensor.into_description(),
            dim1,
            dim2,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseBool(BaseOperationDescription::SwapDims(desc.clone())),
            SwapDimsOps::<D>::new(desc),
        );

        out
    }

    fn bool_permute<const D: usize>(
        tensor: BoolTensor<Self, D>,
        axes: [usize; D],
    ) -> BoolTensor<Self, D> {
        #[derive(new)]
        struct PermuteDimsOps<const D: usize> {
            desc: PermuteOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for PermuteDimsOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D>(&self.desc.input);
                let axes: [usize; D] = self.desc.axes.try_into().unwrap();
                let output = B::bool_permute(input, axes);
                handles.register_bool_tensor(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        // Change the shape of the tensor to match the new axes
        let shape = axes.into_iter().map(|x| tensor.shape[x]).collect();

        let out = tensor.client.tensor_uninitialized(shape);

        let desc = PermuteOperationDescription {
            input: tensor.into_description(),
            axes: axes.to_vec(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseInt(BaseOperationDescription::Permute(desc.clone())),
            PermuteDimsOps::<D>::new(desc),
        );

        out
    }

    fn bool_expand<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<Self, D2> {
        #[derive(new)]
        struct ExpandOps<const D: usize, const D2: usize> {
            desc: ExpandOperationDescription,
        }

        impl<const D: usize, const D2: usize, B: FusionBackend> Operation<B> for ExpandOps<D, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D>(&self.desc.input);
                let shape: [usize; D2] = self.desc.shape.try_into().unwrap();
                let output = B::bool_expand(input, shape.into());

                handles.register_bool_tensor(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        let out = tensor.client.tensor_uninitialized(shape.dims.into());

        let desc = ExpandOperationDescription {
            input: tensor.into_description(),
            shape: shape.dims.into(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseBool(BaseOperationDescription::Expand(desc.clone())),
            ExpandOps::<D1, D2>::new(desc),
        );

        out
    }

    fn bool_flip<const D: usize>(
        tensor: BoolTensor<Self, D>,
        axes: &[usize],
    ) -> BoolTensor<Self, D> {
        #[derive(new)]
        struct FlipOps<const D: usize> {
            desc: FlipOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for FlipOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D>(&self.desc.input);
                let output = B::bool_flip(input, self.desc.axes.as_slice());
                handles.register_bool_tensor(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = FlipOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
            axes: axes.to_vec(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseBool(BaseOperationDescription::Flip(desc.clone())),
            FlipOps::<D>::new(desc),
        );

        out
    }
}
