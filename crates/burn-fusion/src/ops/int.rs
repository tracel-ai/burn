use crate::{
    binary_int_cmp_ops, binary_int_ops,
    client::FusionClient,
    get_client,
    ops::binary::binary_ops_shape,
    scalar_int_cmp_ops, scalar_int_ops,
    stream::{execution::Operation, StreamId},
    unary_int_ops, Fusion, FusionBackend,
};
use burn_tensor::{
    ops::{BoolTensor, FloatTensor, IntElem, IntTensor, IntTensorOps},
    repr::{self, *},
    DType, Data, Device, Distribution, Element, ElementConversion, Reader, Shape,
};
use core::ops::Range;
use std::marker::PhantomData;

impl<B: FusionBackend> IntTensorOps<Self> for Fusion<B> {
    fn int_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        let client = get_client::<B>(&device.clone());
        let tensor = B::int_empty(shape.clone(), device);
        let stream = StreamId::current();

        client.register_tensor(
            B::int_tensor_handle(tensor),
            shape.dims.into(),
            stream,
            B::IntElem::dtype(),
        )
    }

    fn int_shape<const D: usize>(tensor: &IntTensor<Self, D>) -> Shape<D> {
        tensor.shape()
    }

    fn int_into_data<const D: usize>(tensor: IntTensor<Self, D>) -> Reader<Data<IntElem<Self>, D>> {
        tensor.int_into_data::<B, D>()
    }

    fn int_from_data<const D: usize>(
        data: Data<IntElem<Self>, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        let client = get_client::<B>(&device.clone());
        let tensor = B::int_from_data(data, device);
        let shape = B::int_shape(&tensor);
        let stream = StreamId::current();

        client.register_tensor(
            B::int_tensor_handle(tensor),
            shape.dims.into(),
            stream,
            B::IntElem::dtype(),
        )
    }

    fn int_device<const D: usize>(tensor: &IntTensor<Self, D>) -> Device<Self> {
        tensor.client.device().clone()
    }

    fn int_to_device<const D: usize>(
        tensor: IntTensor<Self, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        let device_original: &B::Device = tensor.client.device();
        let device_target: B::Device = device.clone();

        if device_original == &device_target {
            return tensor;
        }

        let id = tensor.stream;
        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.client.clone();

        client_original.clone().change_client_int::<B, D>(
            tensor.into_description(),
            client_target,
            id,
        )
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<Self, D2> {
        #[derive(new)]
        struct ReshapeDimsOps<B: FusionBackend, const D1: usize, const D2: usize> {
            desc: ReshapeDescription,
            _b: PhantomData<B>,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B::FusionRuntime>
            for ReshapeDimsOps<B, D1, D2>
        {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B, D1>(&self.desc.input);
                let output = B::int_reshape::<D1, D2>(input, Shape::from(&self.desc.out.shape));
                handles.register_int_tensor::<B, D2>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let shape: Vec<usize> = shape.dims.into();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ReshapeDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseInt(BaseOperationDescription::Reshape(desc.clone())),
            ReshapeDimsOps::<B, D1, D2>::new(desc),
        );

        out
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> IntTensor<Self, D1> {
        #[derive(new)]
        struct SliceOps<B: FusionBackend, const D1: usize, const D2: usize> {
            desc: SliceOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B::FusionRuntime>
            for SliceOps<B, D1, D2>
        {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B, D1>(&self.desc.tensor);

                let output =
                    B::int_slice::<D1, D2>(tensor, self.desc.ranges.clone().try_into().unwrap());

                handles.register_int_tensor::<B, D1>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..D1 {
            shape.push(tensor.shape[i]);
        }

        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = SliceOperationDescription {
            tensor: tensor.into_description(),
            ranges: ranges.into(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseInt(BaseOperationDescription::Slice(desc.clone())),
            SliceOps::<B, D1, D2>::new(desc),
        );

        out
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: IntTensor<Self, D1>,
    ) -> IntTensor<Self, D1> {
        #[derive(new)]
        struct SliceAssignOps<B: FusionBackend, const D1: usize, const D2: usize> {
            desc: SliceAssignOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B::FusionRuntime>
            for SliceAssignOps<B, D1, D2>
        {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B, D1>(&self.desc.tensor);
                let value = handles.get_int_tensor::<B, D1>(&self.desc.value);

                let output = B::int_slice_assign::<D1, D2>(
                    tensor,
                    self.desc.ranges.clone().try_into().unwrap(),
                    value,
                );

                handles.register_int_tensor::<B, D1>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = SliceAssignOperationDescription {
            tensor: tensor.into_description(),
            ranges: ranges.into(),
            value: value.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::BaseInt(BaseOperationDescription::SliceAssign(desc.clone())),
            SliceAssignOps::<B, D1, D2>::new(desc),
        );

        out
    }

    fn int_mask_where<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct MaskWhereOps<B: FusionBackend, const D: usize> {
            desc: MaskWhereOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for MaskWhereOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B, D>(&self.desc.tensor);
                let value = handles.get_int_tensor::<B, D>(&self.desc.value);
                let mask = handles.get_bool_tensor::<B, D>(&self.desc.mask);

                let output = B::int_mask_where(tensor, mask, value);

                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = mask.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = MaskWhereOperationDescription {
            tensor: tensor.into_description(),
            value: value.into_description(),
            mask: mask.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2, stream_3],
            OperationDescription::NumericInt(NumericOperationDescription::MaskWhere(desc.clone())),
            MaskWhereOps::<B, D>::new(desc),
        );

        out
    }

    fn int_mask_fill<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct MaskFillOps<B: FusionBackend, const D: usize> {
            desc: MaskFillOperationDescription<i32>,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for MaskFillOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B, D>(&self.desc.tensor);
                let mask = handles.get_bool_tensor::<B, D>(&self.desc.mask);

                let output = B::int_mask_fill(tensor, mask, self.desc.value.elem());

                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = mask.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = MaskFillOperationDescription {
            tensor: tensor.into_description(),
            value: value.elem(),
            mask: mask.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::MaskFill(desc.clone())),
            MaskFillOps::<B, D>::new(desc),
        );

        out
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct GatherOps<B: FusionBackend, const D: usize> {
            desc: GatherOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for GatherOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B, D>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B, D>(&self.desc.indices);

                let output = B::int_gather(self.desc.dim, tensor, indices);
                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let shape: Vec<usize> = indices.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = GatherOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::Gather(desc.clone())),
            GatherOps::<B, D>::new(desc),
        );

        out
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct ScatterOps<B: FusionBackend, const D: usize> {
            desc: ScatterOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for ScatterOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B, D>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B, D>(&self.desc.indices);
                let value = handles.get_int_tensor::<B, D>(&self.desc.value);

                let output = B::int_scatter(self.desc.dim, tensor, indices, value);

                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = ScatterOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            value: value.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2, stream_3],
            OperationDescription::NumericInt(NumericOperationDescription::Scatter(desc.clone())),
            ScatterOps::<B, D>::new(desc),
        );

        out
    }

    fn int_select<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct SelectOps<B: FusionBackend, const D: usize> {
            desc: SelectOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for SelectOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B, D>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B, 1>(&self.desc.indices);

                let output = B::int_select(tensor, self.desc.dim, indices);

                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let mut shape: Vec<usize> = tensor.shape.clone();
        shape[dim] = indices.shape[0];
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = SelectOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::Select(desc.clone())),
            SelectOps::<B, D>::new(desc),
        );

        out
    }

    fn int_select_assign<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct SelectAssignOps<B: FusionBackend, const D: usize> {
            desc: SelectAssignOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for SelectAssignOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B, D>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B, 1>(&self.desc.indices);
                let value = handles.get_int_tensor::<B, D>(&self.desc.value);

                let output = B::int_select_assign(tensor, self.desc.dim, indices, value);

                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = SelectAssignOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            value: value.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2, stream_3],
            OperationDescription::NumericInt(NumericOperationDescription::SelectAssign(
                desc.clone(),
            )),
            SelectAssignOps::<B, D>::new(desc),
        );

        out
    }

    fn int_cat<const D: usize>(tensors: Vec<IntTensor<Self, D>>, dim: usize) -> IntTensor<Self, D> {
        #[derive(new)]
        struct CatOps<B: FusionBackend, const D: usize> {
            desc: CatOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for CatOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensors = self
                    .desc
                    .tensors
                    .iter()
                    .map(|tensor| handles.get_int_tensor::<B, D>(tensor))
                    .collect();

                let output = B::int_cat::<D>(tensors, self.desc.dim);

                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let tensor_first = tensors.first().unwrap();
        let client = tensor_first.client.clone();

        // Calculate the output shape
        let streams = tensors.iter().map(|tensor| tensor.stream).collect();
        let mut shape: Vec<usize> = tensor_first.shape.clone();
        shape[dim] = 0;
        for tensor in tensors.iter() {
            shape[dim] += tensor.shape[dim];
        }

        let out = client.tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = CatOperationDescription {
            tensors: tensors.into_iter().map(|t| t.into_description()).collect(),
            dim,
            out: out.to_description_out(),
        };
        client.register(
            streams,
            OperationDescription::BaseInt(BaseOperationDescription::Cat(desc.clone())),
            CatOps::<B, D>::new(desc),
        );

        out
    }

    fn int_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_int_cmp_ops!(EqualOps, B::int_equal);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::BaseInt(BaseOperationDescription::Equal(desc.clone())),
            EqualOps::<B, D>::new(desc),
        );

        out
    }

    fn int_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(EqualElemOps, B::int_equal_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::EqualElem(desc.clone())),
            EqualElemOps::<B, D>::new(desc),
        );

        out
    }

    fn int_greater<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_int_cmp_ops!(GreaterOps, B::int_greater);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::Greater(desc.clone())),
            GreaterOps::<B, D>::new(desc),
        );

        out
    }

    fn int_greater_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(GreaterElemOps, B::int_greater_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::GreaterElem(
                desc.clone(),
            )),
            GreaterElemOps::<B, D>::new(desc),
        );

        out
    }

    fn int_greater_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_int_cmp_ops!(GreaterEqualOps, B::int_greater_equal);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::GreaterEqual(
                desc.clone(),
            )),
            GreaterEqualOps::<B, D>::new(desc),
        );

        out
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(GreaterEqualElemOps, B::int_greater_equal_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::GreaterEqualElem(
                desc.clone(),
            )),
            GreaterEqualElemOps::<B, D>::new(desc),
        );

        out
    }

    fn int_lower<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_int_cmp_ops!(LowerOps, B::int_lower);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::Lower(desc.clone())),
            LowerOps::<B, D>::new(desc),
        );

        out
    }

    fn int_lower_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(LowerElemOps, B::int_lower_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::LowerElem(desc.clone())),
            LowerElemOps::<B, D>::new(desc),
        );

        out
    }

    fn int_lower_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_int_cmp_ops!(LowerEqualOps, B::int_lower_equal);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::LowerEqual(desc.clone())),
            LowerEqualOps::<B, D>::new(desc),
        );

        out
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(LowerEqualElemOps, B::int_lower_equal_elem);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), DType::Bool);

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::LowerEqualElem(
                desc.clone(),
            )),
            LowerEqualElemOps::<B, D>::new(desc),
        );

        out
    }

    fn int_add<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        binary_int_ops!(AddOps, B::int_add);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            repr::OperationDescription::NumericInt(NumericOperationDescription::Add(desc.clone())),
            AddOps::<B, D>::new(desc),
        );

        out
    }

    fn int_add_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(AddOps, B::int_add_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            repr::OperationDescription::NumericInt(NumericOperationDescription::AddScalar(
                desc.clone(),
            )),
            AddOps::<B, D>::new(desc),
        );

        out
    }

    fn int_sub<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        binary_int_ops!(SubOps, B::int_sub);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            repr::OperationDescription::NumericInt(NumericOperationDescription::Sub(desc.clone())),
            SubOps::<B, D>::new(desc),
        );

        out
    }

    fn int_sub_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(SubOps, B::int_sub_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            repr::OperationDescription::NumericInt(NumericOperationDescription::SubScalar(
                desc.clone(),
            )),
            SubOps::<B, D>::new(desc),
        );

        out
    }

    fn int_mul<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        binary_int_ops!(MulOps, B::int_mul);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            repr::OperationDescription::NumericInt(NumericOperationDescription::Mul(desc.clone())),
            MulOps::<B, D>::new(desc),
        );

        out
    }

    fn int_mul_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(MulOps, B::int_mul_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            repr::OperationDescription::NumericInt(NumericOperationDescription::MulScalar(
                desc.clone(),
            )),
            MulOps::<B, D>::new(desc),
        );

        out
    }

    fn int_div<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        binary_int_ops!(DivOps, B::int_div);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::IntElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            repr::OperationDescription::NumericInt(NumericOperationDescription::Div(desc.clone())),
            DivOps::<B, D>::new(desc),
        );

        out
    }

    fn int_div_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(DivOps, B::int_div_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            repr::OperationDescription::NumericInt(NumericOperationDescription::DivScalar(
                desc.clone(),
            )),
            DivOps::<B, D>::new(desc),
        );

        out
    }

    fn int_remainder_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(ModOps, B::int_remainder_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            repr::OperationDescription::NumericInt(NumericOperationDescription::RemScalar(
                desc.clone(),
            )),
            ModOps::<B, D>::new(desc),
        );

        out
    }

    fn int_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        #[derive(new)]
        struct ZerosOps<B: FusionBackend, const D: usize> {
            desc: TensorDescription,
            device: Device<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for ZerosOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.desc.shape.clone());
                let output = B::int_zeros::<D>(shape, &self.device);
                handles.register_int_tensor::<B, D>(&self.desc.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = out.to_description_out();
        client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Zeros(desc.clone())),
            ZerosOps::<B, D>::new(desc, device.clone()),
        );

        out
    }

    fn int_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        #[derive(new)]
        struct OnesOps<B: FusionBackend, const D: usize> {
            desc: TensorDescription,
            device: Device<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for OnesOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.desc.shape.clone());
                let output = B::int_ones::<D>(shape, &self.device);
                handles.register_int_tensor::<B, D>(&self.desc.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = out.to_description_out();
        client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Ones(desc.clone())),
            OnesOps::<B, D>::new(desc, device.clone()),
        );

        out
    }

    fn int_sum<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(SumOps, B::int_sum, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::IntElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Sum(desc.clone())),
            SumOps::<B, D>::new(desc),
        );

        out
    }

    fn int_sum_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(SumDimOps, B::int_sum_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::SumDim(desc.clone())),
            SumDimOps::<B, D>::new(desc),
        );

        out
    }

    fn int_prod<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(ProdOps, B::int_prod, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::IntElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Prod(desc.clone())),
            ProdOps::<B, D>::new(desc),
        );

        out
    }

    fn int_prod_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(ProdDimOps, B::int_prod_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::ProdDim(desc.clone())),
            ProdDimOps::<B, D>::new(desc),
        );

        out
    }

    fn int_mean<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(MeanOps, B::int_mean, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::IntElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Mean(desc.clone())),
            MeanOps::<B, D>::new(desc),
        );

        out
    }

    fn int_mean_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(MeanDimOps, B::int_mean_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::MeanDim(desc.clone())),
            MeanDimOps::<B, D>::new(desc),
        );

        out
    }

    fn int_argmax<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(ArgMaxOps, B::int_argmax, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::ArgMax(desc.clone())),
            ArgMaxOps::<B, D>::new(desc),
        );

        out
    }

    fn int_argmin<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(ArgMinOps, B::int_argmin, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::ArgMin(desc.clone())),
            ArgMinOps::<B, D>::new(desc),
        );

        out
    }

    fn int_clamp<const D: usize>(
        tensor: IntTensor<Self, D>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct ClampOps<B: FusionBackend, const D: usize> {
            desc: ClampOperationDescription<i32>,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for ClampOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B, D>(&self.desc.tensor);
                let output = B::int_clamp(input, self.desc.min.elem(), self.desc.max.elem());

                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::IntElem::dtype());
        let desc = ClampOperationDescription {
            tensor: tensor.into_description(),
            min: min.elem(),
            max: max.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Clamp(desc.clone())),
            ClampOps::<B, D>::new(desc),
        );

        out
    }

    fn int_abs<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, D> {
        unary_int_ops!(AbsOps, B::int_abs);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::IntElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Abs(desc.clone())),
            AbsOps::<B, D>::new(desc),
        );

        out
    }

    fn int_into_float<const D: usize>(tensor: IntTensor<Self, D>) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct IntoFloatOps<B: FusionBackend, const D: usize> {
            desc: UnaryOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for IntoFloatOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B, D>(&self.desc.input);
                let output = B::int_into_float(input);
                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::FloatElem::dtype());
        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Int(repr::IntOperationDescription::IntoFloat(desc.clone())),
            IntoFloatOps::<B, D>::new(desc),
        );

        out
    }

    fn int_swap_dims<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct SwapDimsOps<B: FusionBackend, const D: usize> {
            desc: SwapDimsDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for SwapDimsOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B, D>(&self.desc.input);
                let output = B::int_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];

        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = SwapDimsDescription {
            input: tensor.into_description(),
            dim1,
            dim2,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseInt(BaseOperationDescription::SwapDims(desc.clone())),
            SwapDimsOps::<B, D>::new(desc),
        );

        out
    }

    fn int_max<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(MaxOps, B::int_max, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::IntElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Max(desc.clone())),
            MaxOps::<B, D>::new(desc),
        );

        out
    }

    fn int_max_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(MaxDimOps, B::int_max_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::MaxDim(desc.clone())),
            MaxDimOps::<B, D>::new(desc),
        );

        out
    }

    fn int_max_dim_with_indices<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
    ) -> (IntTensor<Self, D>, IntTensor<Self, D>) {
        #[derive(new)]
        struct MaxDimWithIndicesOps<B: FusionBackend, const D: usize> {
            desc: ReduceDimWithIndicesDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for MaxDimWithIndicesOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B, D>(&self.desc.tensor);
                let (output, indices) = B::int_max_dim_with_indices(tensor, self.desc.dim);

                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
                handles.register_int_tensor::<B, D>(&self.desc.out_indices.id, indices);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone(), B::IntElem::dtype());
        let out_indices = client.tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = ReduceDimWithIndicesDescription {
            tensor: tensor.into_description(),
            dim,
            out: out.to_description_out(),
            out_indices: out_indices.to_description_out(),
        };
        client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::MaxDimWithIndices(
                desc.clone(),
            )),
            MaxDimWithIndicesOps::<B, D>::new(desc),
        );

        (out, out_indices)
    }

    fn int_min<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(MinOps, B::int_min, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::IntElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Min(desc.clone())),
            MinOps::<B, D>::new(desc),
        );

        out
    }

    fn int_min_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(MinDimOps, B::int_min_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::MinDim(desc.clone())),
            MinDimOps::<B, D>::new(desc),
        );

        out
    }

    fn int_min_dim_with_indices<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
    ) -> (IntTensor<Self, D>, IntTensor<Self, D>) {
        #[derive(new)]
        struct MinDimWithIndicesOps<B: FusionBackend, const D: usize> {
            desc: ReduceDimWithIndicesDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for MinDimWithIndicesOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B, D>(&self.desc.tensor);
                let (output, indices) = B::int_min_dim_with_indices(tensor, self.desc.dim);

                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
                handles.register_int_tensor::<B, D>(&self.desc.out_indices.id, indices);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone(), B::IntElem::dtype());
        let out_indices = client.tensor_uninitialized(shape, B::IntElem::dtype());
        let desc = ReduceDimWithIndicesDescription {
            tensor: tensor.into_description(),
            dim,
            out: out.to_description_out(),
            out_indices: out_indices.to_description_out(),
        };
        client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::MinDimWithIndices(
                desc.clone(),
            )),
            MinDimWithIndicesOps::<B, D>::new(desc),
        );

        (out, out_indices)
    }

    fn int_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct IntRandomOps<B: FusionBackend, const D: usize> {
            desc: RandomOperationDescription,
            device: Device<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for IntRandomOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.desc.out.shape.clone());
                let output: B::IntTensorPrimitive<D> =
                    B::int_random(shape, self.desc.distribution, &self.device);
                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = RandomOperationDescription {
            out: out.to_description_out(),
            distribution,
        };
        client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::IntRandom(desc.clone())),
            IntRandomOps::<B, D>::new(desc, device.clone()),
        );

        out
    }

    fn int_permute<const D: usize>(
        tensor: IntTensor<Self, D>,
        axes: [usize; D],
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct PermuteDimsOps<B: FusionBackend, const D: usize> {
            desc: PermuteOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for PermuteDimsOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B, D>(&self.desc.input);
                let axes: [usize; D] = self.desc.axes.try_into().unwrap();
                let output = B::int_permute(input, axes);
                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        // Change the shape of the tensor to match the new axes
        let shape = axes.into_iter().map(|x| tensor.shape[x]).collect();

        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = PermuteOperationDescription {
            input: tensor.into_description(),
            axes: axes.to_vec(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseInt(BaseOperationDescription::Permute(desc.clone())),
            PermuteDimsOps::<B, D>::new(desc),
        );

        out
    }

    fn int_expand<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<Self, D2> {
        #[derive(new)]
        struct ExpandOps<B: FusionBackend, const D: usize, const D2: usize> {
            desc: ExpandOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, const D2: usize, B: FusionBackend> Operation<B::FusionRuntime>
            for ExpandOps<B, D, D2>
        {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_bool_tensor::<B, D>(&self.desc.input);
                let shape: [usize; D2] = self.desc.shape.try_into().unwrap();
                let output = B::bool_expand(input, shape.into());
                handles.register_bool_tensor::<B, D2>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        let out = tensor
            .client
            .tensor_uninitialized(shape.dims.into(), B::IntElem::dtype());

        let desc = ExpandOperationDescription {
            input: tensor.into_description(),
            shape: shape.dims.into(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseInt(BaseOperationDescription::Expand(desc.clone())),
            ExpandOps::<B, D1, D2>::new(desc),
        );

        out
    }

    fn int_flip<const D: usize>(tensor: IntTensor<Self, D>, axes: &[usize]) -> IntTensor<Self, D> {
        #[derive(new)]
        struct FlipDimsOps<B: FusionBackend, const D: usize> {
            desc: FlipOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for FlipDimsOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_int_tensor::<B, D>(&self.desc.input);
                let axes = &self.desc.axes;
                let output = B::int_flip(input, axes);
                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::IntElem::dtype());

        let desc = FlipOperationDescription {
            input: tensor.into_description(),
            axes: axes.to_vec(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseInt(BaseOperationDescription::Flip(desc.clone())),
            FlipDimsOps::<B, D>::new(desc),
        );

        out
    }

    fn int_repeat<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct RepeatOps<B: FusionBackend, const D: usize> {
            desc: RepeatOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for RepeatOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_int_tensor::<B, D>(&self.desc.tensor);

                let output = B::int_repeat::<D>(tensor, self.desc.dim, self.desc.times);

                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] *= times;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = RepeatOperationDescription {
            tensor: tensor.into_description(),
            dim,
            times,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseInt(BaseOperationDescription::Repeat(desc.clone())),
            RepeatOps::<B, D>::new(desc),
        );

        out
    }
}
