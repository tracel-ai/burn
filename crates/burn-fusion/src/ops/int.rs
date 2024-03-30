use crate::{
    binary_int_cmp_ops, binary_int_ops,
    client::FusionClient,
    get_client,
    ops::binary::binary_ops_shape,
    scalar_int_cmp_ops, scalar_int_ops,
    stream::{
        self, BaseOperationDescription, BinaryOperationDescription, CatOperationDescription,
        ClampOperationDescription, ExpandOperationDescription, FlipOperationDescription,
        GatherOperationDescription, MaskFillOperationDescription, MaskWhereOperationDescription,
        NumericOperationDescription, Operation, OperationDescription, PermuteOperationDescription,
        RandomOperationDescription, ReduceDimWithIndicesDescription, ReshapeDescription,
        ScalarOperationDescription, ScatterOperationDescription, SelectAssignOperationDescription,
        SelectOperationDescription, SliceAssignOperationDescription, SliceOperationDescription,
        StreamId, SwapDimsDescription, UnaryOperationDescription,
    },
    unary_int_ops, Fusion, FusionBackend, TensorDescription,
};
use burn_tensor::{
    ops::{BoolTensor, FloatTensor, IntElem, IntTensor, IntTensorOps},
    Data, Device, Distribution, ElementConversion, Reader, Shape,
};
use core::ops::Range;

impl<B: FusionBackend> IntTensorOps<Self> for Fusion<B> {
    fn int_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        let client = get_client::<B>(&device.clone().into());
        let tensor = B::int_empty(shape.clone(), device);
        let stream = StreamId::current();

        client.register_tensor(B::int_tensor_handle(tensor), shape.dims.into(), stream)
    }

    fn int_shape<const D: usize>(tensor: &IntTensor<Self, D>) -> Shape<D> {
        tensor.shape()
    }

    fn int_into_data<const D: usize>(tensor: IntTensor<Self, D>) -> Reader<Data<IntElem<Self>, D>> {
        tensor.int_into_data()
    }

    fn int_from_data<const D: usize>(
        data: Data<IntElem<Self>, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        let client = get_client::<B>(&device.clone().into());
        let tensor = B::int_from_data(data, device);
        let shape = B::int_shape(&tensor);
        let stream = StreamId::current();

        client.register_tensor(B::int_tensor_handle(tensor), shape.dims.into(), stream)
    }

    fn int_device<const D: usize>(tensor: &IntTensor<Self, D>) -> Device<Self> {
        tensor.client.device().clone().into()
    }

    fn int_to_device<const D: usize>(
        tensor: IntTensor<Self, D>,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        let device_original: &B::FusionDevice = tensor.client.device();
        let device_target: B::FusionDevice = device.clone().into();

        if device_original == &device_target {
            return tensor;
        }

        let id = tensor.stream;
        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.client.clone();

        client_original
            .clone()
            .change_client_int::<D>(tensor.into_description(), client_target, id)
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<Self, D2> {
        #[derive(new)]
        struct ReshapeDimsOps<const D1: usize, const D2: usize> {
            desc: ReshapeDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B> for ReshapeDimsOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D1>(&self.desc.input);
                let output = B::int_reshape::<D1, D2>(input, Shape::from(&self.desc.out.shape));
                handles.register_int_tensor(&self.desc.out.id, output);
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
            OperationDescription::BaseInt(BaseOperationDescription::Reshape(desc.clone())),
            ReshapeDimsOps::<D1, D2>::new(desc),
        );

        out
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> IntTensor<Self, D1> {
        #[derive(new)]
        struct SliceOps<const D1: usize, const D2: usize> {
            desc: SliceOperationDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B> for SliceOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D1>(&self.desc.tensor);

                let output =
                    B::int_slice::<D1, D2>(tensor, self.desc.ranges.clone().try_into().unwrap());

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..D1 {
            shape.push(tensor.shape[i]);
        }

        let out = tensor.client.tensor_uninitialized(shape);

        let desc = SliceOperationDescription {
            tensor: tensor.into_description(),
            ranges: ranges.into(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseInt(BaseOperationDescription::Slice(desc.clone())),
            SliceOps::<D1, D2>::new(desc),
        );

        out
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: IntTensor<Self, D1>,
    ) -> IntTensor<Self, D1> {
        #[derive(new)]
        struct SliceAssignOps<const D1: usize, const D2: usize> {
            desc: SliceAssignOperationDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B> for SliceAssignOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D1>(&self.desc.tensor);
                let value = handles.get_int_tensor::<D1>(&self.desc.value);

                let output = B::int_slice_assign::<D1, D2>(
                    tensor,
                    self.desc.ranges.clone().try_into().unwrap(),
                    value,
                );

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);
        let desc = SliceAssignOperationDescription {
            tensor: tensor.into_description(),
            ranges: ranges.into(),
            value: value.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::BaseInt(BaseOperationDescription::SliceAssign(desc.clone())),
            SliceAssignOps::<D1, D2>::new(desc),
        );

        out
    }

    fn int_mask_where<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct MaskWhereOps<const D: usize> {
            desc: MaskWhereOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for MaskWhereOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let value = handles.get_int_tensor(&self.desc.value);
                let mask = handles.get_bool_tensor(&self.desc.mask);

                let output = B::int_mask_where(tensor, mask, value);

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = mask.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = MaskWhereOperationDescription {
            tensor: tensor.into_description(),
            value: value.into_description(),
            mask: mask.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2, stream_3],
            OperationDescription::NumericInt(NumericOperationDescription::MaskWhere(desc.clone())),
            MaskWhereOps::<D>::new(desc),
        );

        out
    }

    fn int_mask_fill<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct MaskFillOps<const D: usize> {
            desc: MaskFillOperationDescription<i32>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for MaskFillOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let mask = handles.get_bool_tensor(&self.desc.mask);

                let output = B::int_mask_fill(tensor, mask, self.desc.value.elem());

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = mask.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);
        let desc = MaskFillOperationDescription {
            tensor: tensor.into_description(),
            value: value.elem(),
            mask: mask.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::MaskFill(desc.clone())),
            MaskFillOps::<D>::new(desc),
        );

        out
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct GatherOps<const D: usize> {
            desc: GatherOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for GatherOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);

                let output = B::int_gather(self.desc.dim, tensor, indices);
                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let shape: Vec<usize> = indices.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);
        let desc = GatherOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::Gather(desc.clone())),
            GatherOps::<D>::new(desc),
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
        struct ScatterOps<const D: usize> {
            desc: ScatterOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for ScatterOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);
                let value = handles.get_int_tensor(&self.desc.value);

                let output = B::int_scatter(self.desc.dim, tensor, indices, value);

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);
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
            ScatterOps::<D>::new(desc),
        );

        out
    }

    fn int_select<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct SelectOps<const D: usize> {
            desc: SelectOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for SelectOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);

                let output = B::int_select(tensor, self.desc.dim, indices);

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let mut shape: Vec<usize> = tensor.shape.clone();
        shape[dim] = indices.shape[0];
        let out = tensor.client.tensor_uninitialized(shape);
        let desc = SelectOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::Select(desc.clone())),
            SelectOps::<D>::new(desc),
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
        struct SelectAssignOps<const D: usize> {
            desc: SelectAssignOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for SelectAssignOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);
                let value = handles.get_int_tensor(&self.desc.value);

                let output = B::int_select_assign(tensor, self.desc.dim, indices, value);

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);
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
            SelectAssignOps::<D>::new(desc),
        );

        out
    }

    fn int_cat<const D: usize>(tensors: Vec<IntTensor<Self, D>>, dim: usize) -> IntTensor<Self, D> {
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
                    .map(|tensor| handles.get_int_tensor(tensor))
                    .collect();

                let output = B::int_cat::<D>(tensors, self.desc.dim);

                handles.register_int_tensor(&self.desc.out.id, output);
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

        let out = client.tensor_uninitialized(shape);

        let desc = CatOperationDescription {
            tensors: tensors.into_iter().map(|t| t.into_description()).collect(),
            dim,
            out: out.to_description_out(),
        };
        client.register(
            streams,
            OperationDescription::BaseInt(BaseOperationDescription::Cat(desc.clone())),
            CatOps::<D>::new(desc),
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
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::BaseInt(BaseOperationDescription::Equal(desc.clone())),
            EqualOps::<D>::new(desc),
        );

        out
    }

    fn int_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(EqualElemOps, B::int_equal_elem);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::EqualElem(desc.clone())),
            EqualElemOps::<D>::new(desc),
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
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::Greater(desc.clone())),
            GreaterOps::<D>::new(desc),
        );

        out
    }

    fn int_greater_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(GreaterElemOps, B::int_greater_elem);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

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
            GreaterElemOps::<D>::new(desc),
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
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

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
            GreaterEqualOps::<D>::new(desc),
        );

        out
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(GreaterEqualElemOps, B::int_greater_equal_elem);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

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
            GreaterEqualElemOps::<D>::new(desc),
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
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::Lower(desc.clone())),
            LowerOps::<D>::new(desc),
        );

        out
    }

    fn int_lower_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(LowerElemOps, B::int_lower_elem);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::LowerElem(desc.clone())),
            LowerElemOps::<D>::new(desc),
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
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericInt(NumericOperationDescription::LowerEqual(desc.clone())),
            LowerEqualOps::<D>::new(desc),
        );

        out
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(LowerEqualElemOps, B::int_lower_equal_elem);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

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
            LowerEqualElemOps::<D>::new(desc),
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
            stream::OperationDescription::NumericInt(NumericOperationDescription::Add(
                desc.clone(),
            )),
            AddOps::<D>::new(desc),
        );

        out
    }

    fn int_add_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(AddOps, B::int_add_scalar);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            stream::OperationDescription::NumericInt(NumericOperationDescription::AddScalar(
                desc.clone(),
            )),
            AddOps::<D>::new(desc),
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
            stream::OperationDescription::NumericInt(NumericOperationDescription::Sub(
                desc.clone(),
            )),
            SubOps::<D>::new(desc),
        );

        out
    }

    fn int_sub_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(SubOps, B::int_sub_scalar);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            stream::OperationDescription::NumericInt(NumericOperationDescription::SubScalar(
                desc.clone(),
            )),
            SubOps::<D>::new(desc),
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
            stream::OperationDescription::NumericInt(NumericOperationDescription::Mul(
                desc.clone(),
            )),
            MulOps::<D>::new(desc),
        );

        out
    }

    fn int_mul_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(MulOps, B::int_mul_scalar);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            stream::OperationDescription::NumericInt(NumericOperationDescription::MulScalar(
                desc.clone(),
            )),
            MulOps::<D>::new(desc),
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
            stream::OperationDescription::NumericInt(NumericOperationDescription::Div(
                desc.clone(),
            )),
            DivOps::<D>::new(desc),
        );

        out
    }

    fn int_div_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(DivOps, B::int_div_scalar);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            stream::OperationDescription::NumericInt(NumericOperationDescription::DivScalar(
                desc.clone(),
            )),
            DivOps::<D>::new(desc),
        );

        out
    }

    fn int_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        #[derive(new)]
        struct ZerosOps<const D: usize> {
            desc: TensorDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for ZerosOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.desc.shape.clone());
                let output = B::int_zeros::<D>(shape, &handles.device);
                handles.register_int_tensor(&self.desc.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);
        let desc = out.to_description_out();
        client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Zeros(desc.clone())),
            ZerosOps::<D>::new(desc),
        );

        out
    }

    fn int_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        #[derive(new)]
        struct OnesOps<const D: usize> {
            desc: TensorDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for OnesOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.desc.shape.clone());
                let output = B::int_ones::<D>(shape, &handles.device);
                handles.register_int_tensor(&self.desc.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);

        let desc = out.to_description_out();
        client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Ones(desc.clone())),
            OnesOps::<D>::new(desc),
        );

        out
    }

    fn int_sum<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(SumOps, B::int_sum);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Sum(desc.clone())),
            SumOps::<D>::new(desc),
        );

        out
    }

    fn int_sum_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(SumDimOps, B::int_sum_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::SumDim(desc.clone())),
            SumDimOps::<D>::new(desc),
        );

        out
    }

    fn int_prod<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(ProdOps, B::int_prod);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Prod(desc.clone())),
            ProdOps::<D>::new(desc),
        );

        out
    }

    fn int_prod_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(ProdDimOps, B::int_prod_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::ProdDim(desc.clone())),
            ProdDimOps::<D>::new(desc),
        );

        out
    }

    fn int_mean<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(MeanOps, B::int_mean);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Mean(desc.clone())),
            MeanOps::<D>::new(desc),
        );

        out
    }

    fn int_mean_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(MeanDimOps, B::int_mean_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::MeanDim(desc.clone())),
            MeanDimOps::<D>::new(desc),
        );

        out
    }

    fn int_argmax<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(ArgMaxOps, B::int_argmax, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::ArgMax(desc.clone())),
            ArgMaxOps::<D>::new(desc),
        );

        out
    }

    fn int_argmin<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(ArgMinOps, B::int_argmin, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::ArgMin(desc.clone())),
            ArgMinOps::<D>::new(desc),
        );

        out
    }

    fn int_clamp<const D: usize>(
        tensor: IntTensor<Self, D>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct ClampOps<const D: usize> {
            desc: ClampOperationDescription<i32>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for ClampOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D>(&self.desc.tensor);
                let output = B::int_clamp(input, self.desc.min.elem(), self.desc.max.elem());

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());
        let desc = ClampOperationDescription {
            tensor: tensor.into_description(),
            min: min.elem(),
            max: max.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Clamp(desc.clone())),
            ClampOps::<D>::new(desc),
        );

        out
    }

    fn int_abs<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, D> {
        unary_int_ops!(AbsOps, B::int_abs);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Abs(desc.clone())),
            AbsOps::<D>::new(desc),
        );

        out
    }

    fn int_into_float<const D: usize>(tensor: IntTensor<Self, D>) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct IntoFloatOps<const D: usize> {
            desc: UnaryOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for IntoFloatOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D>(&self.desc.input);
                let output = B::int_into_float(input);
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
            OperationDescription::Int(stream::IntOperationDescription::IntoFloat(desc.clone())),
            IntoFloatOps::<D>::new(desc),
        );

        out
    }

    fn int_swap_dims<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct SwapDimsOps<const D: usize> {
            desc: SwapDimsDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for SwapDimsOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D>(&self.desc.input);
                let output = B::int_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_int_tensor(&self.desc.out.id, output);
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
            OperationDescription::BaseInt(BaseOperationDescription::SwapDims(desc.clone())),
            SwapDimsOps::<D>::new(desc),
        );

        out
    }

    fn int_max<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(MaxOps, B::int_max);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Max(desc.clone())),
            MaxOps::<D>::new(desc),
        );

        out
    }

    fn int_max_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(MaxDimOps, B::int_max_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::MaxDim(desc.clone())),
            MaxDimOps::<D>::new(desc),
        );

        out
    }

    fn int_max_dim_with_indices<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
    ) -> (IntTensor<Self, D>, IntTensor<Self, D>) {
        #[derive(new)]
        struct MaxDimWithIndicesOps<const D: usize> {
            desc: ReduceDimWithIndicesDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for MaxDimWithIndicesOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let (output, indices) = B::int_max_dim_with_indices(tensor, self.desc.dim);

                handles.register_int_tensor(&self.desc.out.id, output);
                handles.register_int_tensor(&self.desc.out_indices.id, indices);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone());
        let out_indices = client.tensor_uninitialized(shape);
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
            MaxDimWithIndicesOps::<D>::new(desc),
        );

        (out, out_indices)
    }

    fn int_min<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(MinOps, B::int_min);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::Min(desc.clone())),
            MinOps::<D>::new(desc),
        );

        out
    }

    fn int_min_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(MinDimOps, B::int_min_dim, usize, noconvert);

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::MinDim(desc.clone())),
            MinDimOps::<D>::new(desc),
        );

        out
    }

    fn int_min_dim_with_indices<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
    ) -> (IntTensor<Self, D>, IntTensor<Self, D>) {
        #[derive(new)]
        struct MinDimWithIndicesOps<const D: usize> {
            desc: ReduceDimWithIndicesDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for MinDimWithIndicesOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let (output, indices) = B::int_min_dim_with_indices(tensor, self.desc.dim);

                handles.register_int_tensor(&self.desc.out.id, output);
                handles.register_int_tensor(&self.desc.out_indices.id, indices);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone());
        let out_indices = client.tensor_uninitialized(shape);
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
            MinDimWithIndicesOps::<D>::new(desc),
        );

        (out, out_indices)
    }

    fn int_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct IntRandomOps<const D: usize> {
            desc: RandomOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for IntRandomOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.desc.out.shape.clone());
                let output: B::IntTensorPrimitive<D> =
                    B::int_random(shape, self.desc.distribution, &handles.device);
                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);

        let desc = RandomOperationDescription {
            out: out.to_description_out(),
            distribution,
        };
        client.register(
            vec![stream],
            OperationDescription::NumericInt(NumericOperationDescription::IntRandom(desc.clone())),
            IntRandomOps::<D>::new(desc),
        );

        out
    }

    fn int_permute<const D: usize>(
        tensor: IntTensor<Self, D>,
        axes: [usize; D],
    ) -> IntTensor<Self, D> {
        #[derive(new)]
        struct PermuteDimsOps<const D: usize> {
            desc: PermuteOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for PermuteDimsOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D>(&self.desc.input);
                let axes: [usize; D] = self.desc.axes.try_into().unwrap();
                let output = B::int_permute(input, axes);
                handles.register_int_tensor(&self.desc.out.id, output);
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

    fn int_expand<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<Self, D2> {
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
            OperationDescription::BaseInt(BaseOperationDescription::Expand(desc.clone())),
            ExpandOps::<D1, D2>::new(desc),
        );

        out
    }

    fn int_flip<const D: usize>(tensor: IntTensor<Self, D>, axes: &[usize]) -> IntTensor<Self, D> {
        #[derive(new)]
        struct FlipDimsOps<const D: usize> {
            desc: FlipOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for FlipDimsOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D>(&self.desc.input);
                let axes = &self.desc.axes;
                let output = B::int_flip(input, axes);
                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = FlipOperationDescription {
            input: tensor.into_description(),
            axes: axes.to_vec(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseInt(BaseOperationDescription::Flip(desc.clone())),
            FlipDimsOps::<D>::new(desc),
        );

        out
    }
}
