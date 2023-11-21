use crate::{
    binary_int_cmp_ops, binary_int_ops,
    client::FusionClient,
    get_client,
    graph::{
        self, BaseOpsDescription, BinaryOpsDescription, CatOpsDescription, ClampOpsDescription,
        GatherOpsDescription, MaskFillOpsDescription, MaskWhereOpsDescription,
        NumericOpsDescription, Ops, ReduceDimWithIndicesDescription, ReshapeDescription,
        ScalarOpsDescription, ScatterOpsDescription, SelectAssignOpsDescription,
        SelectOpsDescription, SliceAssignOpsDescription, SliceOpsDescription, SwapDimsDescription,
        TensorOpsDescription, UnaryOpsDescription,
    },
    ops::binary::binary_ops_shape,
    scalar_int_cmp_ops, scalar_int_ops, unary_int_ops, Fusion, FusionBackend, TensorDescription,
};
use burn_tensor::{
    ops::{BoolTensor, FloatTensor, IntElem, IntTensor, IntTensorOps},
    Data, Device, ElementConversion, Reader, Shape,
};
use core::ops::Range;

impl<B: FusionBackend> IntTensorOps<Self> for Fusion<B> {
    fn int_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        let client = get_client::<B>(&device.clone().into());
        let tensor = B::int_empty(shape.clone(), device);

        client.register_tensor(B::int_tensor_handle(tensor), shape.dims.into())
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

        client.register_tensor(B::int_tensor_handle(tensor), shape.dims.into())
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

        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.client.clone();

        client_original
            .clone()
            .change_client_int::<D>(tensor.into_description(), client_target)
    }

    fn int_reshape<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> IntTensor<Self, D2> {
        #[derive(new)]
        struct ReshapeDimsOps<const D1: usize, const D2: usize> {
            desc: ReshapeDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for ReshapeDimsOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D1>(&self.desc.input);
                let output = B::int_reshape::<D1, D2>(input, Shape::from(&self.desc.shape));
                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let out = tensor.client.tensor_uninitialized(shape.clone());

        let desc = ReshapeDescription {
            input: tensor.into_description(),
            shape,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::BaseOpsInt(BaseOpsDescription::Reshape(desc.clone())),
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
            desc: SliceOpsDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for SliceOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D1>(&self.desc.tensor);

                let output =
                    B::int_slice::<D1, D2>(tensor, self.desc.ranges.clone().try_into().unwrap());

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..D1 {
            shape.push(tensor.shape[i]);
        }

        let out = tensor.client.tensor_uninitialized(shape);

        let desc = SliceOpsDescription {
            tensor: tensor.into_description(),
            ranges: ranges.into(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::BaseOpsInt(BaseOpsDescription::Slice(desc.clone())),
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
            desc: SliceAssignOpsDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for SliceAssignOps<D1, D2> {
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

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);
        let desc = SliceAssignOpsDescription {
            tensor: tensor.into_description(),
            ranges: ranges.into(),
            value: value.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::BaseOpsInt(BaseOpsDescription::SliceAssign(desc.clone())),
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
            desc: MaskWhereOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for MaskWhereOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let value = handles.get_int_tensor(&self.desc.value);
                let mask = handles.get_bool_tensor(&self.desc.mask);

                let output = B::int_mask_where(tensor, mask, value);

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = MaskWhereOpsDescription {
            tensor: tensor.into_description(),
            value: value.into_description(),
            mask: mask.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::MaskWhere(desc.clone())),
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
            desc: MaskFillOpsDescription<i32>,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for MaskFillOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let mask = handles.get_bool_tensor(&self.desc.mask);

                let output = B::int_mask_fill(tensor, mask, self.desc.value.elem());

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);
        let desc = MaskFillOpsDescription {
            tensor: tensor.into_description(),
            value: value.elem(),
            mask: mask.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::MaskFill(desc.clone())),
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
            desc: GatherOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for GatherOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);

                let output = B::int_gather(self.desc.dim, tensor, indices);
                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let shape: Vec<usize> = indices.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);
        let desc = GatherOpsDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Gather(desc.clone())),
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
            desc: ScatterOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for ScatterOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);
                let value = handles.get_int_tensor(&self.desc.value);

                let output = B::int_scatter(self.desc.dim, tensor, indices, value);

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);
        let desc = ScatterOpsDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            value: value.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Scatter(desc.clone())),
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
            desc: SelectOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for SelectOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);

                let output = B::int_select(tensor, self.desc.dim, indices);

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let mut shape: Vec<usize> = tensor.shape.clone();
        shape[dim] = indices.shape[0];
        let out = tensor.client.tensor_uninitialized(shape);
        let desc = SelectOpsDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Select(desc.clone())),
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
            desc: SelectAssignOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for SelectAssignOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);
                let value = handles.get_int_tensor(&self.desc.value);

                let output = B::int_select_assign(tensor, self.desc.dim, indices, value);

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);
        let desc = SelectAssignOpsDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            value: value.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::SelectAssign(desc.clone())),
            SelectAssignOps::<D>::new(desc),
        );

        out
    }

    fn int_cat<const D: usize>(tensors: Vec<IntTensor<Self, D>>, dim: usize) -> IntTensor<Self, D> {
        #[derive(new)]
        struct CatOps<const D: usize> {
            desc: CatOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for CatOps<D> {
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

        let tensor_first = tensors.get(0).unwrap();
        let client = tensor_first.client.clone();

        // Calculate the output shape
        let mut shape: Vec<usize> = tensor_first.shape.clone();
        shape[dim] = 0;
        for tensor in tensors.iter() {
            shape[dim] += tensor.shape[dim];
        }

        let out = client.tensor_uninitialized(shape);

        let desc = CatOpsDescription {
            tensors: tensors.into_iter().map(|t| t.into_description()).collect(),
            dim,
            out: out.to_description_out(),
        };
        client.register(
            TensorOpsDescription::BaseOpsInt(BaseOpsDescription::Cat(desc.clone())),
            CatOps::<D>::new(desc),
        );

        out
    }

    fn int_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_int_cmp_ops!(EqualOps, B::int_equal);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::BaseOpsInt(BaseOpsDescription::Equal(desc.clone())),
            EqualOps::<D>::new(desc),
        );

        out
    }

    fn int_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(EqualElemOps, B::int_equal_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::EqualElem(desc.clone())),
            EqualElemOps::<D>::new(desc),
        );

        out
    }

    fn int_greater<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_int_cmp_ops!(GreaterOps, B::int_greater);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Greater(desc.clone())),
            GreaterOps::<D>::new(desc),
        );

        out
    }

    fn int_greater_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(GreaterElemOps, B::int_greater_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::GreaterElem(desc.clone())),
            GreaterElemOps::<D>::new(desc),
        );

        out
    }

    fn int_greater_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_int_cmp_ops!(GreaterEqualOps, B::int_greater_equal);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::GreaterEqual(desc.clone())),
            GreaterEqualOps::<D>::new(desc),
        );

        out
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(GreaterEqualElemOps, B::int_greater_equal_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::GreaterEqualElem(
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

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Lower(desc.clone())),
            LowerOps::<D>::new(desc),
        );

        out
    }

    fn int_lower_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(LowerElemOps, B::int_lower_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::LowerElem(desc.clone())),
            LowerElemOps::<D>::new(desc),
        );

        out
    }

    fn int_lower_equal<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_int_cmp_ops!(LowerEqualOps, B::int_lower_equal);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::LowerEqual(desc.clone())),
            LowerEqualOps::<D>::new(desc),
        );

        out
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(LowerEqualElemOps, B::int_lower_equal_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::LowerEqualElem(
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

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            graph::TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Add(desc.clone())),
            AddOps::<D>::new(desc),
        );

        out
    }

    fn int_add_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(AddOps, B::int_add_scalar);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            graph::TensorOpsDescription::NumericOpsInt(NumericOpsDescription::AddScalar(
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

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            graph::TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Sub(desc.clone())),
            SubOps::<D>::new(desc),
        );

        out
    }

    fn int_sub_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(SubOps, B::int_sub_scalar);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            graph::TensorOpsDescription::NumericOpsInt(NumericOpsDescription::SubScalar(
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

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            graph::TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Mul(desc.clone())),
            MulOps::<D>::new(desc),
        );

        out
    }

    fn int_mul_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(MulOps, B::int_mul_scalar);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            graph::TensorOpsDescription::NumericOpsInt(NumericOpsDescription::MulScalar(
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

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            graph::TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Div(desc.clone())),
            DivOps::<D>::new(desc),
        );

        out
    }

    fn int_div_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(DivOps, B::int_div_scalar);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            graph::TensorOpsDescription::NumericOpsInt(NumericOpsDescription::DivScalar(
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

        impl<const D: usize, B: FusionBackend> Ops<B> for ZerosOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.desc.shape.clone());
                let output = B::int_zeros::<D>(shape, &handles.device);
                handles.register_int_tensor(&self.desc.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);
        let desc = out.to_description_out();
        client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Zeros(desc.clone())),
            ZerosOps::<D>::new(desc),
        );

        out
    }

    fn int_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        #[derive(new)]
        struct OnesOps<const D: usize> {
            desc: TensorDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for OnesOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.desc.shape.clone());
                let output = B::int_ones::<D>(shape, &handles.device);
                handles.register_int_tensor(&self.desc.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);

        let desc = out.to_description_out();
        client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Ones(desc.clone())),
            OnesOps::<D>::new(desc),
        );

        out
    }

    fn int_sum<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(SumOps, B::int_sum);

        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Sum(desc.clone())),
            SumOps::<D>::new(desc),
        );

        out
    }

    fn int_sum_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(SumDimOps, B::int_sum_dim, usize, noconvert);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::SumDim(desc.clone())),
            SumDimOps::<D>::new(desc),
        );

        out
    }

    fn int_mean<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(MeanOps, B::int_mean);

        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Mean(desc.clone())),
            MeanOps::<D>::new(desc),
        );

        out
    }

    fn int_mean_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(MeanDimOps, B::int_mean_dim, usize, noconvert);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::MeanDim(desc.clone())),
            MeanDimOps::<D>::new(desc),
        );

        out
    }

    fn int_argmax<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(ArgMaxOps, B::int_argmax, usize, noconvert);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::ArgMax(desc.clone())),
            ArgMaxOps::<D>::new(desc),
        );

        out
    }

    fn int_argmin<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(ArgMinOps, B::int_argmin, usize, noconvert);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::ArgMin(desc.clone())),
            ArgMinOps::<D>::new(desc),
        );

        out
    }

    fn int_clamp_min<const D: usize>(
        tensor: IntTensor<Self, D>,
        min: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(ClampMinOps, B::int_clamp_min);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: min.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::ClampMin(desc.clone())),
            ClampMinOps::<D>::new(desc),
        );

        out
    }

    fn int_clamp_max<const D: usize>(
        tensor: IntTensor<Self, D>,
        max: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(ClampMaxOps, B::int_clamp_max);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: max.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::ClampMax(desc.clone())),
            ClampMaxOps::<D>::new(desc),
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
            desc: ClampOpsDescription<i32>,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for ClampOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D>(&self.desc.tensor);
                let output = B::int_clamp(input, self.desc.min.elem(), self.desc.max.elem());

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());
        let desc = ClampOpsDescription {
            tensor: tensor.into_description(),
            min: min.elem(),
            max: max.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Clamp(desc.clone())),
            ClampOps::<D>::new(desc),
        );

        out
    }

    fn int_abs<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, D> {
        unary_int_ops!(AbsOps, B::int_abs);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Abs(desc.clone())),
            AbsOps::<D>::new(desc),
        );

        out
    }

    fn int_into_float<const D: usize>(tensor: IntTensor<Self, D>) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct IntoFloatOps<const D: usize> {
            desc: UnaryOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for IntoFloatOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D>(&self.desc.input);
                let output = B::int_into_float(input);
                handles.register_float_tensor(&self.desc.out.id, output);
            }
        }

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());
        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::IntOps(graph::IntOpsDescription::IntoFloat(desc.clone())),
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

        impl<const D: usize, B: FusionBackend> Ops<B> for SwapDimsOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D>(&self.desc.input);
                let output = B::int_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

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
            TensorOpsDescription::BaseOpsInt(BaseOpsDescription::SwapDims(desc.clone())),
            SwapDimsOps::<D>::new(desc),
        );

        out
    }

    fn int_max<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(MaxOps, B::int_max);

        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Max(desc.clone())),
            MaxOps::<D>::new(desc),
        );

        out
    }

    fn int_max_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(MaxDimOps, B::int_max_dim, usize, noconvert);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::MaxDim(desc.clone())),
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

        impl<const D: usize, B: FusionBackend> Ops<B> for MaxDimWithIndicesOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let (output, indices) = B::int_max_dim_with_indices(tensor, self.desc.dim);

                handles.register_int_tensor(&self.desc.out.id, output);
                handles.register_int_tensor(&self.desc.out_indices.id, indices);
            }
        }

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
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::MaxDimWithIndices(
                desc.clone(),
            )),
            MaxDimWithIndicesOps::<D>::new(desc),
        );

        (out, out_indices)
    }

    fn int_min<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(MinOps, B::int_min);

        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::Min(desc.clone())),
            MinOps::<D>::new(desc),
        );

        out
    }

    fn int_min_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(MinDimOps, B::int_min_dim, usize, noconvert);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::MinDim(desc.clone())),
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

        impl<const D: usize, B: FusionBackend> Ops<B> for MinDimWithIndicesOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&self.desc.tensor);
                let (output, indices) = B::int_min_dim_with_indices(tensor, self.desc.dim);

                handles.register_int_tensor(&self.desc.out.id, output);
                handles.register_int_tensor(&self.desc.out_indices.id, indices);
            }
        }

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
            TensorOpsDescription::NumericOpsInt(NumericOpsDescription::MinDimWithIndices(
                desc.clone(),
            )),
            MinDimWithIndicesOps::<D>::new(desc),
        );

        (out, out_indices)
    }
}
