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
    Data, Device, Reader, Shape,
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
        struct ReshapeDimsOps<const D1: usize, const D2: usize>;

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for ReshapeDimsOps<D1, D2> {
            type Args = ReshapeDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D1>(&args.input);
                let output = B::int_reshape::<D1, D2>(input, Shape::from(&args.shape));
                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let out = tensor.client.tensor_uninitialized(shape.clone());

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::BaseOpsInt(
                BaseOpsDescription::Reshape(
                    ReshapeDescription {
                        input: tensor.into_description(),
                        shape,
                        out: out.to_description_out(),
                    },
                    Box::new(ReshapeDimsOps::<D1, D2>),
                ),
            ));

        out
    }

    fn int_slice<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> IntTensor<Self, D1> {
        struct SliceOps<const D1: usize, const D2: usize>;

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for SliceOps<D1, D2> {
            type Args = SliceOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D1>(&args.tensor);

                let output =
                    B::int_slice::<D1, D2>(tensor, args.ranges.clone().try_into().unwrap());

                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..D1 {
            shape.push(tensor.shape[i]);
        }

        let out = tensor.client.tensor_uninitialized(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::BaseOpsInt(BaseOpsDescription::Slice(
                SliceOpsDescription {
                    tensor: tensor.into_description(),
                    ranges: ranges.into(),
                    out: out.to_description_out(),
                },
                Box::new(SliceOps::<D1, D2>),
            )));

        out
    }

    fn int_slice_assign<const D1: usize, const D2: usize>(
        tensor: IntTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: IntTensor<Self, D1>,
    ) -> IntTensor<Self, D1> {
        struct SliceAssignOps<const D1: usize, const D2: usize>;

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for SliceAssignOps<D1, D2> {
            type Args = SliceAssignOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D1>(&args.tensor);
                let value = handles.get_int_tensor::<D1>(&args.value);

                let output = B::int_slice_assign::<D1, D2>(
                    tensor,
                    args.ranges.clone().try_into().unwrap(),
                    value,
                );

                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::BaseOpsInt(
                BaseOpsDescription::SliceAssign(
                    SliceAssignOpsDescription {
                        tensor: tensor.into_description(),
                        ranges: ranges.into(),
                        value: value.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(SliceAssignOps::<D1, D2>),
                ),
            ));

        out
    }

    fn int_mask_where<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        struct MaskWhereOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for MaskWhereOps<D> {
            type Args = MaskWhereOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&args.tensor);
                let value = handles.get_int_tensor(&args.value);
                let mask = handles.get_bool_tensor(&args.mask);

                let output = B::int_mask_where(tensor, mask, value);

                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::MaskWhere(
                    MaskWhereOpsDescription {
                        tensor: tensor.into_description(),
                        value: value.into_description(),
                        mask: mask.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(MaskWhereOps::<D>),
                ),
            ));

        out
    }

    fn int_mask_fill<const D: usize>(
        tensor: IntTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        struct MaskFillOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for MaskFillOps<D> {
            type Args = MaskFillOpsDescription<IntElem<B>>;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&args.tensor);
                let mask = handles.get_bool_tensor(&args.mask);

                let output = B::int_mask_fill(tensor, mask, args.value);

                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::MaskFill(
                    MaskFillOpsDescription {
                        tensor: tensor.into_description(),
                        value,
                        mask: mask.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(MaskFillOps::<D>),
                ),
            ));

        out
    }

    fn int_gather<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        struct GatherOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for GatherOps<D> {
            type Args = GatherOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&args.tensor);
                let indices = handles.get_int_tensor(&args.indices);

                let output = B::int_gather(args.dim, tensor, indices);
                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = indices.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::Gather(
                    GatherOpsDescription {
                        tensor: tensor.into_description(),
                        dim,
                        indices: indices.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(GatherOps::<D>),
                ),
            ));

        out
    }

    fn int_scatter<const D: usize>(
        dim: usize,
        tensor: IntTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        struct ScatterOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for ScatterOps<D> {
            type Args = ScatterOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&args.tensor);
                let indices = handles.get_int_tensor(&args.indices);
                let value = handles.get_int_tensor(&args.value);

                let output = B::int_scatter(args.dim, tensor, indices, value);

                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::Scatter(
                    ScatterOpsDescription {
                        tensor: tensor.into_description(),
                        dim,
                        indices: indices.into_description(),
                        value: value.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(ScatterOps::<D>),
                ),
            ));

        out
    }

    fn int_select<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> IntTensor<Self, D> {
        struct SelectOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for SelectOps<D> {
            type Args = SelectOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&args.tensor);
                let indices = handles.get_int_tensor(&args.indices);

                let output = B::int_select(tensor, args.dim, indices);

                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let mut shape: Vec<usize> = tensor.shape.clone();
        shape[dim] = indices.shape[0];
        let out = tensor.client.tensor_uninitialized(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::Select(
                    SelectOpsDescription {
                        tensor: tensor.into_description(),
                        dim,
                        indices: indices.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(SelectOps::<D>),
                ),
            ));

        out
    }

    fn int_select_assign<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: IntTensor<Self, D>,
    ) -> IntTensor<Self, D> {
        struct SelectAssignOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for SelectAssignOps<D> {
            type Args = SelectAssignOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&args.tensor);
                let indices = handles.get_int_tensor(&args.indices);
                let value = handles.get_int_tensor(&args.value);

                let output = B::int_select_assign(tensor, args.dim, indices, value);

                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::SelectAssign(
                    SelectAssignOpsDescription {
                        tensor: tensor.into_description(),
                        dim,
                        indices: indices.into_description(),
                        value: value.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(SelectAssignOps::<D>),
                ),
            ));

        out
    }

    fn int_cat<const D: usize>(tensors: Vec<IntTensor<Self, D>>, dim: usize) -> IntTensor<Self, D> {
        struct CatOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for CatOps<D> {
            type Args = CatOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensors = args
                    .tensors
                    .iter()
                    .map(|tensor| handles.get_int_tensor(tensor))
                    .collect();

                let output = B::int_cat::<D>(tensors, args.dim);

                handles.register_int_tensor(&args.out.id, output);
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

        client.register(TensorOpsDescription::BaseOpsInt(BaseOpsDescription::Cat(
            CatOpsDescription {
                tensors: tensors.into_iter().map(|t| t.into_description()).collect(),
                dim,
                out: out.to_description_out(),
            },
            Box::new(CatOps::<D>),
        )));

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

        out.client
            .register(TensorOpsDescription::BaseOpsInt(BaseOpsDescription::Equal(
                BinaryOpsDescription {
                    lhs: lhs.into_description(),
                    rhs: rhs.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(EqualOps::<D>),
            )));

        out
    }

    fn int_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(EqualElemOps, B::int_equal_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::EqualElem(
                ScalarOpsDescription {
                    lhs: lhs.into_description(),
                    rhs,
                    out: out.to_description_out(),
                },
                Box::new(EqualElemOps::<D>),
            ),
        ));

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

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::Greater(
                BinaryOpsDescription {
                    lhs: lhs.into_description(),
                    rhs: rhs.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(GreaterOps::<D>),
            ),
        ));

        out
    }

    fn int_greater_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(GreaterElemOps, B::int_greater_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::GreaterElem(
                ScalarOpsDescription {
                    lhs: lhs.into_description(),
                    rhs,
                    out: out.to_description_out(),
                },
                Box::new(GreaterElemOps::<D>),
            ),
        ));

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

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::GreaterEqual(
                BinaryOpsDescription {
                    lhs: lhs.into_description(),
                    rhs: rhs.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(GreaterEqualOps::<D>),
            ),
        ));

        out
    }

    fn int_greater_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(GreaterEqualElemOps, B::int_greater_equal_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::GreaterEqualElem(
                ScalarOpsDescription {
                    lhs: lhs.into_description(),
                    rhs,
                    out: out.to_description_out(),
                },
                Box::new(GreaterEqualElemOps::<D>),
            ),
        ));

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

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::Lower(
                BinaryOpsDescription {
                    lhs: lhs.into_description(),
                    rhs: rhs.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(LowerOps::<D>),
            ),
        ));

        out
    }

    fn int_lower_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(LowerElemOps, B::int_lower_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::LowerElem(
                ScalarOpsDescription {
                    lhs: lhs.into_description(),
                    rhs,
                    out: out.to_description_out(),
                },
                Box::new(LowerElemOps::<D>),
            ),
        ));

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

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::LowerEqual(
                BinaryOpsDescription {
                    lhs: lhs.into_description(),
                    rhs: rhs.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(LowerEqualOps::<D>),
            ),
        ));

        out
    }

    fn int_lower_equal_elem<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_int_cmp_ops!(LowerEqualElemOps, B::int_lower_equal_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::LowerEqualElem(
                ScalarOpsDescription {
                    lhs: lhs.into_description(),
                    rhs,
                    out: out.to_description_out(),
                },
                Box::new(LowerEqualElemOps::<D>),
            ),
        ));

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

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::Add(
                    BinaryOpsDescription {
                        lhs: lhs.into_description(),
                        rhs: rhs.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(AddOps::<D>),
                ),
            ));

        out
    }

    fn int_add_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(AddOps, B::int_add_scalar);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::AddScalar(
                    ScalarOpsDescription {
                        lhs: lhs.into_description(),
                        rhs,
                        out: out.to_description_out(),
                    },
                    Box::new(AddOps::<D>),
                ),
            ));

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

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::Sub(
                    BinaryOpsDescription {
                        lhs: lhs.into_description(),
                        rhs: rhs.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(SubOps::<D>),
                ),
            ));

        out
    }

    fn int_sub_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(SubOps, B::int_sub_scalar);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::SubScalar(
                    ScalarOpsDescription {
                        lhs: lhs.into_description(),
                        rhs,
                        out: out.to_description_out(),
                    },
                    Box::new(SubOps::<D>),
                ),
            ));

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

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::Mul(
                    BinaryOpsDescription {
                        lhs: lhs.into_description(),
                        rhs: rhs.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(MulOps::<D>),
                ),
            ));

        out
    }

    fn int_mul_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(MulOps, B::int_mul_scalar);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::MulScalar(
                    ScalarOpsDescription {
                        lhs: lhs.into_description(),
                        rhs,
                        out: out.to_description_out(),
                    },
                    Box::new(MulOps::<D>),
                ),
            ));

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

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::Div(
                    BinaryOpsDescription {
                        lhs: lhs.into_description(),
                        rhs: rhs.into_description(),
                        out: out.to_description_out(),
                    },
                    Box::new(DivOps::<D>),
                ),
            ));

        out
    }

    fn int_div_scalar<const D: usize>(
        lhs: IntTensor<Self, D>,
        rhs: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(DivOps, B::int_div_scalar);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        out.client
            .register(graph::TensorOpsDescription::NumericOpsInt(
                NumericOpsDescription::DivScalar(
                    ScalarOpsDescription {
                        lhs: lhs.into_description(),
                        rhs,
                        out: out.to_description_out(),
                    },
                    Box::new(DivOps::<D>),
                ),
            ));

        out
    }

    fn int_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        struct ZerosOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for ZerosOps<D> {
            type Args = TensorDescription;

            fn execute(&self, out: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(out.shape.clone());
                let output = B::int_zeros::<D>(shape, &handles.device);
                handles.register_int_tensor(&out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);

        client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::Zeros(out.to_description_out(), Box::new(ZerosOps::<D>)),
        ));

        out
    }

    fn int_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> IntTensor<Self, D> {
        struct OnesOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for OnesOps<D> {
            type Args = TensorDescription;

            fn execute(&self, out: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(out.shape.clone());
                let output = B::int_ones::<D>(shape, &handles.device);
                handles.register_int_tensor(&out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);

        client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::Ones(out.to_description_out(), Box::new(OnesOps::<D>)),
        ));

        out
    }

    fn int_sum<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(SumOps, B::int_sum);

        let out = tensor.client.tensor_uninitialized(vec![1]);

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::Sum(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(SumOps::<D>),
            ),
        ));

        out
    }

    fn int_sum_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(SumDimOps, B::int_sum_dim, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::SumDim(
                ScalarOpsDescription {
                    lhs: tensor.into_description(),
                    rhs: dim,
                    out: out.to_description_out(),
                },
                Box::new(SumDimOps::<D>),
            ),
        ));

        out
    }

    fn int_mean<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(MeanOps, B::int_mean);

        let out = tensor.client.tensor_uninitialized(vec![1]);

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::Mean(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(MeanOps::<D>),
            ),
        ));

        out
    }

    fn int_mean_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(MeanDimOps, B::int_mean_dim, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::MeanDim(
                ScalarOpsDescription {
                    lhs: tensor.into_description(),
                    rhs: dim,
                    out: out.to_description_out(),
                },
                Box::new(MeanDimOps::<D>),
            ),
        ));

        out
    }

    fn int_argmax<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(ArgMaxOps, B::int_argmax, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::ArgMax(
                ScalarOpsDescription {
                    lhs: tensor.into_description(),
                    rhs: dim,
                    out: out.to_description_out(),
                },
                Box::new(ArgMaxOps::<D>),
            ),
        ));

        out
    }

    fn int_argmin<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(ArgMinOps, B::int_argmin, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::ArgMin(
                ScalarOpsDescription {
                    lhs: tensor.into_description(),
                    rhs: dim,
                    out: out.to_description_out(),
                },
                Box::new(ArgMinOps::<D>),
            ),
        ));

        out
    }

    fn int_clamp_min<const D: usize>(
        tensor: IntTensor<Self, D>,
        min: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(ClampMinOps, B::int_clamp_min);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::ClampMin(
                ScalarOpsDescription {
                    lhs: tensor.into_description(),
                    rhs: min,
                    out: out.to_description_out(),
                },
                Box::new(ClampMinOps::<D>),
            ),
        ));

        out
    }

    fn int_clamp_max<const D: usize>(
        tensor: IntTensor<Self, D>,
        max: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        scalar_int_ops!(ClampMaxOps, B::int_clamp_max);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::ClampMax(
                ScalarOpsDescription {
                    lhs: tensor.into_description(),
                    rhs: max,
                    out: out.to_description_out(),
                },
                Box::new(ClampMaxOps::<D>),
            ),
        ));

        out
    }

    fn int_clamp<const D: usize>(
        tensor: IntTensor<Self, D>,
        min: IntElem<Self>,
        max: IntElem<Self>,
    ) -> IntTensor<Self, D> {
        struct ClampOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for ClampOps<D> {
            type Args = ClampOpsDescription<IntElem<B>>;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D>(&args.tensor);
                let output = B::int_clamp(input, args.min, args.max);

                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::Clamp(
                ClampOpsDescription {
                    tensor: tensor.into_description(),
                    min,
                    max,
                    out: out.to_description_out(),
                },
                Box::new(ClampOps::<D>),
            ),
        ));

        out
    }

    fn int_abs<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, D> {
        unary_int_ops!(AbsOps, B::int_abs);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::Abs(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(AbsOps::<D>),
            ),
        ));

        out
    }

    fn int_into_float<const D: usize>(tensor: IntTensor<Self, D>) -> FloatTensor<Self, D> {
        struct IntoFloatOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for IntoFloatOps<D> {
            type Args = UnaryOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D>(&args.input);
                let output = B::int_into_float(input);
                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        out.client.register(TensorOpsDescription::IntOps(
            graph::IntOpsDescription::IntoFloat(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(IntoFloatOps::<D>),
            ),
        ));

        out
    }

    fn int_swap_dims<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> IntTensor<Self, D> {
        struct SwapDimsOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for SwapDimsOps<D> {
            type Args = SwapDimsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_int_tensor::<D>(&args.input);
                let output = B::int_swap_dims(input, args.dim1, args.dim2);
                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];

        let out = tensor.client.tensor_uninitialized(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::BaseOpsInt(
                BaseOpsDescription::SwapDims(
                    SwapDimsDescription {
                        input: tensor.into_description(),
                        dim1,
                        dim2,
                        out: out.to_description_out(),
                    },
                    Box::new(SwapDimsOps::<D>),
                ),
            ));

        out
    }

    fn int_max<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(MaxOps, B::int_max);

        let out = tensor.client.tensor_uninitialized(vec![1]);

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::Max(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(MaxOps::<D>),
            ),
        ));

        out
    }

    fn int_max_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(MaxDimOps, B::int_max_dim, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::MaxDim(
                ScalarOpsDescription {
                    lhs: tensor.into_description(),
                    rhs: dim,
                    out: out.to_description_out(),
                },
                Box::new(MaxDimOps::<D>),
            ),
        ));

        out
    }

    fn int_max_dim_with_indices<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
    ) -> (IntTensor<Self, D>, IntTensor<Self, D>) {
        struct MaxDimWithIndicesOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for MaxDimWithIndicesOps<D> {
            type Args = ReduceDimWithIndicesDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&args.tensor);
                let (output, indices) = B::int_max_dim_with_indices(tensor, args.dim);

                handles.register_int_tensor(&args.out.id, output);
                handles.register_int_tensor(&args.out_indices.id, indices);
            }
        }

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone());
        let out_indices = client.tensor_uninitialized(shape);

        client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::MaxDimWithIndices(
                ReduceDimWithIndicesDescription {
                    tensor: tensor.into_description(),
                    dim,
                    out: out.to_description_out(),
                    out_indices: out_indices.to_description_out(),
                },
                Box::new(MaxDimWithIndicesOps::<D>),
            ),
        ));

        (out, out_indices)
    }

    fn int_min<const D: usize>(tensor: IntTensor<Self, D>) -> IntTensor<Self, 1> {
        unary_int_ops!(MinOps, B::int_min);

        let out = tensor.client.tensor_uninitialized(vec![1]);

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::Min(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(MinOps::<D>),
            ),
        ));

        out
    }

    fn int_min_dim<const D: usize>(tensor: IntTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_int_ops!(MinDimOps, B::int_min_dim, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        out.client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::MinDim(
                ScalarOpsDescription {
                    lhs: tensor.into_description(),
                    rhs: dim,
                    out: out.to_description_out(),
                },
                Box::new(MinDimOps::<D>),
            ),
        ));

        out
    }

    fn int_min_dim_with_indices<const D: usize>(
        tensor: IntTensor<Self, D>,
        dim: usize,
    ) -> (IntTensor<Self, D>, IntTensor<Self, D>) {
        struct MinDimWithIndicesOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for MinDimWithIndicesOps<D> {
            type Args = ReduceDimWithIndicesDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_int_tensor::<D>(&args.tensor);
                let (output, indices) = B::int_min_dim_with_indices(tensor, args.dim);

                handles.register_int_tensor(&args.out.id, output);
                handles.register_int_tensor(&args.out_indices.id, indices);
            }
        }

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone());
        let out_indices = client.tensor_uninitialized(shape);

        client.register(TensorOpsDescription::NumericOpsInt(
            NumericOpsDescription::MinDimWithIndices(
                ReduceDimWithIndicesDescription {
                    tensor: tensor.into_description(),
                    dim,
                    out: out.to_description_out(),
                    out_indices: out_indices.to_description_out(),
                },
                Box::new(MinDimWithIndicesOps::<D>),
            ),
        ));

        (out, out_indices)
    }
}
