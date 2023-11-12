use crate::{
    binary_float_cmp_ops, binary_float_ops,
    client::FusionClient,
    get_client,
    graph::{
        BaseOpsDescription, BinaryOpsDescription, CatOpsDescription, ClampOpsDescription,
        FloatOpsDescription, GatherOpsDescription, MaskFillOpsDescription, MaskWhereOpsDescription,
        NumericOpsDescription, Ops, ReduceDimWithIndicesDescription, ReshapeDescription,
        ScalarOpsDescription, ScatterOpsDescription, SelectAssignOpsDescription,
        SelectOpsDescription, SliceAssignOpsDescription, SliceOpsDescription, SwapDimsDescription,
        TensorOpsDescription, UnaryOpsDescription,
    },
    ops::binary::binary_ops_shape,
    scalar_float2int_ops, scalar_float_cmp_ops, scalar_float_ops, unary_float_ops, Fusion,
    FusionBackend, TensorDescription,
};
use burn_tensor::{
    ops::{BoolTensor, FloatElem, FloatTensor, FullPrecisionBackend, IntTensor, TensorOps},
    Data, Device, Distribution, Reader, Shape,
};
use std::ops::Range;

impl<B: FusionBackend> TensorOps<Self> for Fusion<B> {
    fn from_data<const D: usize>(
        data: Data<FloatElem<Self>, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        let client = get_client::<B>(&device.clone().into());

        client.create_tensor_float(data.value, data.shape.dims.into())
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<FloatElem<Self>>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        struct RandomOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for RandomOps<D> {
            type Args = (TensorDescription, Distribution<FloatElem<B>>);

            fn execute(
                &self,
                (out, distribution): &Self::Args,
                handles: &mut crate::HandleContainer<B>,
            ) {
                let shape = Shape::from(out.shape.clone());
                let output: B::TensorPrimitive<D> =
                    B::random(shape, *distribution, &handles.device);
                handles.register_float_tensor(&out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.create_tensor_empty(shape);

        client.register(TensorOpsDescription::FloatOps(FloatOpsDescription::Random(
            (out.to_description_out(), distribution),
            Box::new(RandomOps::<D>),
        )));

        out
    }

    fn zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        struct ZerosOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for ZerosOps<D> {
            type Args = TensorDescription;

            fn execute(&self, out: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(out.shape.clone());
                let output = B::zeros::<D>(shape, &handles.device);
                handles.register_float_tensor(&out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.create_tensor_empty(shape);

        client.register(TensorOpsDescription::NumericOpsFloat(
            NumericOpsDescription::Zeros(out.to_description_out(), Box::new(ZerosOps::<D>)),
        ));

        out
    }

    fn ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        struct OnesOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for OnesOps<D> {
            type Args = TensorDescription;

            fn execute(&self, out: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(out.shape.clone());
                let output = B::ones::<D>(shape, &handles.device);
                handles.register_float_tensor(&out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.create_tensor_empty(shape);

        client.register(TensorOpsDescription::NumericOpsFloat(
            NumericOpsDescription::Ones(out.to_description_out(), Box::new(OnesOps::<D>)),
        ));

        out
    }

    fn full<const D: usize>(
        shape: Shape<D>,
        fill_value: FloatElem<Self>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        struct FullOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for FullOps<D> {
            type Args = (TensorDescription, FloatElem<B>);

            fn execute(&self, (out, value): &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(out.shape.clone());
                let output: B::TensorPrimitive<D> = B::full(shape, *value, &handles.device);
                handles.register_float_tensor(&out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.create_tensor_empty(shape);

        client.register(TensorOpsDescription::NumericOpsFloat(
            NumericOpsDescription::Full(
                (out.to_description_out(), fill_value),
                Box::new(FullOps::<D>),
            ),
        ));

        out
    }

    fn shape<const D: usize>(tensor: &FloatTensor<Self, D>) -> Shape<D> {
        tensor.shape()
    }

    fn into_data<const D: usize>(tensor: FloatTensor<Self, D>) -> Reader<Data<FloatElem<Self>, D>> {
        tensor.into_data()
    }

    fn device<const D: usize>(tensor: &FloatTensor<Self, D>) -> Device<Self> {
        tensor.client.device().clone().into()
    }

    fn to_device<const D: usize>(
        tensor: FloatTensor<Self, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        let device_original: &B::FusionDevice = tensor.client.device();
        let device_target: B::FusionDevice = device.clone().into();

        if device_original == &device_target {
            return tensor;
        }

        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.client.clone();

        client_original
            .clone()
            .change_client_float::<D>(tensor.into_description(), client_target)
    }

    fn into_int<const D: usize>(tensor: FloatTensor<Self, D>) -> IntTensor<Self, D> {
        struct IntoIntOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for IntoIntOps<D> {
            type Args = UnaryOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&args.input);
                let output = B::into_int(input);

                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());

        out.client.register(TensorOpsDescription::FloatOps(
            FloatOpsDescription::IntoInt(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(IntoIntOps::<D>),
            ),
        ));

        out
    }

    fn empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        let client = get_client::<B>(&device.clone().into());

        client.create_tensor_empty(shape.dims.into())
    }

    fn add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(AddOps, B::add);

        let out = lhs
            .client
            .create_tensor_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(AddOps, B::add_scalar);

        let out = lhs.client.create_tensor_empty(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn clamp_min<const D: usize>(
        tensor: FloatTensor<Self, D>,
        min: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(ClampMinOps, B::clamp_min);

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn clamp_max<const D: usize>(
        tensor: FloatTensor<Self, D>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(ClampMaxOps, B::clamp_max);

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn clamp<const D: usize>(
        tensor: FloatTensor<Self, D>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        struct ClampOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for ClampOps<D> {
            type Args = ClampOpsDescription<FloatElem<B>>;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&args.tensor);
                let output = B::clamp(input, args.min, args.max);

                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(SubOps, B::sub);

        let out = lhs
            .client
            .create_tensor_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(SubOps, B::sub_scalar);

        let out = lhs.client.create_tensor_empty(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn mul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(MulOps, B::mul);

        let out = lhs
            .client
            .create_tensor_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(MulOps, B::mul_scalar);

        let out = lhs.client.create_tensor_empty(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn div<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(DivOps, B::div);

        let out = lhs
            .client
            .create_tensor_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(DivOps, B::div_scalar);

        let out = lhs.client.create_tensor_empty(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn matmul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(MatmulOps, B::matmul);

        let mut shape = binary_ops_shape(&lhs.shape, &rhs.shape);

        shape[D - 2] = lhs.shape[D - 2];
        shape[D - 1] = rhs.shape[D - 1];

        let out = lhs.client.create_tensor_empty(shape);

        out.client
            .register(TensorOpsDescription::FloatOps(FloatOpsDescription::Matmul(
                BinaryOpsDescription {
                    lhs: lhs.into_description(),
                    rhs: rhs.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(MatmulOps::<D>),
            )));

        out
    }

    fn swap_dims<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<Self, D> {
        struct SwapDimsOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for SwapDimsOps<D> {
            type Args = SwapDimsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&args.input);
                let output = B::swap_dims(input, args.dim1, args.dim2);
                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];

        let out = tensor.client.create_tensor_empty(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::BaseOpsFloat(
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

    fn reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        struct ReshapeDimsOps<const D1: usize, const D2: usize>;

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for ReshapeDimsOps<D1, D2> {
            type Args = ReshapeDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D1>(&args.input);
                let output = B::reshape::<D1, D2>(input, Shape::from(&args.shape));
                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let out = tensor.client.create_tensor_empty(shape.clone());

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::BaseOpsFloat(
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

    fn gather<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        struct GatherOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for GatherOps<D> {
            type Args = GatherOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&args.tensor);
                let indices = handles.get_int_tensor(&args.indices);

                let output = B::gather(args.dim, tensor, indices);
                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = indices.shape.clone();
        let out = tensor.client.create_tensor_empty(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::NumericOpsFloat(
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

    fn scatter<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        struct ScatterOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for ScatterOps<D> {
            type Args = ScatterOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&args.tensor);
                let indices = handles.get_int_tensor(&args.indices);
                let value = handles.get_float_tensor(&args.value);

                let output = B::scatter(args.dim, tensor, indices, value);

                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.create_tensor_empty(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::NumericOpsFloat(
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

    fn select<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> FloatTensor<Self, D> {
        struct SelectOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for SelectOps<D> {
            type Args = SelectOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&args.tensor);
                let indices = handles.get_int_tensor(&args.indices);

                let output = B::select(tensor, args.dim, indices);

                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let mut shape: Vec<usize> = tensor.shape.clone();
        shape[dim] = indices.shape[0];
        let out = tensor.client.create_tensor_empty(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::NumericOpsFloat(
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

    fn select_assign<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        struct SelectAssignOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for SelectAssignOps<D> {
            type Args = SelectAssignOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&args.tensor);
                let indices = handles.get_int_tensor(&args.indices);
                let value = handles.get_float_tensor(&args.value);

                let output = B::select_assign(tensor, args.dim, indices, value);

                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.create_tensor_empty(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::NumericOpsFloat(
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

    fn slice<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> FloatTensor<Self, D1> {
        struct SliceOps<const D1: usize, const D2: usize>;

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for SliceOps<D1, D2> {
            type Args = SliceOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D1>(&args.tensor);

                let output = B::slice::<D1, D2>(tensor, args.ranges.clone().try_into().unwrap());

                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..D1 {
            shape.push(tensor.shape[i]);
        }

        let out = tensor.client.create_tensor_empty(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::BaseOpsFloat(
                BaseOpsDescription::Slice(
                    SliceOpsDescription {
                        tensor: tensor.into_description(),
                        ranges: ranges.into(),
                        out: out.to_description_out(),
                    },
                    Box::new(SliceOps::<D1, D2>),
                ),
            ));

        out
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        struct SliceAssignOps<const D1: usize, const D2: usize>;

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for SliceAssignOps<D1, D2> {
            type Args = SliceAssignOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D1>(&args.tensor);
                let value = handles.get_float_tensor::<D1>(&args.value);

                let output = B::slice_assign::<D1, D2>(
                    tensor,
                    args.ranges.clone().try_into().unwrap(),
                    value,
                );

                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.create_tensor_empty(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::BaseOpsFloat(
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

    fn mask_where<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        struct MaskWhereOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for MaskWhereOps<D> {
            type Args = MaskWhereOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&args.tensor);
                let value = handles.get_float_tensor(&args.value);
                let mask = handles.get_bool_tensor(&args.mask);

                let output = B::mask_where(tensor, mask, value);

                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.create_tensor_empty(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::NumericOpsFloat(
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

    fn mask_fill<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        struct MaskFillOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for MaskFillOps<D> {
            type Args = MaskFillOpsDescription<FloatElem<B>>;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&args.tensor);
                let mask = handles.get_bool_tensor(&args.mask);

                let output = B::mask_fill(tensor, mask, args.value);

                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.create_tensor_empty(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::NumericOpsFloat(
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

    fn equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(EqualOps, B::equal);

        let out = lhs
            .client
            .create_tensor_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client.register(TensorOpsDescription::BaseOpsFloat(
            BaseOpsDescription::Equal(
                BinaryOpsDescription {
                    lhs: lhs.into_description(),
                    rhs: rhs.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(EqualOps::<D>),
            ),
        ));

        out
    }

    fn equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(EqualElemOps, B::equal_elem);

        let out = lhs.client.create_tensor_empty(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn greater<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(GreaterOps, B::greater);

        let out = lhs
            .client
            .create_tensor_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn greater_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(GreaterElemOps, B::greater_elem);

        let out = lhs.client.create_tensor_empty(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn greater_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(GreaterEqualOps, B::greater_equal);

        let out = lhs
            .client
            .create_tensor_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn greater_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(GreaterEqualElemOps, B::greater_equal_elem);

        let out = lhs.client.create_tensor_empty(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn lower<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(LowerOps, B::lower);

        let out = lhs
            .client
            .create_tensor_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn lower_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(LowerElemOps, B::lower_elem);

        let out = lhs.client.create_tensor_empty(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn lower_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(LowerEqualOps, B::lower_equal);

        let out = lhs
            .client
            .create_tensor_empty(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn lower_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(LowerEqualElemOps, B::lower_equal_elem);

        let out = lhs.client.create_tensor_empty(lhs.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn sum<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(SumOps, B::sum);

        let out = tensor.client.create_tensor_empty(vec![1]);

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn sum_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        scalar_float_ops!(SumDimOps, B::sum_dim, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.create_tensor_empty(shape);

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn mean<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(MeanOps, B::mean);

        let out = tensor.client.create_tensor_empty(vec![1]);

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn mean_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        scalar_float_ops!(MeanDimOps, B::mean_dim, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.create_tensor_empty(shape);

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn to_full_precision<const D: usize>(
        tensor: &FloatTensor<Self, D>,
    ) -> FloatTensor<FullPrecisionBackend<Self>, D> {
        tensor.clone()
    }

    fn from_full_precision<const D: usize>(
        tensor: FloatTensor<FullPrecisionBackend<Self>, D>,
    ) -> FloatTensor<Self, D> {
        tensor
    }

    fn exp<const D: usize>(lhs: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(ExpOps, B::exp);

        let out = lhs.client.create_tensor_empty(lhs.shape.clone());

        out.client
            .register(TensorOpsDescription::FloatOps(FloatOpsDescription::Exp(
                UnaryOpsDescription {
                    input: lhs.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(ExpOps::<D>),
            )));

        out
    }

    fn log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(LogOps, B::log);

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());

        out.client
            .register(TensorOpsDescription::FloatOps(FloatOpsDescription::Log(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(LogOps::<D>),
            )));

        out
    }

    fn log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(Log1pOps, B::log1p);

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());

        out.client
            .register(TensorOpsDescription::FloatOps(FloatOpsDescription::Log1p(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(Log1pOps::<D>),
            )));

        out
    }

    fn powf<const D: usize>(lhs: FloatTensor<Self, D>, rhs: f32) -> FloatTensor<Self, D> {
        scalar_float_ops!(PowfOps, B::powf, f32);

        let out = lhs.client.create_tensor_empty(lhs.shape.clone());

        out.client
            .register(TensorOpsDescription::FloatOps(FloatOpsDescription::Powf(
                ScalarOpsDescription {
                    lhs: lhs.into_description(),
                    rhs,
                    out: out.to_description_out(),
                },
                Box::new(PowfOps::<D>),
            )));

        out
    }

    fn sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(SqrtOps, B::sqrt);

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());

        out.client
            .register(TensorOpsDescription::FloatOps(FloatOpsDescription::Sqrt(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(SqrtOps::<D>),
            )));

        out
    }

    fn abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(AbsOps, B::abs);

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(CosOps, B::cos);

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());

        out.client
            .register(TensorOpsDescription::FloatOps(FloatOpsDescription::Cos(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(CosOps::<D>),
            )));

        out
    }

    fn sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(SinOps, B::sin);

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());

        out.client
            .register(TensorOpsDescription::FloatOps(FloatOpsDescription::Sin(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(SinOps::<D>),
            )));

        out
    }

    fn tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(TanhOps, B::tanh);

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());

        out.client
            .register(TensorOpsDescription::FloatOps(FloatOpsDescription::Tanh(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(TanhOps::<D>),
            )));

        out
    }

    fn recip<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(Recip, B::recip);

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());
        out.client
            .register(TensorOpsDescription::FloatOps(FloatOpsDescription::Recip(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(Recip::<D>),
            )));
        out
    }

    fn erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(TanhOps, B::erf);

        let out = tensor.client.create_tensor_empty(tensor.shape.clone());

        out.client
            .register(TensorOpsDescription::FloatOps(FloatOpsDescription::Erf(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(TanhOps::<D>),
            )));

        out
    }

    fn cat<const D: usize>(tensors: Vec<FloatTensor<Self, D>>, dim: usize) -> FloatTensor<Self, D> {
        struct CatOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for CatOps<D> {
            type Args = CatOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensors = args
                    .tensors
                    .iter()
                    .map(|tensor| handles.get_float_tensor(tensor))
                    .collect();

                let output = B::cat::<D>(tensors, args.dim);

                handles.register_float_tensor(&args.out.id, output);
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

        let out = client.create_tensor_empty(shape);

        client.register(TensorOpsDescription::BaseOpsFloat(BaseOpsDescription::Cat(
            CatOpsDescription {
                tensors: tensors.into_iter().map(|t| t.into_description()).collect(),
                dim,
                out: out.to_description_out(),
            },
            Box::new(CatOps::<D>),
        )));

        out
    }

    fn argmax<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_float2int_ops!(ArgMaxOps, B::argmax, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.create_tensor_empty(shape);

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn argmin<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_float2int_ops!(ArgMinOps, B::argmin, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.create_tensor_empty(shape);

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn max<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(MaxOps, B::max);

        let out = tensor.client.create_tensor_empty(vec![1]);

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn max_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        scalar_float_ops!(MaxDimOps, B::max_dim, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.create_tensor_empty(shape);

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn max_dim_with_indices<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> (FloatTensor<Self, D>, IntTensor<Self, D>) {
        struct MaxDimWithIndicesOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for MaxDimWithIndicesOps<D> {
            type Args = ReduceDimWithIndicesDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&args.tensor);
                let (output, indices) = B::max_dim_with_indices(tensor, args.dim);

                handles.register_float_tensor(&args.out.id, output);
                handles.register_int_tensor(&args.out_indices.id, indices);
            }
        }

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.create_tensor_empty(shape.clone());
        let out_indices = client.create_tensor_empty(shape);

        client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn min<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(MinOps, B::min);

        let out = tensor.client.create_tensor_empty(vec![1]);

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn min_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        scalar_float_ops!(MinDimOps, B::min_dim, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.create_tensor_empty(shape);

        out.client.register(TensorOpsDescription::NumericOpsFloat(
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

    fn min_dim_with_indices<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> (FloatTensor<Self, D>, IntTensor<Self, D>) {
        struct MinDimWithIndicesOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for MinDimWithIndicesOps<D> {
            type Args = ReduceDimWithIndicesDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&args.tensor);
                let (output, indices) = B::min_dim_with_indices(tensor, args.dim);

                handles.register_float_tensor(&args.out.id, output);
                handles.register_int_tensor(&args.out_indices.id, indices);
            }
        }

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.create_tensor_empty(shape.clone());
        let out_indices = client.create_tensor_empty(shape);

        client.register(TensorOpsDescription::NumericOpsFloat(
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
