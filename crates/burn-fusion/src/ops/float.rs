use crate::{
    binary_float_cmp_ops, binary_float_ops,
    client::FusionClient,
    get_client,
    ops::binary::binary_ops_shape,
    scalar_float2int_ops, scalar_float_cmp_ops, scalar_float_ops,
    stream::{
        BaseOperationDescription, BinaryOperationDescription, CatOperationDescription,
        ClampOperationDescription, ExpandOperationDescription, FlipOperationDescription,
        FloatOperationDescription, GatherOperationDescription, MaskFillOperationDescription,
        MaskWhereOperationDescription, NumericOperationDescription, Operation,
        OperationDescription, PermuteOperationDescription, RandomOperationDescription,
        ReduceDimWithIndicesDescription, ReshapeDescription, ScalarOperationDescription,
        ScatterOperationDescription, SelectAssignOperationDescription, SelectOperationDescription,
        SliceAssignOperationDescription, SliceOperationDescription, StreamId, SwapDimsDescription,
        UnaryOperationDescription,
    },
    unary_float_ops, Fusion, FusionBackend, TensorDescription,
};
use burn_tensor::{
    ops::{BoolTensor, FloatElem, FloatTensor, FloatTensorOps, FullPrecisionBackend, IntTensor},
    Data, Device, Distribution, ElementConversion, Reader, Shape,
};
use std::ops::Range;

impl<B: FusionBackend> FloatTensorOps<Self> for Fusion<B> {
    fn float_from_data<const D: usize>(
        data: Data<FloatElem<Self>, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        let client = get_client::<B>(&device.clone().into());
        let tensor = B::float_from_data(data, device);
        let shape = B::float_shape(&tensor);

        client.register_tensor(
            B::float_tensor_handle(tensor),
            shape.dims.into(),
            StreamId::current(),
        )
    }

    fn float_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct RandomOps<const D: usize> {
            desc: RandomOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for RandomOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.desc.out.shape.clone());
                let output: B::FloatTensorPrimitive<D> =
                    B::float_random(shape, self.desc.distribution, &handles.device);
                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::Float(FloatOperationDescription::Random(desc.clone())),
            RandomOps::<D>::new(desc),
        );

        out
    }

    fn float_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct ZerosOps<const D: usize> {
            out: TensorDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for ZerosOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.out.shape.clone());
                let output = B::float_zeros::<D>(shape, &handles.device);
                handles.register_float_tensor(&self.out.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);

        let desc = out.to_description_out();
        client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Zeros(desc.clone())),
            ZerosOps::<D>::new(desc),
        );

        out
    }

    fn float_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct OnesOps<const D: usize> {
            out: TensorDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for OnesOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.out.shape.clone());
                let output = B::float_ones::<D>(shape, &handles.device);
                handles.register_float_tensor(&self.out.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);

        let desc = out.to_description_out();
        client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Ones(desc.clone())),
            OnesOps::<D>::new(desc),
        );

        out
    }

    fn float_full<const D: usize>(
        shape: Shape<D>,
        fill_value: FloatElem<Self>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct FullOps<const D: usize> {
            out: TensorDescription,
            elem: f32,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for FullOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.out.shape.clone());
                let output: B::FloatTensorPrimitive<D> =
                    B::float_full(shape, self.elem.elem(), &handles.device);
                handles.register_float_tensor(&self.out.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);

        let desc = (out.to_description_out(), fill_value.elem::<f32>());
        client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Full(desc.clone())),
            FullOps::<D>::new(desc.0, desc.1),
        );

        out
    }

    fn float_shape<const D: usize>(tensor: &FloatTensor<Self, D>) -> Shape<D> {
        tensor.shape()
    }

    fn float_into_data<const D: usize>(
        tensor: FloatTensor<Self, D>,
    ) -> Reader<Data<FloatElem<Self>, D>> {
        tensor.into_data()
    }

    fn float_device<const D: usize>(tensor: &FloatTensor<Self, D>) -> Device<Self> {
        tensor.client.device().clone().into()
    }

    fn float_to_device<const D: usize>(
        tensor: FloatTensor<Self, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        let device_original: &B::FusionDevice = tensor.client.device();
        let device_target: B::FusionDevice = device.clone().into();

        if device_original == &device_target {
            return tensor;
        }

        let id = tensor.stream;
        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.client.clone();

        client_original.clone().change_client_float::<D>(
            tensor.into_description(),
            client_target,
            id,
        )
    }

    fn float_into_int<const D: usize>(tensor: FloatTensor<Self, D>) -> IntTensor<Self, D> {
        #[derive(new)]
        struct IntoIntOps<const D: usize> {
            desc: UnaryOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for IntoIntOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&self.desc.input);
                let output = B::float_into_int(input);

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
            OperationDescription::Float(FloatOperationDescription::IntoInt(desc.clone())),
            IntoIntOps::<D>::new(desc),
        );

        out
    }

    fn float_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        let client = get_client::<B>(&device.clone().into());
        let stream = StreamId::current();
        let tensor = B::float_empty(shape.clone(), device);

        client.register_tensor(B::float_tensor_handle(tensor), shape.dims.into(), stream)
    }

    fn float_add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(AddOps, B::float_add);

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
            OperationDescription::NumericFloat(NumericOperationDescription::Add(desc.clone())),
            AddOps::<D>::new(desc),
        );

        out
    }

    fn float_add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(AddOps, B::float_add_scalar);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem::<f32>(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::AddScalar(
                desc.clone(),
            )),
            AddOps::<D>::new(desc),
        );

        out
    }

    fn float_clamp<const D: usize>(
        tensor: FloatTensor<Self, D>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct ClampOps<const D: usize> {
            desc: ClampOperationDescription<f32>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for ClampOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&self.desc.tensor);
                let output = B::float_clamp(input, self.desc.min.elem(), self.desc.max.elem());

                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::NumericFloat(NumericOperationDescription::Clamp(desc.clone())),
            ClampOps::<D>::new(desc),
        );

        out
    }

    fn float_sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(SubOps, B::float_sub);

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
            OperationDescription::NumericFloat(NumericOperationDescription::Sub(desc.clone())),
            SubOps::<D>::new(desc),
        );

        out
    }

    fn float_sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(SubOps, B::float_sub_scalar);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());
        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::SubScalar(
                desc.clone(),
            )),
            SubOps::<D>::new(desc),
        );

        out
    }

    fn float_mul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(MulOps, B::float_mul);

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
            OperationDescription::NumericFloat(NumericOperationDescription::Mul(desc.clone())),
            MulOps::<D>::new(desc),
        );

        out
    }

    fn float_mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(MulOps, B::float_mul_scalar);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::MulScalar(
                desc.clone(),
            )),
            MulOps::<D>::new(desc),
        );

        out
    }

    fn float_div<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(DivOps, B::float_div);

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
            OperationDescription::NumericFloat(NumericOperationDescription::Div(desc.clone())),
            DivOps::<D>::new(desc),
        );

        out
    }

    fn float_div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(DivOps, B::float_div_scalar);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::DivScalar(
                desc.clone(),
            )),
            DivOps::<D>::new(desc),
        );

        out
    }

    fn float_matmul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(MatmulOps, B::float_matmul);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let mut shape = binary_ops_shape(&lhs.shape, &rhs.shape);

        shape[D - 2] = lhs.shape[D - 2];
        shape[D - 1] = rhs.shape[D - 1];

        let out = lhs.client.tensor_uninitialized(shape);
        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::Float(FloatOperationDescription::Matmul(desc.clone())),
            MatmulOps::<D>::new(desc),
        );

        out
    }

    fn float_swap_dims<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct SwapDimsOps<const D: usize> {
            desc: SwapDimsDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for SwapDimsOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&self.desc.input);
                let output = B::float_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_float_tensor(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];

        let mut out = tensor.client.tensor_uninitialized(shape);

        let desc = SwapDimsDescription {
            input: tensor.into_description(),
            dim1,
            dim2,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseFloat(BaseOperationDescription::SwapDims(desc.clone())),
            SwapDimsOps::<D>::new(desc),
        );
        out.stream = stream;

        out
    }

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        #[derive(new)]
        struct ReshapeDimsOps<const D1: usize, const D2: usize> {
            desc: ReshapeDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B> for ReshapeDimsOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D1>(&self.desc.input);
                let output = B::float_reshape::<D1, D2>(input, Shape::from(&self.desc.out.shape));
                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::BaseFloat(BaseOperationDescription::Reshape(desc.clone())),
            ReshapeDimsOps::<D1, D2>::new(desc),
        );

        out
    }

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct GatherOps<const D: usize> {
            desc: GatherOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for GatherOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);

                let output = B::float_gather(self.desc.dim, tensor, indices);
                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::NumericFloat(NumericOperationDescription::Gather(desc.clone())),
            GatherOps::<D>::new(desc),
        );

        out
    }

    fn float_scatter<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct ScatterOps<const D: usize> {
            desc: ScatterOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for ScatterOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);
                let value = handles.get_float_tensor(&self.desc.value);

                let output = B::float_scatter(self.desc.dim, tensor, indices, value);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::NumericFloat(NumericOperationDescription::Scatter(desc.clone())),
            ScatterOps::<D>::new(desc),
        );

        out
    }

    fn float_select<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct SelectOps<const D: usize> {
            desc: SelectOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for SelectOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);

                let output = B::float_select(tensor, self.desc.dim, indices);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::NumericFloat(NumericOperationDescription::Select(desc.clone())),
            SelectOps::<D>::new(desc),
        );

        out
    }

    fn float_select_assign<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct SelectAssignOps<const D: usize> {
            desc: SelectAssignOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for SelectAssignOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);
                let value = handles.get_float_tensor(&self.desc.value);

                let output = B::float_select_assign(tensor, self.desc.dim, indices, value);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::NumericFloat(NumericOperationDescription::SelectAssign(
                desc.clone(),
            )),
            SelectAssignOps::<D>::new(desc),
        );

        out
    }

    fn float_slice<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> FloatTensor<Self, D1> {
        #[derive(new)]
        struct SliceOps<const D1: usize, const D2: usize> {
            desc: SliceOperationDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B> for SliceOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D1>(&self.desc.tensor);

                let output =
                    B::float_slice::<D1, D2>(tensor, self.desc.ranges.clone().try_into().unwrap());

                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::BaseFloat(BaseOperationDescription::Slice(desc.clone())),
            SliceOps::<D1, D2>::new(desc),
        );

        out
    }

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        #[derive(new)]
        struct SliceAssignOps<const D1: usize, const D2: usize> {
            desc: SliceAssignOperationDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B> for SliceAssignOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D1>(&self.desc.tensor);
                let value = handles.get_float_tensor::<D1>(&self.desc.value);

                let output = B::float_slice_assign::<D1, D2>(
                    tensor,
                    self.desc.ranges.clone().try_into().unwrap(),
                    value,
                );

                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::BaseFloat(BaseOperationDescription::SliceAssign(desc.clone())),
            SliceAssignOps::<D1, D2>::new(desc),
        );

        out
    }

    fn float_mask_where<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct MaskWhereOps<const D: usize> {
            desc: MaskWhereOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for MaskWhereOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let value = handles.get_float_tensor(&self.desc.value);
                let mask = handles.get_bool_tensor(&self.desc.mask);

                let output = B::float_mask_where(tensor, mask, value);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::NumericFloat(NumericOperationDescription::MaskWhere(
                desc.clone(),
            )),
            MaskWhereOps::<D>::new(desc),
        );

        out
    }

    fn float_mask_fill<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct MaskFillOps<const D: usize> {
            desc: MaskFillOperationDescription<f32>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for MaskFillOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let mask = handles.get_bool_tensor(&self.desc.mask);

                let output = B::float_mask_fill(tensor, mask, self.desc.value.elem());

                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::NumericFloat(NumericOperationDescription::MaskFill(desc.clone())),
            MaskFillOps::<D>::new(desc),
        );

        out
    }

    fn float_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(EqualOps, B::float_equal);

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
            OperationDescription::BaseFloat(BaseOperationDescription::Equal(desc.clone())),
            EqualOps::<D>::new(desc),
        );

        out
    }

    fn float_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(EqualElemOps, B::float_equal_elem);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::EqualElem(
                desc.clone(),
            )),
            EqualElemOps::<D>::new(desc),
        );

        out
    }

    fn float_greater<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(GreaterOps, B::float_greater);

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
            OperationDescription::NumericFloat(NumericOperationDescription::Greater(desc.clone())),
            GreaterOps::<D>::new(desc),
        );

        out
    }

    fn float_greater_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(GreaterElemOps, B::float_greater_elem);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::GreaterElem(
                desc.clone(),
            )),
            GreaterElemOps::<D>::new(desc),
        );

        out
    }

    fn float_greater_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(GreaterEqualOps, B::float_greater_equal);

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
            OperationDescription::NumericFloat(NumericOperationDescription::GreaterEqual(
                desc.clone(),
            )),
            GreaterEqualOps::<D>::new(desc),
        );

        out
    }

    fn float_greater_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(GreaterEqualElemOps, B::float_greater_equal_elem);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::GreaterEqualElem(
                desc.clone(),
            )),
            GreaterEqualElemOps::<D>::new(desc),
        );

        out
    }

    fn float_lower<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(LowerOps, B::float_lower);

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
            OperationDescription::NumericFloat(NumericOperationDescription::Lower(desc.clone())),
            LowerOps::<D>::new(desc),
        );

        out
    }

    fn float_lower_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(LowerElemOps, B::float_lower_elem);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::LowerElem(
                desc.clone(),
            )),
            LowerElemOps::<D>::new(desc),
        );

        out
    }

    fn float_lower_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(LowerEqualOps, B::float_lower_equal);

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
            OperationDescription::NumericFloat(NumericOperationDescription::LowerEqual(
                desc.clone(),
            )),
            LowerEqualOps::<D>::new(desc),
        );

        out
    }

    fn float_lower_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(LowerEqualElemOps, B::float_lower_equal_elem);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::LowerEqualElem(
                desc.clone(),
            )),
            LowerEqualElemOps::<D>::new(desc),
        );

        out
    }

    fn float_sum<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(SumOps, B::float_sum);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Sum(desc.clone())),
            SumOps::<D>::new(desc),
        );

        out
    }

    fn float_sum_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(SumDimOps, B::float_sum_dim, usize, noconvert);

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
            OperationDescription::NumericFloat(NumericOperationDescription::SumDim(desc.clone())),
            SumDimOps::<D>::new(desc),
        );

        out
    }

    fn float_mean<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(MeanOps, B::float_mean);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Mean(desc.clone())),
            MeanOps::<D>::new(desc),
        );

        out
    }

    fn float_mean_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(MeanDimOps, B::float_mean_dim, usize, noconvert);

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
            OperationDescription::NumericFloat(NumericOperationDescription::MeanDim(desc.clone())),
            MeanDimOps::<D>::new(desc),
        );

        out
    }

    fn float_to_full_precision<const D: usize>(
        tensor: &FloatTensor<Self, D>,
    ) -> FloatTensor<FullPrecisionBackend<Self>, D> {
        tensor.clone()
    }

    fn float_from_full_precision<const D: usize>(
        tensor: FloatTensor<FullPrecisionBackend<Self>, D>,
    ) -> FloatTensor<Self, D> {
        tensor
    }

    fn float_exp<const D: usize>(lhs: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(ExpOps, B::float_exp);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = UnaryOperationDescription {
            input: lhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::Exp(desc.clone())),
            ExpOps::<D>::new(desc),
        );

        out
    }

    fn float_log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(LogOps, B::float_log);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::Log(desc.clone())),
            LogOps::<D>::new(desc),
        );

        out
    }

    fn float_log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(Log1pOps, B::float_log1p);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::Log1p(desc.clone())),
            Log1pOps::<D>::new(desc),
        );

        out
    }

    fn float_powf_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: f32,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(PowfOps, B::float_powf_scalar, f32);

        let stream = lhs.stream;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::PowfScalar(desc.clone())),
            PowfOps::<D>::new(desc),
        );

        out
    }

    fn float_sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(SqrtOps, B::float_sqrt);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::Sqrt(desc.clone())),
            SqrtOps::<D>::new(desc),
        );

        out
    }

    fn float_abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(AbsOps, B::float_abs);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Abs(desc.clone())),
            AbsOps::<D>::new(desc),
        );

        out
    }

    fn float_cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(CosOps, B::float_cos);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::Cos(desc.clone())),
            CosOps::<D>::new(desc),
        );

        out
    }

    fn float_sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(SinOps, B::float_sin);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::Sin(desc.clone())),
            SinOps::<D>::new(desc),
        );

        out
    }

    fn float_tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(TanhOps, B::float_tanh);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::Tanh(desc.clone())),
            TanhOps::<D>::new(desc),
        );

        out
    }

    fn float_recip<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(Recip, B::float_recip);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());
        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::Recip(desc.clone())),
            Recip::<D>::new(desc),
        );

        out
    }

    fn float_erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(TanhOps, B::float_erf);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::Erf(desc.clone())),
            TanhOps::<D>::new(desc),
        );

        out
    }

    fn float_cat<const D: usize>(
        tensors: Vec<FloatTensor<Self, D>>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
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
                    .map(|tensor| handles.get_float_tensor(tensor))
                    .collect();

                let output = B::float_cat::<D>(tensors, self.desc.dim);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::BaseFloat(BaseOperationDescription::Cat(desc.clone())),
            CatOps::<D>::new(desc),
        );

        out
    }

    fn float_argmax<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        scalar_float2int_ops!(ArgMaxOps, B::float_argmax, usize);

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
            OperationDescription::NumericFloat(NumericOperationDescription::ArgMax(desc.clone())),
            ArgMaxOps::<D>::new(desc),
        );

        out
    }

    fn float_argmin<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        scalar_float2int_ops!(ArgMinOps, B::float_argmin, usize);

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
            OperationDescription::NumericFloat(NumericOperationDescription::ArgMin(desc.clone())),
            ArgMinOps::<D>::new(desc),
        );

        out
    }

    fn float_max<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(MaxOps, B::float_max);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Max(desc.clone())),
            MaxOps::<D>::new(desc),
        );

        out
    }

    fn float_max_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(MaxDimOps, B::float_max_dim, usize, noconvert);

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
            OperationDescription::NumericFloat(NumericOperationDescription::MaxDim(desc.clone())),
            MaxDimOps::<D>::new(desc),
        );

        out
    }

    fn float_max_dim_with_indices<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> (FloatTensor<Self, D>, IntTensor<Self, D>) {
        #[derive(new)]
        struct MaxDimWithIndicesOps<const D: usize> {
            desc: ReduceDimWithIndicesDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for MaxDimWithIndicesOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let (output, indices) = B::float_max_dim_with_indices(tensor, self.desc.dim);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::NumericFloat(NumericOperationDescription::MaxDimWithIndices(
                desc.clone(),
            )),
            MaxDimWithIndicesOps::<D>::new(desc),
        );

        (out, out_indices)
    }

    fn float_min<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(MinOps, B::float_min);

        let stream = tensor.stream;
        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Min(desc.clone())),
            MinOps::<D>::new(desc),
        );

        out
    }

    fn float_min_dim<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(MinDimOps, B::float_min_dim, usize, noconvert);

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
            OperationDescription::NumericFloat(NumericOperationDescription::MinDim(desc.clone())),
            MinDimOps::<D>::new(desc),
        );

        out
    }

    fn float_min_dim_with_indices<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> (FloatTensor<Self, D>, IntTensor<Self, D>) {
        #[derive(new)]
        struct MinDimWithIndicesOps<const D: usize> {
            desc: ReduceDimWithIndicesDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for MinDimWithIndicesOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let (output, indices) = B::float_min_dim_with_indices(tensor, self.desc.dim);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::NumericFloat(NumericOperationDescription::MinDimWithIndices(
                desc.clone(),
            )),
            MinDimWithIndicesOps::<D>::new(desc),
        );

        (out, out_indices)
    }

    fn float_powf<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(PowOps, B::float_powf);
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
            OperationDescription::NumericFloat(NumericOperationDescription::Powf(desc.clone())),
            PowOps::<D>::new(desc),
        );

        out
    }

    fn float_permute<const D: usize>(
        tensor: FloatTensor<Self, D>,
        axes: [usize; D],
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct PermuteDimsOps<const D: usize> {
            desc: PermuteOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for PermuteDimsOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&self.desc.input);
                let axes: [usize; D] = self.desc.axes.try_into().unwrap();
                let output = B::float_permute(input, axes);
                handles.register_float_tensor(&self.desc.out.id, output);
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

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        #[derive(new)]
        struct ExpandOps<const D: usize, const D2: usize> {
            desc: ExpandOperationDescription,
        }

        impl<const D: usize, const D2: usize, B: FusionBackend> Operation<B> for ExpandOps<D, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&self.desc.input);
                let shape: [usize; D2] = self.desc.shape.try_into().unwrap();
                let output = B::float_expand(input, shape.into());

                handles.register_float_tensor(&self.desc.out.id, output);
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
            OperationDescription::BaseFloat(BaseOperationDescription::Expand(desc.clone())),
            ExpandOps::<D1, D2>::new(desc),
        );

        out
    }

    fn float_flip<const D: usize>(
        tensor: FloatTensor<Self, D>,
        axes: &[usize],
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct FlipOps<const D: usize> {
            desc: FlipOperationDescription,
        }

        impl<const D: usize, B: FusionBackend> Operation<B> for FlipOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&self.desc.input);
                let output = B::float_flip(input, &self.desc.axes);
                handles.register_float_tensor(&self.desc.out.id, output);
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
            FlipOps::<D>::new(desc),
        );

        out
    }
}
