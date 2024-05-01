use crate::{
    binary_float_cmp_ops, binary_float_ops,
    client::FusionClient,
    get_client,
    ops::binary::binary_ops_shape,
    scalar_float2int_ops, scalar_float_cmp_ops, scalar_float_ops,
    stream::{execution::Operation, StreamId},
    unary_float_ops, Fusion, FusionBackend,
};
use burn_tensor::{
    ops::{BoolTensor, FloatElem, FloatTensor, FloatTensorOps, IntTensor},
    repr::*,
    DType, Data, Device, Distribution, Element, ElementConversion, Reader, Shape,
};
use std::{marker::PhantomData, ops::Range};

impl<B: FusionBackend> FloatTensorOps<Self> for Fusion<B> {
    fn float_from_data<const D: usize>(
        data: Data<FloatElem<Self>, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        let client = get_client::<B>(&device.clone());
        let tensor = B::float_from_data(data, device);
        let shape = B::float_shape(&tensor);

        client.register_tensor(
            B::float_tensor_handle(tensor),
            shape.dims.into(),
            StreamId::current(),
            B::FloatElem::dtype(),
        )
    }

    fn float_random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct RandomOps<B: FusionBackend, const D: usize> {
            desc: RandomOperationDescription,
            device: Device<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for RandomOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.desc.out.shape.clone());
                let output: B::FloatTensorPrimitive<D> =
                    B::float_random(shape, self.desc.distribution, &self.device);
                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = RandomOperationDescription {
            out: out.to_description_out(),
            distribution,
        };
        client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::Random(desc.clone())),
            RandomOps::<B, D>::new(desc, device.clone()),
        );

        out
    }

    fn float_zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct ZerosOps<B: FusionBackend, const D: usize> {
            out: TensorDescription,
            device: Device<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for ZerosOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.out.shape.clone());
                let output = B::float_zeros::<D>(shape, &self.device);
                handles.register_float_tensor::<B, D>(&self.out.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = out.to_description_out();
        client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Zeros(desc.clone())),
            ZerosOps::<B, D>::new(desc, device.clone()),
        );

        out
    }

    fn float_ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct OnesOps<B: FusionBackend, const D: usize> {
            out: TensorDescription,
            device: Device<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for OnesOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.out.shape.clone());
                let output = B::float_ones::<D>(shape, &self.device);
                handles.register_float_tensor::<B, D>(&self.out.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = out.to_description_out();
        client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Ones(desc.clone())),
            OnesOps::<B, D>::new(desc, device.clone()),
        );

        out
    }

    fn float_full<const D: usize>(
        shape: Shape<D>,
        fill_value: FloatElem<Self>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct FullOps<B: FusionBackend, const D: usize> {
            out: TensorDescription,
            elem: f32,
            device: Device<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for FullOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let shape = Shape::from(self.out.shape.clone());
                let output: B::FloatTensorPrimitive<D> =
                    B::float_full(shape, self.elem.elem(), &self.device);
                handles.register_float_tensor::<B, D>(&self.out.id, output);
            }
        }

        let stream = StreamId::current();
        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = (out.to_description_out(), fill_value.elem::<f32>());
        client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Full(desc.clone())),
            FullOps::<B, D>::new(desc.0, desc.1, device.clone()),
        );

        out
    }

    fn float_shape<const D: usize>(tensor: &FloatTensor<Self, D>) -> Shape<D> {
        tensor.shape()
    }

    fn float_into_data<const D: usize>(
        tensor: FloatTensor<Self, D>,
    ) -> Reader<Data<FloatElem<Self>, D>> {
        tensor.into_data::<B, D>()
    }

    fn float_device<const D: usize>(tensor: &FloatTensor<Self, D>) -> Device<Self> {
        tensor.client.device().clone()
    }

    fn float_to_device<const D: usize>(
        tensor: FloatTensor<Self, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        let device_original: &B::Device = tensor.client.device();
        let device_target: B::Device = device.clone();

        if device_original == &device_target {
            return tensor;
        }

        let id = tensor.stream;
        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.client.clone();

        client_original.clone().change_client_float::<B, D>(
            tensor.into_description(),
            client_target,
            id,
        )
    }

    fn float_into_int<const D: usize>(tensor: FloatTensor<Self, D>) -> IntTensor<Self, D> {
        #[derive(new)]
        struct IntoIntOps<B: FusionBackend, const D: usize> {
            desc: UnaryOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for IntoIntOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B, D>(&self.desc.input);
                let output = B::float_into_int(input);

                handles.register_int_tensor::<B, D>(&self.desc.out.id, output);
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
            OperationDescription::Float(FloatOperationDescription::IntoInt(desc.clone())),
            IntoIntOps::<B, D>::new(desc),
        );

        out
    }

    fn float_empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        let client = get_client::<B>(&device.clone());
        let stream = StreamId::current();
        let tensor = B::float_empty(shape.clone(), device);

        client.register_tensor(
            B::float_tensor_handle(tensor),
            shape.dims.into(),
            stream,
            B::FloatElem::dtype(),
        )
    }

    fn float_add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(AddOps, B::float_add);

        let stream_1 = lhs.stream;
        let stream_2 = rhs.stream;
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::FloatElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericFloat(NumericOperationDescription::Add(desc.clone())),
            AddOps::<B, D>::new(desc),
        );

        out
    }

    fn float_add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(AddOps, B::float_add_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::FloatElem::dtype());

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
            AddOps::<B, D>::new(desc),
        );

        out
    }

    fn float_clamp<const D: usize>(
        tensor: FloatTensor<Self, D>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct ClampOps<B: FusionBackend, const D: usize> {
            desc: ClampOperationDescription<f32>,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for ClampOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B, D>(&self.desc.tensor);
                let output = B::float_clamp(input, self.desc.min.elem(), self.desc.max.elem());

                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::FloatElem::dtype());

        let desc = ClampOperationDescription {
            tensor: tensor.into_description(),
            min: min.elem(),
            max: max.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Clamp(desc.clone())),
            ClampOps::<B, D>::new(desc),
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
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::FloatElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericFloat(NumericOperationDescription::Sub(desc.clone())),
            SubOps::<B, D>::new(desc),
        );

        out
    }

    fn float_sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(SubOps, B::float_sub_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::FloatElem::dtype());
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
            SubOps::<B, D>::new(desc),
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
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::FloatElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericFloat(NumericOperationDescription::Mul(desc.clone())),
            MulOps::<B, D>::new(desc),
        );

        out
    }

    fn float_mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(MulOps, B::float_mul_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::FloatElem::dtype());

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
            MulOps::<B, D>::new(desc),
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
        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::FloatElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericFloat(NumericOperationDescription::Div(desc.clone())),
            DivOps::<B, D>::new(desc),
        );

        out
    }

    fn float_div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(DivOps, B::float_div_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::FloatElem::dtype());

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
            DivOps::<B, D>::new(desc),
        );

        out
    }

    fn float_remainder_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(ModOps, B::float_remainder_scalar);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::FloatElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::RemScalar(
                desc.clone(),
            )),
            ModOps::<B, D>::new(desc),
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

        let out = lhs
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());
        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::Float(FloatOperationDescription::Matmul(desc.clone())),
            MatmulOps::<B, D>::new(desc),
        );

        out
    }

    fn float_swap_dims<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct SwapDimsOps<B: FusionBackend, const D: usize> {
            desc: SwapDimsDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for SwapDimsOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B, D>(&self.desc.input);
                let output = B::float_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];

        let mut out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = SwapDimsDescription {
            input: tensor.into_description(),
            dim1,
            dim2,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseFloat(BaseOperationDescription::SwapDims(desc.clone())),
            SwapDimsOps::<B, D>::new(desc),
        );
        out.stream = stream;

        out
    }

    fn float_reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        #[derive(new)]
        struct ReshapeDimsOps<B: FusionBackend, const D1: usize, const D2: usize> {
            desc: ReshapeDescription,
            _b: PhantomData<B>,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B::FusionRuntime>
            for ReshapeDimsOps<B, D1, D2>
        {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B, D1>(&self.desc.input);
                let output = B::float_reshape::<D1, D2>(input, Shape::from(&self.desc.out.shape));
                handles.register_float_tensor::<B, D2>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let shape: Vec<usize> = shape.dims.into();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = ReshapeDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseFloat(BaseOperationDescription::Reshape(desc.clone())),
            ReshapeDimsOps::<B, D1, D2>::new(desc),
        );

        out
    }

    fn float_gather<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct GatherOps<B: FusionBackend, const D: usize> {
            desc: GatherOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for GatherOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B, D>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B, D>(&self.desc.indices);

                let output = B::float_gather(self.desc.dim, tensor, indices);
                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let shape: Vec<usize> = indices.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = GatherOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericFloat(NumericOperationDescription::Gather(desc.clone())),
            GatherOps::<B, D>::new(desc),
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
        struct ScatterOps<B: FusionBackend, const D: usize> {
            desc: ScatterOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for ScatterOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B, D>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B, D>(&self.desc.indices);
                let value = handles.get_float_tensor::<B, D>(&self.desc.value);

                let output = B::float_scatter(self.desc.dim, tensor, indices, value);

                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

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
            ScatterOps::<B, D>::new(desc),
        );

        out
    }

    fn float_select<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct SelectOps<B: FusionBackend, const D: usize> {
            desc: SelectOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for SelectOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B, D>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B, 1>(&self.desc.indices);

                let output = B::float_select(tensor, self.desc.dim, indices);

                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let mut shape: Vec<usize> = tensor.shape.clone();
        shape[dim] = indices.shape[0];
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());
        let desc = SelectOperationDescription {
            tensor: tensor.into_description(),
            dim,
            indices: indices.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericFloat(NumericOperationDescription::Select(desc.clone())),
            SelectOps::<B, D>::new(desc),
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
        struct SelectAssignOps<B: FusionBackend, const D: usize> {
            desc: SelectAssignOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for SelectAssignOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B, D>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B, 1>(&self.desc.indices);
                let value = handles.get_float_tensor::<B, D>(&self.desc.value);

                let output = B::float_select_assign(tensor, self.desc.dim, indices, value);

                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = indices.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

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
            SelectAssignOps::<B, D>::new(desc),
        );

        out
    }

    fn float_slice<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> FloatTensor<Self, D1> {
        #[derive(new)]
        struct SliceOps<B: FusionBackend, const D1: usize, const D2: usize> {
            desc: SliceOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B::FusionRuntime>
            for SliceOps<B, D1, D2>
        {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B, D1>(&self.desc.tensor);

                let output =
                    B::float_slice::<D1, D2>(tensor, self.desc.ranges.clone().try_into().unwrap());

                handles.register_float_tensor::<B, D1>(&self.desc.out.id, output);
            }
        }
        let stream = tensor.stream;
        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..D1 {
            shape.push(tensor.shape[i]);
        }

        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = SliceOperationDescription {
            tensor: tensor.into_description(),
            ranges: ranges.into(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseFloat(BaseOperationDescription::Slice(desc.clone())),
            SliceOps::<B, D1, D2>::new(desc),
        );

        out
    }

    fn float_slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        #[derive(new)]
        struct SliceAssignOps<B: FusionBackend, const D1: usize, const D2: usize> {
            desc: SliceAssignOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Operation<B::FusionRuntime>
            for SliceAssignOps<B, D1, D2>
        {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B, D1>(&self.desc.tensor);
                let value = handles.get_float_tensor::<B, D1>(&self.desc.value);

                let output = B::float_slice_assign::<D1, D2>(
                    tensor,
                    self.desc.ranges.clone().try_into().unwrap(),
                    value,
                );

                handles.register_float_tensor::<B, D1>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = SliceAssignOperationDescription {
            tensor: tensor.into_description(),
            ranges: ranges.into(),
            value: value.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::BaseFloat(BaseOperationDescription::SliceAssign(desc.clone())),
            SliceAssignOps::<B, D1, D2>::new(desc),
        );

        out
    }

    fn float_mask_where<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct MaskWhereOps<B: FusionBackend, const D: usize> {
            desc: MaskWhereOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for MaskWhereOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B, D>(&self.desc.tensor);
                let value = handles.get_float_tensor::<B, D>(&self.desc.value);
                let mask = handles.get_bool_tensor::<B, D>(&self.desc.mask);

                let output = B::float_mask_where(tensor, mask, value);

                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = mask.stream;
        let stream_3 = value.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

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
            MaskWhereOps::<B, D>::new(desc),
        );

        out
    }

    fn float_mask_fill<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct MaskFillOps<B: FusionBackend, const D: usize> {
            desc: MaskFillOperationDescription<f32>,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for MaskFillOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B, D>(&self.desc.tensor);
                let mask = handles.get_bool_tensor::<B, D>(&self.desc.mask);

                let output = B::float_mask_fill(tensor, mask, self.desc.value.elem());

                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream_1 = tensor.stream;
        let stream_2 = mask.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());
        let desc = MaskFillOperationDescription {
            tensor: tensor.into_description(),
            value: value.elem(),
            mask: mask.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericFloat(NumericOperationDescription::MaskFill(desc.clone())),
            MaskFillOps::<B, D>::new(desc),
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
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::BaseFloat(BaseOperationDescription::Equal(desc.clone())),
            EqualOps::<B, D>::new(desc),
        );

        out
    }

    fn float_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(EqualElemOps, B::float_equal_elem);

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
            OperationDescription::NumericFloat(NumericOperationDescription::EqualElem(
                desc.clone(),
            )),
            EqualElemOps::<B, D>::new(desc),
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
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericFloat(NumericOperationDescription::Greater(desc.clone())),
            GreaterOps::<B, D>::new(desc),
        );

        out
    }

    fn float_greater_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(GreaterElemOps, B::float_greater_elem);

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
            OperationDescription::NumericFloat(NumericOperationDescription::GreaterElem(
                desc.clone(),
            )),
            GreaterElemOps::<B, D>::new(desc),
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
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

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
            GreaterEqualOps::<B, D>::new(desc),
        );

        out
    }

    fn float_greater_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(GreaterEqualElemOps, B::float_greater_equal_elem);

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
            OperationDescription::NumericFloat(NumericOperationDescription::GreaterEqualElem(
                desc.clone(),
            )),
            GreaterEqualElemOps::<B, D>::new(desc),
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
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericFloat(NumericOperationDescription::Lower(desc.clone())),
            LowerOps::<B, D>::new(desc),
        );

        out
    }

    fn float_lower_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(LowerElemOps, B::float_lower_elem);

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
            OperationDescription::NumericFloat(NumericOperationDescription::LowerElem(
                desc.clone(),
            )),
            LowerElemOps::<B, D>::new(desc),
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
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape), DType::Bool);

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
            LowerEqualOps::<B, D>::new(desc),
        );

        out
    }

    fn float_lower_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(LowerEqualElemOps, B::float_lower_equal_elem);

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
            OperationDescription::NumericFloat(NumericOperationDescription::LowerEqualElem(
                desc.clone(),
            )),
            LowerEqualElemOps::<B, D>::new(desc),
        );

        out
    }

    fn float_sum<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(SumOps, B::float_sum, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::FloatElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Sum(desc.clone())),
            SumOps::<B, D>::new(desc),
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
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::SumDim(desc.clone())),
            SumDimOps::<B, D>::new(desc),
        );

        out
    }

    fn float_mean<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(MeanOps, B::float_mean, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::FloatElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Mean(desc.clone())),
            MeanOps::<B, D>::new(desc),
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
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::MeanDim(desc.clone())),
            MeanDimOps::<B, D>::new(desc),
        );

        out
    }

    fn float_exp<const D: usize>(lhs: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(ExpOps, B::float_exp);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::FloatElem::dtype());

        let desc = UnaryOperationDescription {
            input: lhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::Exp(desc.clone())),
            ExpOps::<B, D>::new(desc),
        );

        out
    }

    fn float_log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(LogOps, B::float_log);

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
            OperationDescription::Float(FloatOperationDescription::Log(desc.clone())),
            LogOps::<B, D>::new(desc),
        );

        out
    }

    fn float_log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(Log1pOps, B::float_log1p);

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
            OperationDescription::Float(FloatOperationDescription::Log1p(desc.clone())),
            Log1pOps::<B, D>::new(desc),
        );

        out
    }

    fn float_powf_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: f32,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(PowfOps, B::float_powf_scalar, f32);

        let stream = lhs.stream;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::FloatElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: lhs.into_description(),
            rhs,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::Float(FloatOperationDescription::PowfScalar(desc.clone())),
            PowfOps::<B, D>::new(desc),
        );

        out
    }

    fn float_sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(SqrtOps, B::float_sqrt);

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
            OperationDescription::Float(FloatOperationDescription::Sqrt(desc.clone())),
            SqrtOps::<B, D>::new(desc),
        );

        out
    }

    fn float_abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(AbsOps, B::float_abs);

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
            OperationDescription::NumericFloat(NumericOperationDescription::Abs(desc.clone())),
            AbsOps::<B, D>::new(desc),
        );

        out
    }

    fn float_cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(CosOps, B::float_cos);

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
            OperationDescription::Float(FloatOperationDescription::Cos(desc.clone())),
            CosOps::<B, D>::new(desc),
        );

        out
    }

    fn float_sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(SinOps, B::float_sin);

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
            OperationDescription::Float(FloatOperationDescription::Sin(desc.clone())),
            SinOps::<B, D>::new(desc),
        );

        out
    }

    fn float_tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(TanhOps, B::float_tanh);

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
            OperationDescription::Float(FloatOperationDescription::Tanh(desc.clone())),
            TanhOps::<B, D>::new(desc),
        );

        out
    }

    fn float_recip<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(Recip, B::float_recip);

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
            OperationDescription::Float(FloatOperationDescription::Recip(desc.clone())),
            Recip::<B, D>::new(desc),
        );

        out
    }

    fn float_erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(TanhOps, B::float_erf);

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
            OperationDescription::Float(FloatOperationDescription::Erf(desc.clone())),
            TanhOps::<B, D>::new(desc),
        );

        out
    }

    fn float_cat<const D: usize>(
        tensors: Vec<FloatTensor<Self, D>>,
        dim: usize,
    ) -> FloatTensor<Self, D> {
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
                    .map(|tensor| handles.get_float_tensor::<B, D>(tensor))
                    .collect();

                let output = B::float_cat::<D>(tensors, self.desc.dim);

                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
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

        let out = client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = CatOperationDescription {
            tensors: tensors.into_iter().map(|t| t.into_description()).collect(),
            dim,
            out: out.to_description_out(),
        };
        client.register(
            streams,
            OperationDescription::BaseFloat(BaseOperationDescription::Cat(desc.clone())),
            CatOps::<B, D>::new(desc),
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
            OperationDescription::NumericFloat(NumericOperationDescription::ArgMax(desc.clone())),
            ArgMaxOps::<B, D>::new(desc),
        );

        out
    }

    fn float_repeat<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        times: usize,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct RepeatOps<B: FusionBackend, const D: usize> {
            desc: RepeatOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for RepeatOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B, D>(&self.desc.tensor);

                let output = B::float_repeat::<D>(tensor, self.desc.dim, self.desc.times);

                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] *= times;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = RepeatOperationDescription {
            tensor: tensor.into_description(),
            dim,
            times,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::BaseFloat(BaseOperationDescription::Repeat(desc.clone())),
            RepeatOps::<B, D>::new(desc),
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
            OperationDescription::NumericFloat(NumericOperationDescription::ArgMin(desc.clone())),
            ArgMinOps::<B, D>::new(desc),
        );

        out
    }

    fn float_max<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(MaxOps, B::float_max, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::FloatElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Max(desc.clone())),
            MaxOps::<B, D>::new(desc),
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
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::MaxDim(desc.clone())),
            MaxDimOps::<B, D>::new(desc),
        );

        out
    }

    fn float_max_dim_with_indices<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> (FloatTensor<Self, D>, IntTensor<Self, D>) {
        #[derive(new)]
        struct MaxDimWithIndicesOps<B: FusionBackend, const D: usize> {
            desc: ReduceDimWithIndicesDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for MaxDimWithIndicesOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B, D>(&self.desc.tensor);
                let (output, indices) = B::float_max_dim_with_indices(tensor, self.desc.dim);

                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
                handles.register_int_tensor::<B, D>(&self.desc.out_indices.id, indices);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone(), B::FloatElem::dtype());
        let out_indices = client.tensor_uninitialized(shape, B::IntElem::dtype());

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
            MaxDimWithIndicesOps::<B, D>::new(desc),
        );

        (out, out_indices)
    }

    fn float_min<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(MinOps, B::float_min, reduce);

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(vec![1], B::FloatElem::dtype());

        let desc = UnaryOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::Min(desc.clone())),
            MinOps::<B, D>::new(desc),
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
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = ScalarOperationDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream],
            OperationDescription::NumericFloat(NumericOperationDescription::MinDim(desc.clone())),
            MinDimOps::<B, D>::new(desc),
        );

        out
    }

    fn float_min_dim_with_indices<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> (FloatTensor<Self, D>, IntTensor<Self, D>) {
        #[derive(new)]
        struct MinDimWithIndicesOps<B: FusionBackend, const D: usize> {
            desc: ReduceDimWithIndicesDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for MinDimWithIndicesOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B, D>(&self.desc.tensor);
                let (output, indices) = B::float_min_dim_with_indices(tensor, self.desc.dim);

                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
                handles.register_int_tensor::<B, D>(&self.desc.out_indices.id, indices);
            }
        }

        let stream = tensor.stream;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone(), B::FloatElem::dtype());
        let out_indices = client.tensor_uninitialized(shape, B::IntElem::dtype());

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
            MinDimWithIndicesOps::<B, D>::new(desc),
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

        let out = lhs.client.tensor_uninitialized(
            binary_ops_shape(&lhs.shape, &rhs.shape),
            B::FloatElem::dtype(),
        );

        let desc = BinaryOperationDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            vec![stream_1, stream_2],
            OperationDescription::NumericFloat(NumericOperationDescription::Powf(desc.clone())),
            PowOps::<B, D>::new(desc),
        );

        out
    }

    fn float_permute<const D: usize>(
        tensor: FloatTensor<Self, D>,
        axes: [usize; D],
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct PermuteDimsOps<B: FusionBackend, const D: usize> {
            desc: PermuteOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for PermuteDimsOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B, D>(&self.desc.input);
                let axes: [usize; D] = self.desc.axes.try_into().unwrap();
                let output = B::float_permute(input, axes);
                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        // Change the shape of the tensor to match the new axes
        let shape = axes.into_iter().map(|x| tensor.shape[x]).collect();

        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

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

    fn float_expand<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        #[derive(new)]
        struct ExpandOps<B: FusionBackend, const D: usize, const D2: usize> {
            desc: ExpandOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, const D2: usize, B: FusionBackend> Operation<B::FusionRuntime>
            for ExpandOps<B, D, D2>
        {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B, D>(&self.desc.input);
                let shape: [usize; D2] = self.desc.shape.try_into().unwrap();
                let output = B::float_expand(input, shape.into());

                handles.register_float_tensor::<B, D2>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;

        let out = tensor
            .client
            .tensor_uninitialized(shape.dims.into(), B::FloatElem::dtype());

        let desc = ExpandOperationDescription {
            input: tensor.into_description(),
            shape: shape.dims.into(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseFloat(BaseOperationDescription::Expand(desc.clone())),
            ExpandOps::<B, D1, D2>::new(desc),
        );

        out
    }

    fn float_flip<const D: usize>(
        tensor: FloatTensor<Self, D>,
        axes: &[usize],
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct FlipOps<B: FusionBackend, const D: usize> {
            desc: FlipOperationDescription,
            _b: PhantomData<B>,
        }

        impl<const D: usize, B: FusionBackend> Operation<B::FusionRuntime> for FlipOps<B, D> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B, D>(&self.desc.input);
                let output = B::float_flip(input, &self.desc.axes);
                handles.register_float_tensor::<B, D>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::FloatElem::dtype());

        let desc = FlipOperationDescription {
            input: tensor.into_description(),
            axes: axes.to_vec(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::BaseInt(BaseOperationDescription::Flip(desc.clone())),
            FlipOps::<B, D>::new(desc),
        );

        out
    }
}
