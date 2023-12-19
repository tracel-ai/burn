use crate::{
    binary_float_cmp_ops, binary_float_ops,
    client::FusionClient,
    get_client,
    graph::{
        BaseOpsDescription, BinaryOpsDescription, CatOpsDescription, ClampOpsDescription,
        FloatOpsDescription, GatherOpsDescription, MaskFillOpsDescription, MaskWhereOpsDescription,
        NumericOpsDescription, Ops, RandomOpsDescription, ReduceDimWithIndicesDescription,
        ReshapeDescription, ScalarOpsDescription, ScatterOpsDescription,
        SelectAssignOpsDescription, SelectOpsDescription, SliceAssignOpsDescription,
        SliceOpsDescription, SwapDimsDescription, TensorOpsDescription, UnaryOpsDescription,
    },
    ops::binary::binary_ops_shape,
    scalar_float2int_ops, scalar_float_cmp_ops, scalar_float_ops, unary_float_ops, Fusion,
    FusionBackend, TensorDescription,
};
use burn_tensor::{
    ops::{BoolTensor, FloatElem, FloatTensor, FullPrecisionBackend, IntTensor, TensorOps},
    Data, Device, Distribution, ElementConversion, Reader, Shape,
};
use std::ops::Range;

impl<B: FusionBackend> TensorOps<Self> for Fusion<B> {
    fn from_data<const D: usize>(
        data: Data<FloatElem<Self>, D>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        let client = get_client::<B>(&device.clone().into());
        let tensor = B::from_data(data, device);
        let shape = B::shape(&tensor);

        client.register_tensor(B::float_tensor_handle(tensor), shape.dims.into())
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct RandomOps<const D: usize> {
            desc: RandomOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for RandomOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.desc.out.shape.clone());
                let output: B::TensorPrimitive<D> =
                    B::random(shape, self.desc.distribution, &handles.device);
                handles.register_float_tensor(&self.desc.out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);

        let desc = RandomOpsDescription {
            out: out.to_description_out(),
            distribution,
        };
        client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::Random(desc.clone())),
            RandomOps::<D>::new(desc),
        );

        out
    }

    fn zeros<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct ZerosOps<const D: usize> {
            out: TensorDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for ZerosOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.out.shape.clone());
                let output = B::zeros::<D>(shape, &handles.device);
                handles.register_float_tensor(&self.out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);

        let desc = out.to_description_out();
        client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Zeros(desc.clone())),
            ZerosOps::<D>::new(desc),
        );

        out
    }

    fn ones<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct OnesOps<const D: usize> {
            out: TensorDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for OnesOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.out.shape.clone());
                let output = B::ones::<D>(shape, &handles.device);
                handles.register_float_tensor(&self.out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);

        let desc = out.to_description_out();
        client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Ones(desc.clone())),
            OnesOps::<D>::new(desc),
        );

        out
    }

    fn full<const D: usize>(
        shape: Shape<D>,
        fill_value: FloatElem<Self>,
        device: &Device<Self>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct FullOps<const D: usize> {
            out: TensorDescription,
            elem: f32,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for FullOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let shape = Shape::from(self.out.shape.clone());
                let output: B::TensorPrimitive<D> =
                    B::full(shape, self.elem.elem(), &handles.device);
                handles.register_float_tensor(&self.out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let client = get_client::<B>(&device.clone().into());
        let out = client.tensor_uninitialized(shape);

        let desc = (out.to_description_out(), fill_value.elem::<f32>());
        client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Full(desc.clone())),
            FullOps::<D>::new(desc.0, desc.1),
        );

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
        #[derive(new)]
        struct IntoIntOps<const D: usize> {
            desc: UnaryOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for IntoIntOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&self.desc.input);
                let output = B::into_int(input);

                handles.register_int_tensor(&self.desc.out.id, output);
            }
        }

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::IntoInt(desc.clone())),
            IntoIntOps::<D>::new(desc),
        );

        out
    }

    fn empty<const D: usize>(shape: Shape<D>, device: &Device<Self>) -> FloatTensor<Self, D> {
        let client = get_client::<B>(&device.clone().into());
        let tensor = B::empty(shape.clone(), device);

        client.register_tensor(B::float_tensor_handle(tensor), shape.dims.into())
    }

    fn add<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(AddOps, B::add);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Add(desc.clone())),
            AddOps::<D>::new(desc),
        );

        out
    }

    fn add_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(AddOps, B::add_scalar);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem::<f32>(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::AddScalar(desc.clone())),
            AddOps::<D>::new(desc),
        );

        out
    }

    fn clamp<const D: usize>(
        tensor: FloatTensor<Self, D>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct ClampOps<const D: usize> {
            desc: ClampOpsDescription<f32>,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for ClampOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&self.desc.tensor);
                let output = B::clamp(input, self.desc.min.elem(), self.desc.max.elem());

                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Clamp(desc.clone())),
            ClampOps::<D>::new(desc),
        );

        out
    }

    fn sub<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(SubOps, B::sub);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Sub(desc.clone())),
            SubOps::<D>::new(desc),
        );

        out
    }

    fn sub_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(SubOps, B::sub_scalar);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());
        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };

        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::SubScalar(desc.clone())),
            SubOps::<D>::new(desc),
        );

        out
    }

    fn mul<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(MulOps, B::mul);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Mul(desc.clone())),
            MulOps::<D>::new(desc),
        );

        out
    }

    fn mul_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(MulOps, B::mul_scalar);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::MulScalar(desc.clone())),
            MulOps::<D>::new(desc),
        );

        out
    }

    fn div<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        binary_float_ops!(DivOps, B::div);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Div(desc.clone())),
            DivOps::<D>::new(desc),
        );

        out
    }

    fn div_scalar<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        scalar_float_ops!(DivOps, B::div_scalar);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::DivScalar(desc.clone())),
            DivOps::<D>::new(desc),
        );

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

        let out = lhs.client.tensor_uninitialized(shape);
        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };

        out.client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::Matmul(desc.clone())),
            MatmulOps::<D>::new(desc),
        );

        out
    }

    fn swap_dims<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct SwapDimsOps<const D: usize> {
            desc: SwapDimsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for SwapDimsOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D>(&self.desc.input);
                let output = B::swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::BaseOpsFloat(BaseOpsDescription::SwapDims(desc.clone())),
            SwapDimsOps::<D>::new(desc),
        );

        out
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> FloatTensor<Self, D2> {
        #[derive(new)]
        struct ReshapeDimsOps<const D1: usize, const D2: usize> {
            desc: ReshapeDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for ReshapeDimsOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_float_tensor::<D1>(&self.desc.input);
                let output = B::reshape::<D1, D2>(input, Shape::from(&self.desc.out.shape));
                handles.register_float_tensor(&self.desc.out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ReshapeDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::BaseOpsFloat(BaseOpsDescription::Reshape(desc.clone())),
            ReshapeDimsOps::<D1, D2>::new(desc),
        );

        out
    }

    fn gather<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct GatherOps<const D: usize> {
            desc: GatherOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for GatherOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);

                let output = B::gather(self.desc.dim, tensor, indices);
                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Gather(desc.clone())),
            GatherOps::<D>::new(desc),
        );

        out
    }

    fn scatter<const D: usize>(
        dim: usize,
        tensor: FloatTensor<Self, D>,
        indices: IntTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct ScatterOps<const D: usize> {
            desc: ScatterOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for ScatterOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);
                let value = handles.get_float_tensor(&self.desc.value);

                let output = B::scatter(self.desc.dim, tensor, indices, value);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Scatter(desc.clone())),
            ScatterOps::<D>::new(desc),
        );

        out
    }

    fn select<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct SelectOps<const D: usize> {
            desc: SelectOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for SelectOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);

                let output = B::select(tensor, self.desc.dim, indices);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Select(desc.clone())),
            SelectOps::<D>::new(desc),
        );

        out
    }

    fn select_assign<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct SelectAssignOps<const D: usize> {
            desc: SelectAssignOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for SelectAssignOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let indices = handles.get_int_tensor(&self.desc.indices);
                let value = handles.get_float_tensor(&self.desc.value);

                let output = B::select_assign(tensor, self.desc.dim, indices, value);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::SelectAssign(
                desc.clone(),
            )),
            SelectAssignOps::<D>::new(desc),
        );

        out
    }

    fn slice<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> FloatTensor<Self, D1> {
        #[derive(new)]
        struct SliceOps<const D1: usize, const D2: usize> {
            desc: SliceOpsDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for SliceOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D1>(&self.desc.tensor);

                let output =
                    B::slice::<D1, D2>(tensor, self.desc.ranges.clone().try_into().unwrap());

                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::BaseOpsFloat(BaseOpsDescription::Slice(desc.clone())),
            SliceOps::<D1, D2>::new(desc),
        );

        out
    }

    fn slice_assign<const D1: usize, const D2: usize>(
        tensor: FloatTensor<Self, D1>,
        ranges: [Range<usize>; D2],
        value: FloatTensor<Self, D1>,
    ) -> FloatTensor<Self, D1> {
        #[derive(new)]
        struct SliceAssignOps<const D1: usize, const D2: usize> {
            desc: SliceAssignOpsDescription,
        }

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for SliceAssignOps<D1, D2> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D1>(&self.desc.tensor);
                let value = handles.get_float_tensor::<D1>(&self.desc.value);

                let output = B::slice_assign::<D1, D2>(
                    tensor,
                    self.desc.ranges.clone().try_into().unwrap(),
                    value,
                );

                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::BaseOpsFloat(BaseOpsDescription::SliceAssign(desc.clone())),
            SliceAssignOps::<D1, D2>::new(desc),
        );

        out
    }

    fn mask_where<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatTensor<Self, D>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct MaskWhereOps<const D: usize> {
            desc: MaskWhereOpsDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for MaskWhereOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let value = handles.get_float_tensor(&self.desc.value);
                let mask = handles.get_bool_tensor(&self.desc.mask);

                let output = B::mask_where(tensor, mask, value);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::MaskWhere(desc.clone())),
            MaskWhereOps::<D>::new(desc),
        );

        out
    }

    fn mask_fill<const D: usize>(
        tensor: FloatTensor<Self, D>,
        mask: BoolTensor<Self, D>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self, D> {
        #[derive(new)]
        struct MaskFillOps<const D: usize> {
            desc: MaskFillOpsDescription<f32>,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for MaskFillOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let mask = handles.get_bool_tensor(&self.desc.mask);

                let output = B::mask_fill(tensor, mask, self.desc.value.elem());

                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::MaskFill(desc.clone())),
            MaskFillOps::<D>::new(desc),
        );

        out
    }

    fn equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(EqualOps, B::equal);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::BaseOpsFloat(BaseOpsDescription::Equal(desc.clone())),
            EqualOps::<D>::new(desc),
        );

        out
    }

    fn equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(EqualElemOps, B::equal_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::EqualElem(desc.clone())),
            EqualElemOps::<D>::new(desc),
        );

        out
    }

    fn greater<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(GreaterOps, B::greater);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Greater(desc.clone())),
            GreaterOps::<D>::new(desc),
        );

        out
    }

    fn greater_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(GreaterElemOps, B::greater_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::GreaterElem(desc.clone())),
            GreaterElemOps::<D>::new(desc),
        );

        out
    }

    fn greater_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(GreaterEqualOps, B::greater_equal);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::GreaterEqual(
                desc.clone(),
            )),
            GreaterEqualOps::<D>::new(desc),
        );

        out
    }

    fn greater_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(GreaterEqualElemOps, B::greater_equal_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::GreaterEqualElem(
                desc.clone(),
            )),
            GreaterEqualElemOps::<D>::new(desc),
        );

        out
    }

    fn lower<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(LowerOps, B::lower);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Lower(desc.clone())),
            LowerOps::<D>::new(desc),
        );

        out
    }

    fn lower_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(LowerElemOps, B::lower_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::LowerElem(desc.clone())),
            LowerElemOps::<D>::new(desc),
        );

        out
    }

    fn lower_equal<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        binary_float_cmp_ops!(LowerEqualOps, B::lower_equal);

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        let desc = BinaryOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::LowerEqual(desc.clone())),
            LowerEqualOps::<D>::new(desc),
        );

        out
    }

    fn lower_equal_elem<const D: usize>(
        lhs: FloatTensor<Self, D>,
        rhs: FloatElem<Self>,
    ) -> BoolTensor<Self, D> {
        scalar_float_cmp_ops!(LowerEqualElemOps, B::lower_equal_elem);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs: rhs.elem(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::LowerEqualElem(
                desc.clone(),
            )),
            LowerEqualElemOps::<D>::new(desc),
        );

        out
    }

    fn sum<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(SumOps, B::sum);

        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Sum(desc.clone())),
            SumOps::<D>::new(desc),
        );

        out
    }

    fn sum_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        scalar_float_ops!(SumDimOps, B::sum_dim, usize, noconvert);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::SumDim(desc.clone())),
            SumDimOps::<D>::new(desc),
        );

        out
    }

    fn mean<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(MeanOps, B::mean);

        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Mean(desc.clone())),
            MeanOps::<D>::new(desc),
        );

        out
    }

    fn mean_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        scalar_float_ops!(MeanDimOps, B::mean_dim, usize, noconvert);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::MeanDim(desc.clone())),
            MeanDimOps::<D>::new(desc),
        );

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

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = UnaryOpsDescription {
            input: lhs.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::Exp(desc.clone())),
            ExpOps::<D>::new(desc),
        );

        out
    }

    fn log<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(LogOps, B::log);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::Log(desc.clone())),
            LogOps::<D>::new(desc),
        );

        out
    }

    fn log1p<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(Log1pOps, B::log1p);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::Log1p(desc.clone())),
            Log1pOps::<D>::new(desc),
        );

        out
    }

    fn powf<const D: usize>(lhs: FloatTensor<Self, D>, rhs: f32) -> FloatTensor<Self, D> {
        scalar_float_ops!(PowfOps, B::powf, f32);

        let out = lhs.client.tensor_uninitialized(lhs.shape.clone());

        let desc = ScalarOpsDescription {
            lhs: lhs.into_description(),
            rhs,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::Powf(desc.clone())),
            PowfOps::<D>::new(desc),
        );

        out
    }

    fn sqrt<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(SqrtOps, B::sqrt);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::Sqrt(desc.clone())),
            SqrtOps::<D>::new(desc),
        );

        out
    }

    fn abs<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(AbsOps, B::abs);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Abs(desc.clone())),
            AbsOps::<D>::new(desc),
        );

        out
    }

    fn cos<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(CosOps, B::cos);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::Cos(desc.clone())),
            CosOps::<D>::new(desc),
        );

        out
    }

    fn sin<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(SinOps, B::sin);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::Sin(desc.clone())),
            SinOps::<D>::new(desc),
        );

        out
    }

    fn tanh<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(TanhOps, B::tanh);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::Tanh(desc.clone())),
            TanhOps::<D>::new(desc),
        );

        out
    }

    fn recip<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(Recip, B::recip);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());
        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::Recip(desc.clone())),
            Recip::<D>::new(desc),
        );

        out
    }

    fn erf<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, D> {
        unary_float_ops!(TanhOps, B::erf);

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::FloatOps(FloatOpsDescription::Erf(desc.clone())),
            TanhOps::<D>::new(desc),
        );

        out
    }

    fn cat<const D: usize>(tensors: Vec<FloatTensor<Self, D>>, dim: usize) -> FloatTensor<Self, D> {
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
                    .map(|tensor| handles.get_float_tensor(tensor))
                    .collect();

                let output = B::cat::<D>(tensors, self.desc.dim);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::BaseOpsFloat(BaseOpsDescription::Cat(desc.clone())),
            CatOps::<D>::new(desc),
        );

        out
    }

    fn argmax<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_float2int_ops!(ArgMaxOps, B::argmax, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::ArgMax(desc.clone())),
            ArgMaxOps::<D>::new(desc),
        );

        out
    }

    fn argmin<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> IntTensor<Self, D> {
        scalar_float2int_ops!(ArgMinOps, B::argmin, usize);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::ArgMin(desc.clone())),
            ArgMinOps::<D>::new(desc),
        );

        out
    }

    fn max<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(MaxOps, B::max);

        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Max(desc.clone())),
            MaxOps::<D>::new(desc),
        );

        out
    }

    fn max_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        scalar_float_ops!(MaxDimOps, B::max_dim, usize, noconvert);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::MaxDim(desc.clone())),
            MaxDimOps::<D>::new(desc),
        );

        out
    }

    fn max_dim_with_indices<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> (FloatTensor<Self, D>, IntTensor<Self, D>) {
        #[derive(new)]
        struct MaxDimWithIndicesOps<const D: usize> {
            desc: ReduceDimWithIndicesDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for MaxDimWithIndicesOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let (output, indices) = B::max_dim_with_indices(tensor, self.desc.dim);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::MaxDimWithIndices(
                desc.clone(),
            )),
            MaxDimWithIndicesOps::<D>::new(desc),
        );

        (out, out_indices)
    }

    fn min<const D: usize>(tensor: FloatTensor<Self, D>) -> FloatTensor<Self, 1> {
        unary_float_ops!(MinOps, B::min);

        let out = tensor.client.tensor_uninitialized(vec![1]);

        let desc = UnaryOpsDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::Min(desc.clone())),
            MinOps::<D>::new(desc),
        );

        out
    }

    fn min_dim<const D: usize>(tensor: FloatTensor<Self, D>, dim: usize) -> FloatTensor<Self, D> {
        scalar_float_ops!(MinDimOps, B::min_dim, usize, noconvert);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape);

        let desc = ScalarOpsDescription {
            lhs: tensor.into_description(),
            rhs: dim,
            out: out.to_description_out(),
        };
        out.client.register(
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::MinDim(desc.clone())),
            MinDimOps::<D>::new(desc),
        );

        out
    }

    fn min_dim_with_indices<const D: usize>(
        tensor: FloatTensor<Self, D>,
        dim: usize,
    ) -> (FloatTensor<Self, D>, IntTensor<Self, D>) {
        #[derive(new)]
        struct MinDimWithIndicesOps<const D: usize> {
            desc: ReduceDimWithIndicesDescription,
        }

        impl<const D: usize, B: FusionBackend> Ops<B> for MinDimWithIndicesOps<D> {
            fn execute(self: Box<Self>, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_float_tensor::<D>(&self.desc.tensor);
                let (output, indices) = B::min_dim_with_indices(tensor, self.desc.dim);

                handles.register_float_tensor(&self.desc.out.id, output);
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
            TensorOpsDescription::NumericOpsFloat(NumericOpsDescription::MinDimWithIndices(
                desc.clone(),
            )),
            MinDimWithIndicesOps::<D>::new(desc),
        );

        (out, out_indices)
    }
}
