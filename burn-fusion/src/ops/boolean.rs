use crate::{
    client::FusionClient,
    get_client,
    graph::{
        BaseOpsDescription, BinaryOpsDescription, BoolOpsDescription, CatOpsDescription, Ops,
        ReshapeDescription, SliceAssignOpsDescription, SliceOpsDescription, SwapDimsDescription,
        TensorOpsDescription, UnaryOpsDescription,
    },
    ops::binary::binary_ops_shape,
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

        client.register_tensor(B::bool_tensor_handle(tensor), shape.dims.into())
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

        client.register_tensor(B::bool_tensor_handle(tensor), shape.dims.into())
    }

    fn bool_into_int<const D: usize>(
        tensor: BoolTensor<Self, D>,
    ) -> burn_tensor::ops::IntTensor<Self, D> {
        struct IntoIntOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for IntoIntOps<D> {
            type Args = UnaryOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D>(&args.input);
                let output = B::bool_into_int(input);
                handles.register_int_tensor(&args.out.id, output);
            }
        }

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        out.client
            .register(TensorOpsDescription::BoolOps(BoolOpsDescription::IntoInt(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(IntoIntOps::<D>),
            )));

        out
    }

    fn bool_into_float<const D: usize>(
        tensor: BoolTensor<Self, D>,
    ) -> burn_tensor::ops::FloatTensor<Self, D> {
        struct IntoFloatOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for IntoFloatOps<D> {
            type Args = UnaryOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D>(&args.input);
                let output = B::bool_into_float(input);
                handles.register_float_tensor(&args.out.id, output);
            }
        }

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        out.client.register(TensorOpsDescription::BoolOps(
            BoolOpsDescription::IntoFloat(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(IntoFloatOps::<D>),
            ),
        ));

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

        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.client.clone();

        client_original
            .clone()
            .change_client_bool::<D>(tensor.into_description(), client_target)
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> BoolTensor<Self, D2> {
        struct ReshapeDimsOps<const D1: usize, const D2: usize>;

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for ReshapeDimsOps<D1, D2> {
            type Args = ReshapeDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D1>(&args.input);
                let output = B::bool_reshape::<D1, D2>(input, Shape::from(&args.shape));
                handles.register_bool_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = shape.dims.into();
        let out = tensor.client.tensor_uninitialized(shape.clone());

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::BaseOpsBool(
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

    fn bool_slice<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
    ) -> BoolTensor<Self, D1> {
        struct SliceOps<const D1: usize, const D2: usize>;

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for SliceOps<D1, D2> {
            type Args = SliceOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_bool_tensor::<D1>(&args.tensor);

                let output =
                    B::bool_slice::<D1, D2>(tensor, args.ranges.clone().try_into().unwrap());

                handles.register_bool_tensor(&args.out.id, output);
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
            .register(TensorOpsDescription::BaseOpsBool(
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

    fn bool_slice_assign<const D1: usize, const D2: usize>(
        tensor: BoolTensor<Self, D1>,
        ranges: [std::ops::Range<usize>; D2],
        value: BoolTensor<Self, D1>,
    ) -> BoolTensor<Self, D1> {
        struct SliceAssignOps<const D1: usize, const D2: usize>;

        impl<const D1: usize, const D2: usize, B: FusionBackend> Ops<B> for SliceAssignOps<D1, D2> {
            type Args = SliceAssignOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensor = handles.get_bool_tensor::<D1>(&args.tensor);
                let value = handles.get_bool_tensor::<D1>(&args.value);

                let output = B::bool_slice_assign::<D1, D2>(
                    tensor,
                    args.ranges.clone().try_into().unwrap(),
                    value,
                );

                handles.register_bool_tensor(&args.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::BaseOpsBool(
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

    fn bool_cat<const D: usize>(
        tensors: Vec<BoolTensor<Self, D>>,
        dim: usize,
    ) -> BoolTensor<Self, D> {
        struct CatOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for CatOps<D> {
            type Args = CatOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let tensors = args
                    .tensors
                    .iter()
                    .map(|tensor| handles.get_bool_tensor(tensor))
                    .collect();

                let output = B::bool_cat::<D>(tensors, args.dim);

                handles.register_bool_tensor(&args.out.id, output);
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

        client.register(TensorOpsDescription::BaseOpsBool(BaseOpsDescription::Cat(
            CatOpsDescription {
                tensors: tensors.into_iter().map(|t| t.into_description()).collect(),
                dim,
                out: out.to_description_out(),
            },
            Box::new(CatOps::<D>),
        )));

        out
    }

    fn bool_equal<const D: usize>(
        lhs: BoolTensor<Self, D>,
        rhs: BoolTensor<Self, D>,
    ) -> BoolTensor<Self, D> {
        struct EqualOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for EqualOps<D> {
            type Args = BinaryOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let lhs = handles.get_bool_tensor::<D>(&args.lhs);
                let rhs = handles.get_bool_tensor(&args.rhs);
                let output = B::bool_equal(lhs, rhs);
                handles.register_bool_tensor(&args.out.id, output);
            }
        }

        let out = lhs
            .client
            .tensor_uninitialized(binary_ops_shape(&lhs.shape, &rhs.shape));

        out.client.register(TensorOpsDescription::BaseOpsBool(
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

    fn bool_not<const D: usize>(tensor: BoolTensor<Self, D>) -> BoolTensor<Self, D> {
        struct NotOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for NotOps<D> {
            type Args = UnaryOpsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D>(&args.input);
                let output = B::bool_not(input);
                handles.register_bool_tensor(&args.out.id, output);
            }
        }

        let out = tensor.client.tensor_uninitialized(tensor.shape.clone());

        out.client.register(TensorOpsDescription::BoolOps(
            crate::graph::BoolOpsDescription::Not(
                UnaryOpsDescription {
                    input: tensor.into_description(),
                    out: out.to_description_out(),
                },
                Box::new(NotOps::<D>),
            ),
        ));

        out
    }

    fn bool_swap_dims<const D: usize>(
        tensor: BoolTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> BoolTensor<Self, D> {
        struct SwapDimsOps<const D: usize>;

        impl<const D: usize, B: FusionBackend> Ops<B> for SwapDimsOps<D> {
            type Args = SwapDimsDescription;

            fn execute(&self, args: &Self::Args, handles: &mut crate::HandleContainer<B>) {
                let input = handles.get_bool_tensor::<D>(&args.input);
                let output = B::bool_swap_dims(input, args.dim1, args.dim2);
                handles.register_bool_tensor(&args.out.id, output);
            }
        }

        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];

        let out = tensor.client.tensor_uninitialized(shape);

        tensor
            .client
            .clone()
            .register(TensorOpsDescription::BaseOpsBool(
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
}
