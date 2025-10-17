use super::NoOp;
use crate::{
    Fusion, FusionBackend, binary_float_cmp_ops, binary_float_ops,
    client::OperationOutput,
    get_client, reduce_float_ops, reduce_float2int_ops, scalar_float_cmp_ops, scalar_float_ops,
    stream::{OperationStreams, execution::Operation},
    unary_float_ops,
};
use burn_ir::*;
use burn_tensor::{
    Device, Distribution, Element, FloatDType, Shape, Slice, TensorData,
    ops::{BoolTensor, FloatElem, FloatTensor, FloatTensorOps, IntTensor},
};
use std::marker::PhantomData;

impl<B: FusionBackend> FloatTensorOps<Self> for Fusion<B> {
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        let client = get_client::<B>(device);
        let dtype = data.dtype;
        let tensor = B::float_from_data(data, device);
        let shape = burn_tensor::TensorMetadata::shape(&tensor);

        let handle = B::float_tensor_handle(tensor);
        let desc = InitOperationIr::create(shape, dtype, || client.register_tensor_handle(handle));

        client
            .register(
                OperationStreams::default(),
                OperationIr::Init(desc),
                NoOp::<B>::new(),
            )
            .output()
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &Device<Self>,
    ) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct RandomOps<B: FusionBackend> {
            desc: RandomOpIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for RandomOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let output: B::FloatTensorPrimitive = B::float_random(
                    self.desc.out.shape.clone(),
                    self.desc.distribution,
                    &self.device,
                );
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let dtype = FloatElem::<Self>::dtype();
        let client = get_client::<B>(device);
        let desc = RandomOpIr::create(shape, dtype, distribution, || client.create_empty_handle());

        client
            .register(
                OperationStreams::default(),
                OperationIr::Float(dtype, FloatOperationIr::Random(desc.clone())),
                RandomOps::<B>::new(desc, device.clone()),
            )
            .output()
    }

    fn float_zeros(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct ZerosOps<B: FusionBackend> {
            out: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ZerosOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let shape = self.out.shape.clone();
                let output = B::float_zeros(shape, &self.device, self.out.dtype.into());
                handles.register_float_tensor::<B>(&self.out.id, output);
            }
        }

        let client = get_client::<B>(device);
        let desc = CreationOpIr::create(shape, dtype.into(), || client.create_empty_handle());

        client
            .register(
                OperationStreams::default(),
                OperationIr::BaseFloat(BaseOperationIr::Zeros(desc.clone())),
                ZerosOps::<B>::new(desc.out, device.clone()),
            )
            .output()
    }

    fn float_ones(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct OnesOps<B: FusionBackend> {
            out: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for OnesOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let shape = self.out.shape.clone();
                let output = B::float_ones(shape, &self.device, self.out.dtype.into());
                handles.register_float_tensor::<B>(&self.out.id, output);
            }
        }

        let client = get_client::<B>(device);
        let desc = CreationOpIr::create(shape, dtype.into(), || client.create_empty_handle());

        client
            .register(
                OperationStreams::default(),
                OperationIr::BaseFloat(BaseOperationIr::Ones(desc.clone())),
                OnesOps::<B>::new(desc.out, device.clone()),
            )
            .output()
    }

    fn float_full(
        shape: Shape,
        fill_value: FloatElem<Self>,
        device: &Device<Self>,
        dtype: FloatDType,
    ) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct FullOps<B: FusionBackend> {
            out: TensorIr,
            elem: ScalarIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for FullOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let shape = self.out.shape.clone();
                let output: B::FloatTensorPrimitive =
                    B::float_full(shape, self.elem.elem(), &self.device, self.out.dtype.into());
                handles.register_float_tensor::<B>(&self.out.id, output);
            }
        }

        let dtype = dtype.into();
        let client = get_client::<B>(device);
        let value = ScalarIr::with_dtype(fill_value, &dtype);
        let desc = FullOpIr::create(shape, dtype, value, || client.create_empty_handle());

        client
            .register(
                OperationStreams::default(),
                OperationIr::NumericFloat(dtype, NumericOperationIr::Full(desc.clone())),
                FullOps::<B>::new(desc.out, desc.value, device.clone()),
            )
            .output()
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> TensorData {
        tensor.into_data::<B>().await
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device<Self> {
        tensor.client.device().clone()
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self> {
        let device_original: &B::Device = tensor.client.device();

        if device_original == device {
            return tensor;
        }

        let id = tensor.stream;
        let client_target = get_client::<B>(device);
        let client_original = tensor.client.clone();

        client_original
            .clone()
            .change_client_float::<B>(tensor.into_ir(), client_target, id)
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct IntoIntOps<B: FusionBackend> {
            desc: CastOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntoIntOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_into_int(input);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = CastOpIr::create(tensor.into_ir(), B::IntElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Float(desc.input.dtype, FloatOperationIr::IntoInt(desc.clone())),
                IntoIntOps::<B>::new(desc),
            )
            .output()
    }

    fn float_empty(shape: Shape, device: &Device<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct EmptyOps<B: FusionBackend> {
            desc: TensorIr,
            device: Device<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for EmptyOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let output = B::float_empty(
                    self.desc.shape.clone(),
                    &self.device,
                    self.desc.dtype.into(),
                );
                handles.register_float_tensor::<B>(&self.desc.id, output);
            }
        }

        let client = get_client::<B>(device);
        let desc = CreationOpIr::create(shape, dtype.into(), || client.create_empty_handle());

        client
            .register(
                OperationStreams::default(),
                OperationIr::BaseFloat(BaseOperationIr::Empty(desc.clone())),
                EmptyOps::<B>::new(desc.out, device.clone()),
            )
            .output()
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(AddOps, B::float_add);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Add(desc.clone())),
                AddOps::<B>::new(desc),
            )
            .output()
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        scalar_float_ops!(AddOps, B::float_add_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.out.dtype,
                    NumericOperationIr::AddScalar(desc.clone()),
                ),
                AddOps::<B>::new(desc),
            )
            .output()
    }

    fn float_clamp(
        tensor: FloatTensor<Self>,
        min: FloatElem<Self>,
        max: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct ClampOps<B: FusionBackend> {
            desc: ClampOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ClampOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.tensor);
                let output = B::float_clamp(input, self.desc.min.elem(), self.desc.max.elem());

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let min = ScalarIr::with_dtype(min, &tensor.dtype);
        let max = ScalarIr::with_dtype(max, &tensor.dtype);
        let desc = ClampOpIr::create(tensor.into_ir(), min, max, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.tensor.dtype,
                    NumericOperationIr::Clamp(desc.clone()),
                ),
                ClampOps::<B>::new(desc),
            )
            .output()
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(SubOps, B::float_sub);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Sub(desc.clone())),
                SubOps::<B>::new(desc),
            )
            .output()
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        scalar_float_ops!(SubOps, B::float_sub_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.out.dtype,
                    NumericOperationIr::SubScalar(desc.clone()),
                ),
                SubOps::<B>::new(desc),
            )
            .output()
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(MulOps, B::float_mul);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Mul(desc.clone())),
                MulOps::<B>::new(desc),
            )
            .output()
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        scalar_float_ops!(MulOps, B::float_mul_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.out.dtype,
                    NumericOperationIr::MulScalar(desc.clone()),
                ),
                MulOps::<B>::new(desc),
            )
            .output()
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(DivOps, B::float_div);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Div(desc.clone())),
                DivOps::<B>::new(desc),
            )
            .output()
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        scalar_float_ops!(DivOps, B::float_div_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.out.dtype,
                    NumericOperationIr::DivScalar(desc.clone()),
                ),
                DivOps::<B>::new(desc),
            )
            .output()
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(ModOps, B::float_remainder);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Rem(desc.clone())),
                ModOps::<B>::new(desc),
            )
            .output()
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        scalar_float_ops!(ModOps, B::float_remainder_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.out.dtype,
                    NumericOperationIr::RemScalar(desc.clone()),
                ),
                ModOps::<B>::new(desc),
            )
            .output()
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(MatmulOps, B::float_matmul);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = MatmulOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Matmul(desc.clone())),
                MatmulOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn float_cross(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
        dim: usize,
    ) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct CrossOps<B: FusionBackend> {
            desc: CrossOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CrossOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = handles.get_float_tensor::<B>(&self.desc.lhs);
                let rhs = handles.get_float_tensor::<B>(&self.desc.rhs);
                let output = B::float_cross(lhs, rhs, self.desc.dim);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = CrossOpIr::create(lhs.into_ir(), rhs.into_ir(), dim, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Cross(desc.clone())),
                CrossOps::<B>::new(desc),
            )
            .output()
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct SwapDimsOps<B: FusionBackend> {
            desc: SwapDimsOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SwapDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = SwapDimsOpIr::create(tensor.into_ir(), dim1, dim2, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::SwapDims(desc.clone())),
                SwapDimsOps::<B>::new(desc),
            )
            .output()
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        if tensor.shape == shape {
            return tensor;
        }

        #[derive(new, Debug)]
        struct ReshapeDimsOps<B: FusionBackend> {
            desc: ShapeOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ReshapeDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_reshape(input, self.desc.out.shape.clone());
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ShapeOpIr::reshape(tensor.into_ir(), shape, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::Reshape(desc.clone())),
                ReshapeDimsOps::<B>::new(desc),
            )
            .output()
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct GatherOps<B: FusionBackend> {
            desc: GatherOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for GatherOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);

                let output = B::float_gather(self.desc.dim, tensor, indices);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &indices]);

        let client = tensor.client.clone();
        let desc = GatherOpIr::create(tensor.into_ir(), dim, indices.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Gather(desc.clone())),
                GatherOps::<B>::new(desc),
            )
            .output()
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct ScatterOps<B: FusionBackend> {
            desc: ScatterOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ScatterOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);
                let value = handles.get_float_tensor::<B>(&self.desc.value);

                let output = B::float_scatter(self.desc.dim, tensor, indices, value);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &indices, &value]);

        let client = tensor.client.clone();
        let desc = ScatterOpIr::create(
            tensor.into_ir(),
            dim,
            indices.into_ir(),
            value.into_ir(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.out.dtype,
                    NumericOperationIr::Scatter(desc.clone()),
                ),
                ScatterOps::<B>::new(desc),
            )
            .output()
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct SelectOps<B: FusionBackend> {
            desc: SelectOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SelectOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);

                let output = B::float_select(tensor, self.desc.dim, indices);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &indices]);

        let client = tensor.client.clone();
        let desc = SelectOpIr::create(tensor.into_ir(), dim, indices.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Select(desc.clone())),
                SelectOps::<B>::new(desc),
            )
            .output()
    }

    fn float_select_assign(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct SelectAssignOps<B: FusionBackend> {
            desc: SelectAssignOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SelectAssignOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);
                let value = handles.get_float_tensor::<B>(&self.desc.value);

                let output = B::float_select_assign(tensor, self.desc.dim, indices, value);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &indices, &value]);

        let client = tensor.client.clone();
        let desc = SelectAssignOpIr::create(
            tensor.into_ir(),
            dim,
            indices.into_ir(),
            value.into_ir(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.out.dtype,
                    NumericOperationIr::SelectAssign(desc.clone()),
                ),
                SelectAssignOps::<B>::new(desc),
            )
            .output()
    }

    fn float_slice(tensor: FloatTensor<Self>, slices: &[Slice]) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct SliceOps<B: FusionBackend> {
            desc: SliceOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);

                let output = B::float_slice(tensor, self.desc.ranges.as_slice());

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = SliceOpIr::create(tensor.into_ir(), slices.into(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::Slice(desc.clone())),
                SliceOps::<B>::new(desc),
            )
            .output()
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        slices: &[burn_tensor::Slice],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct SliceAssignOps<B: FusionBackend> {
            desc: SliceAssignOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceAssignOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let value = handles.get_float_tensor::<B>(&self.desc.value);

                let output = B::float_slice_assign(tensor, self.desc.ranges.as_slice(), value);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &value]);

        let client = tensor.client.clone();
        let desc =
            SliceAssignOpIr::create(tensor.into_ir(), slices.into(), value.into_ir(), || {
                client.create_empty_handle()
            });

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::SliceAssign(desc.clone())),
                SliceAssignOps::<B>::new(desc),
            )
            .output()
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct MaskWhereOps<B: FusionBackend> {
            desc: MaskWhereOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaskWhereOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let value = handles.get_float_tensor::<B>(&self.desc.value);
                let mask = handles.get_bool_tensor::<B>(&self.desc.mask);

                let output = B::float_mask_where(tensor, mask, value);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &mask, &value]);

        let client = tensor.client.clone();
        let desc = MaskWhereOpIr::create(tensor.into_ir(), mask.into_ir(), value.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.out.dtype,
                    NumericOperationIr::MaskWhere(desc.clone()),
                ),
                MaskWhereOps::<B>::new(desc),
            )
            .output()
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: BoolTensor<Self>,
        value: FloatElem<Self>,
    ) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct MaskFillOps<B: FusionBackend> {
            desc: MaskFillOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaskFillOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let mask = handles.get_bool_tensor::<B>(&self.desc.mask);

                let output = B::float_mask_fill(tensor, mask, self.desc.value.elem());

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &mask]);

        let client = tensor.client.clone();
        let value = ScalarIr::with_dtype(value, &tensor.dtype);
        let desc = MaskFillOpIr::create(tensor.into_ir(), mask.into_ir(), value, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.out.dtype,
                    NumericOperationIr::MaskFill(desc.clone()),
                ),
                MaskFillOps::<B>::new(desc),
            )
            .output()
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float_cmp_ops!(EqualOps, B::float_equal);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            B::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::Equal(desc.clone())),
                EqualOps::<B>::new(desc),
            )
            .output()
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        scalar_float_cmp_ops!(EqualElemOps, B::float_equal_elem);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.lhs.dtype,
                    NumericOperationIr::EqualElem(desc.clone()),
                ),
                EqualElemOps::<B>::new(desc),
            )
            .output()
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float_cmp_ops!(GreaterOps, B::float_greater);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            B::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.lhs.dtype,
                    NumericOperationIr::Greater(desc.clone()),
                ),
                GreaterOps::<B>::new(desc),
            )
            .output()
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        scalar_float_cmp_ops!(GreaterElemOps, B::float_greater_elem);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.lhs.dtype,
                    NumericOperationIr::GreaterElem(desc.clone()),
                ),
                GreaterElemOps::<B>::new(desc),
            )
            .output()
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float_cmp_ops!(GreaterEqualOps, B::float_greater_equal);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            B::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.lhs.dtype,
                    NumericOperationIr::GreaterEqual(desc.clone()),
                ),
                GreaterEqualOps::<B>::new(desc),
            )
            .output()
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        scalar_float_cmp_ops!(GreaterEqualElemOps, B::float_greater_equal_elem);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.lhs.dtype,
                    NumericOperationIr::GreaterEqualElem(desc.clone()),
                ),
                GreaterEqualElemOps::<B>::new(desc),
            )
            .output()
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float_cmp_ops!(LowerOps, B::float_lower);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            B::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.lhs.dtype, NumericOperationIr::Lower(desc.clone())),
                LowerOps::<B>::new(desc),
            )
            .output()
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        scalar_float_cmp_ops!(LowerElemOps, B::float_lower_elem);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.lhs.dtype,
                    NumericOperationIr::LowerElem(desc.clone()),
                ),
                LowerElemOps::<B>::new(desc),
            )
            .output()
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float_cmp_ops!(LowerEqualOps, B::float_lower_equal);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create_comparison(
            lhs.into_ir(),
            rhs.into_ir(),
            B::BoolElem::dtype(),
            || client.create_empty_handle(),
        );

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.lhs.dtype,
                    NumericOperationIr::LowerEqual(desc.clone()),
                ),
                LowerEqualOps::<B>::new(desc),
            )
            .output()
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        scalar_float_cmp_ops!(LowerEqualElemOps, B::float_lower_equal_elem);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create_comparison(lhs.into_ir(), rhs, B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.lhs.dtype,
                    NumericOperationIr::LowerEqualElem(desc.clone()),
                ),
                LowerEqualElemOps::<B>::new(desc),
            )
            .output()
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(SumOps, B::float_sum, reduce);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Sum(desc.clone())),
                SumOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, axis: usize) -> FloatTensor<Self> {
        reduce_float_ops!(SumDimOps, B::float_sum_dim);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), axis, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::SumDim(desc.clone())),
                SumDimOps::<B>::new(desc),
            )
            .output()
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(ProdOps, B::float_prod, reduce);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Prod(desc.clone())),
                ProdOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce_float_ops!(ProdDimOps, B::float_prod_dim);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.out.dtype,
                    NumericOperationIr::ProdDim(desc.clone()),
                ),
                ProdDimOps::<B>::new(desc),
            )
            .output()
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(MeanOps, B::float_mean, reduce);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Mean(desc.clone())),
                MeanOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce_float_ops!(MeanDimOps, B::float_mean_dim);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.out.dtype,
                    NumericOperationIr::MeanDim(desc.clone()),
                ),
                MeanDimOps::<B>::new(desc),
            )
            .output()
    }

    fn float_cumsum(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct CumsumOps<B: FusionBackend> {
            desc: DimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CumsumOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_cumsum(input, self.desc.axis);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::CumSum(desc.clone())),
                CumsumOps::<B>::new(desc),
            )
            .output()
    }

    fn float_cumprod(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct CumprodOps<B: FusionBackend> {
            desc: DimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CumprodOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_cumprod(input, self.desc.axis);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::CumProd(desc.clone())),
                CumprodOps::<B>::new(desc),
            )
            .output()
    }

    fn float_cummin(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct CumminOps<B: FusionBackend> {
            desc: DimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CumminOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_cummin(input, self.desc.axis);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::CumMin(desc.clone())),
                CumminOps::<B>::new(desc),
            )
            .output()
    }

    fn float_cummax(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct CummaxOps<B: FusionBackend> {
            desc: DimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CummaxOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_cummax(input, self.desc.axis);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = DimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::CumMax(desc.clone())),
                CummaxOps::<B>::new(desc),
            )
            .output()
    }

    fn float_exp(lhs: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(ExpOps, B::float_exp);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let desc = UnaryOpIr::create(lhs.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Exp(desc.clone())),
                ExpOps::<B>::new(desc),
            )
            .output()
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(LogOps, B::float_log);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Log(desc.clone())),
                LogOps::<B>::new(desc),
            )
            .output()
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(Log1pOps, B::float_log1p);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Log1p(desc.clone())),
                Log1pOps::<B>::new(desc),
            )
            .output()
    }

    fn float_powf_scalar_impl(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        scalar_float_ops!(PowfOps, B::float_powf_scalar);

        let streams = OperationStreams::with_inputs([&lhs]);

        let client = lhs.client.clone();
        let rhs = ScalarIr::with_dtype(rhs, &lhs.dtype);
        let desc = ScalarOpIr::create(lhs.into_ir(), rhs, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::PowfScalar(desc.clone())),
                PowfOps::<B>::new(desc),
            )
            .output()
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(SqrtOps, B::float_sqrt);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Sqrt(desc.clone())),
                SqrtOps::<B>::new(desc),
            )
            .output()
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(AbsOps, B::float_abs);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Abs(desc.clone())),
                AbsOps::<B>::new(desc),
            )
            .output()
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(CosOps, B::float_cos);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Cos(desc.clone())),
                CosOps::<B>::new(desc),
            )
            .output()
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(SinOps, B::float_sin);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Sin(desc.clone())),
                SinOps::<B>::new(desc),
            )
            .output()
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(TanhOps, B::float_tanh);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Tanh(desc.clone())),
                TanhOps::<B>::new(desc),
            )
            .output()
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(Recip, B::float_recip);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Recip(desc.clone())),
                Recip::<B>::new(desc),
            )
            .output()
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(TanhOps, B::float_erf);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Erf(desc.clone())),
                TanhOps::<B>::new(desc),
            )
            .output()
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct CatOps<B: FusionBackend> {
            desc: CatOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CatOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensors = self
                    .desc
                    .tensors
                    .iter()
                    .map(|tensor| handles.get_float_tensor::<B>(tensor))
                    .collect();

                let output = B::float_cat(tensors, self.desc.dim);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs(&tensors);

        let client = tensors.first().unwrap().client.clone();
        let tensors = tensors.into_iter().map(|t| t.into_ir()).collect();
        let desc = CatOpIr::create(tensors, dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::Cat(desc.clone())),
                CatOps::<B>::new(desc),
            )
            .output()
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce_float2int_ops!(ArgMaxOps, B::float_argmax);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        // TODO: rename `create_with_dtype` specifically for ARG / indices
        let desc = ReduceDimOpIr::create_arg(tensor.into_ir(), dim, B::IntElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.input.dtype,
                    NumericOperationIr::ArgMax(desc.clone()),
                ),
                ArgMaxOps::<B>::new(desc),
            )
            .output()
    }

    fn float_repeat_dim(tensor: FloatTensor<Self>, dim: usize, times: usize) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct RepeatDimOps<B: FusionBackend> {
            desc: RepeatDimOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for RepeatDimOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);

                let output = B::float_repeat_dim(tensor, self.desc.dim, self.desc.times);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = RepeatDimOpIr::create(tensor.into_ir(), dim, times, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::RepeatDim(desc.clone())),
                RepeatDimOps::<B>::new(desc),
            )
            .output()
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce_float2int_ops!(ArgMinOps, B::float_argmin);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create_arg(tensor.into_ir(), dim, B::IntElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.input.dtype,
                    NumericOperationIr::ArgMin(desc.clone()),
                ),
                ArgMinOps::<B>::new(desc),
            )
            .output()
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(MaxOps, B::float_max, reduce);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Max(desc.clone())),
                MaxOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce_float_ops!(MaxDimOps, B::float_max_dim);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::MaxDim(desc.clone())),
                MaxDimOps::<B>::new(desc),
            )
            .output()
    }

    fn float_max_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        #[derive(new, Debug)]
        struct MaxDimWithIndicesOps<B: FusionBackend> {
            desc: ReduceDimWithIndicesOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MaxDimWithIndicesOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let (output, indices) = B::float_max_dim_with_indices(tensor, self.desc.dim);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
                handles.register_int_tensor::<B>(&self.desc.out_indices.id, indices);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc =
            ReduceDimWithIndicesOpIr::create(tensor.into_ir(), dim, B::IntElem::dtype(), || {
                client.create_empty_handle()
            });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.tensor.dtype,
                    NumericOperationIr::MaxDimWithIndices(desc.clone()),
                ),
                MaxDimWithIndicesOps::<B>::new(desc),
            )
            .outputs()
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(MinOps, B::float_min, reduce);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Min(desc.clone())),
                MinOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce_float_ops!(MinDimOps, B::float_min_dim);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::MinDim(desc.clone())),
                MinDimOps::<B>::new(desc),
            )
            .output()
    }

    fn float_min_dim_with_indices(
        tensor: FloatTensor<Self>,
        dim: usize,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        #[derive(new, Debug)]
        struct MinDimWithIndicesOps<B: FusionBackend> {
            desc: ReduceDimWithIndicesOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MinDimWithIndicesOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let (output, indices) = B::float_min_dim_with_indices(tensor, self.desc.dim);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
                handles.register_int_tensor::<B>(&self.desc.out_indices.id, indices);
            }
        }
        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc =
            ReduceDimWithIndicesOpIr::create(tensor.into_ir(), dim, B::IntElem::dtype(), || {
                client.create_empty_handle()
            });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.tensor.dtype,
                    NumericOperationIr::MinDimWithIndices(desc.clone()),
                ),
                MinDimWithIndicesOps::<B>::new(desc),
            )
            .outputs()
    }

    fn float_max_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(MaxAbsOps, B::float_max_abs, reduce);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::MaxAbs(desc.clone())),
                MaxAbsOps::<B>::new(desc.into()),
            )
            .output()
    }

    fn float_max_abs_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce_float_ops!(MaxAbsDimOps, B::float_max_abs_dim);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ReduceDimOpIr::create(tensor.into_ir(), dim, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.out.dtype,
                    NumericOperationIr::MaxAbsDim(desc.clone()),
                ),
                MaxAbsDimOps::<B>::new(desc),
            )
            .output()
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(PowOps, B::float_powf);

        let streams = OperationStreams::with_inputs([&lhs, &rhs]);

        let client = lhs.client.clone();
        let desc = BinaryOpIr::create(lhs.into_ir(), rhs.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(desc.out.dtype, NumericOperationIr::Powf(desc.clone())),
                PowOps::<B>::new(desc),
            )
            .output()
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct PermuteDimsOps<B: FusionBackend> {
            desc: PermuteOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for PermuteDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_permute(input, self.desc.axes.as_slice());
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = PermuteOpIr::create(tensor.into_ir(), axes.into(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::Permute(desc.clone())),
                PermuteDimsOps::<B>::new(desc),
            )
            .output()
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct ExpandOps<B: FusionBackend> {
            desc: ShapeOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ExpandOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_expand(input, self.desc.out.shape.clone());

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = ShapeOpIr::expand(tensor.into_ir(), shape, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::Expand(desc.clone())),
                ExpandOps::<B>::new(desc),
            )
            .output()
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct FlipOps<B: FusionBackend> {
            desc: FlipOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for FlipOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_flip(input, &self.desc.axes);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = FlipOpIr::create(tensor.into_ir(), axes.into(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseInt(BaseOperationIr::Flip(desc.clone())),
                FlipOps::<B>::new(desc),
            )
            .output()
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(RoundOps, B::float_round);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Round(desc.clone())),
                RoundOps::<B>::new(desc),
            )
            .output()
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(FloorOps, B::float_floor);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Floor(desc.clone())),
                FloorOps::<B>::new(desc),
            )
            .output()
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(CeilOps, B::float_ceil);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Ceil(desc.clone())),
                CeilOps::<B>::new(desc),
            )
            .output()
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(TruncOps, B::float_trunc);

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create(tensor.into_ir(), || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(desc.out.dtype, FloatOperationIr::Trunc(desc.clone())),
                TruncOps::<B>::new(desc),
            )
            .output()
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: burn_tensor::FloatDType) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct CastOps<B: FusionBackend> {
            desc: CastOpIr,
            dtype: burn_tensor::FloatDType,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for CastOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output: B::FloatTensorPrimitive = B::float_cast(input, self.dtype);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = CastOpIr::create(tensor.into_ir(), dtype.into(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::Cast(desc.clone())),
                CastOps::<B>::new(desc, dtype),
            )
            .output()
    }

    fn float_unfold(
        tensor: FloatTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct UnfoldOps<B: FusionBackend> {
            desc: UnfoldOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for UnfoldOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_unfold(input, self.desc.dim, self.desc.size, self.desc.step);

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnfoldOpIr::create(tensor.into_ir(), dim, size, step, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::BaseFloat(BaseOperationIr::Unfold(desc.clone())),
                UnfoldOps::<B>::new(desc),
            )
            .output()
    }

    fn float_is_nan(tensor: FloatTensor<Self>) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct IsNanOps<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }
        impl<B: FusionBackend> Operation<B::FusionRuntime> for IsNanOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_is_nan(input);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create_comparison(tensor.into_ir(), B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Float(desc.input.dtype, FloatOperationIr::IsNan(desc.clone())),
                IsNanOps::<B>::new(desc),
            )
            .output()
    }

    fn float_is_inf(tensor: FloatTensor<Self>) -> BoolTensor<Self> {
        #[derive(new, Debug)]
        struct IsInfOps<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }
        impl<B: FusionBackend> Operation<B::FusionRuntime> for IsInfOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_is_inf(input);
                handles.register_bool_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = UnaryOpIr::create_comparison(tensor.into_ir(), B::BoolElem::dtype(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Float(desc.input.dtype, FloatOperationIr::IsInf(desc.clone())),
                IsInfOps::<B>::new(desc),
            )
            .output()
    }
}
