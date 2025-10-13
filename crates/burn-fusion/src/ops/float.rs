use super::NoOp;
use crate::{
    Fusion, FusionBackend, binary_float_cmp_ops, binary_float_ops,
    client::FusionClient,
    get_client,
    ops::binary::check_binary_op_types,
    reduce_float_ops, reduce_float2int_ops, scalar_float_cmp_ops, scalar_float_ops,
    stream::{OperationStreams, StreamId, execution::Operation},
    unary_float_ops,
};
use burn_ir::*;
use burn_tensor::ops::unfold::calculate_unfold_shape;
use burn_tensor::{
    Device, Distribution, Element, FloatDType, Shape, Slice, TensorData, TensorMetadata,
    ops::{BoolTensor, FloatElem, FloatTensor, FloatTensorOps, IntTensor},
};
use std::marker::PhantomData;

impl<B: FusionBackend> FloatTensorOps<Self> for Fusion<B> {
    fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
        let stream = StreamId::current();
        let client = get_client::<B>(&device.clone());
        let dtype = data.dtype;
        let tensor = B::float_from_data(data, device);
        let shape = tensor.shape();

        let handle = B::float_tensor_handle(tensor);
        let out = client.register_tensor(handle, shape, stream, dtype);
        let desc = out.to_ir_out();

        client.register(
            OperationStreams::default(),
            OperationIr::Init(InitOperationIr { out: desc }),
            NoOp::<B>::new(),
        );

        out
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

        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = RandomOpIr {
            out: out.to_ir_out(),
            distribution,
        };
        client.register(
            OperationStreams::default(),
            OperationIr::Float(
                FloatElem::<Self>::dtype(),
                FloatOperationIr::Random(desc.clone()),
            ),
            RandomOps::<B>::new(desc, device.clone()),
        );

        out
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

        let dtype = dtype.into();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape, dtype);

        let desc = out.to_ir_out();
        client.register(
            OperationStreams::default(),
            OperationIr::NumericFloat(dtype, NumericOperationIr::Zeros(desc.clone())),
            ZerosOps::<B>::new(desc, device.clone()),
        );

        out
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

        let dtype = dtype.into();
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape, dtype);

        let desc = out.to_ir_out();
        client.register(
            OperationStreams::default(),
            OperationIr::NumericFloat(dtype, NumericOperationIr::Ones(desc.clone())),
            OnesOps::<B>::new(desc, device.clone()),
        );

        out
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
        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape, dtype);

        let desc = (out.to_ir_out(), ScalarIr::with_dtype(fill_value, &dtype));
        client.register(
            OperationStreams::default(),
            OperationIr::NumericFloat(dtype, NumericOperationIr::Full(desc.clone())),
            FullOps::<B>::new(desc.0, desc.1, device.clone()),
        );

        out
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> TensorData {
        tensor.into_data::<B>().await
    }

    fn float_device(tensor: &FloatTensor<Self>) -> Device<Self> {
        tensor.client.device().clone()
    }

    fn float_to_device(tensor: FloatTensor<Self>, device: &Device<Self>) -> FloatTensor<Self> {
        let device_original: &B::Device = tensor.client.device();
        let device_target: B::Device = device.clone();

        if device_original == &device_target {
            return tensor;
        }

        let id = tensor.stream;
        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.client.clone();

        client_original
            .clone()
            .change_client_float::<B>(tensor.into_ir(), client_target, id)
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> IntTensor<Self> {
        #[derive(new, Debug)]
        struct IntoIntOps<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for IntoIntOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_into_int(input);

                handles.register_int_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::IntElem::dtype());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::IntoInt(desc.clone())),
            IntoIntOps::<B>::new(desc),
        );

        out
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

        let client = get_client::<B>(&device.clone());
        let out = client.tensor_uninitialized(shape.clone(), dtype.into());

        let desc = out.to_ir_out();

        client.register(
            OperationStreams::default(),
            OperationIr::BaseFloat(BaseOperationIr::Empty(desc.clone())),
            EmptyOps::<B>::new(desc, device.clone()),
        );

        out
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(AddOps, B::float_add);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let dtype = lhs.dtype;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.broadcast(&rhs.shape).unwrap(), lhs.dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Add(desc.clone())),
            AddOps::<B>::new(desc),
        );

        out
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        scalar_float_ops!(AddOps, B::float_add_scalar);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        let dtype = lhs.dtype;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::AddScalar(desc.clone())),
            AddOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = ClampOpIr {
            tensor: tensor.into_ir(),
            min: ScalarIr::with_dtype(min, &dtype),
            max: ScalarIr::with_dtype(max, &dtype),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Clamp(desc.clone())),
            ClampOps::<B>::new(desc),
        );

        out
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(SubOps, B::float_sub);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let dtype = lhs.dtype;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.broadcast(&rhs.shape).unwrap(), lhs.dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Sub(desc.clone())),
            SubOps::<B>::new(desc),
        );

        out
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        scalar_float_ops!(SubOps, B::float_sub_scalar);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        let dtype = lhs.dtype;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone(), dtype);
        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::SubScalar(desc.clone())),
            SubOps::<B>::new(desc),
        );

        out
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(MulOps, B::float_mul);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let dtype = lhs.dtype;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.broadcast(&rhs.shape).unwrap(), lhs.dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Mul(desc.clone())),
            MulOps::<B>::new(desc),
        );

        out
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        scalar_float_ops!(MulOps, B::float_mul_scalar);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        let dtype = lhs.dtype;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::MulScalar(desc.clone())),
            MulOps::<B>::new(desc),
        );

        out
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(DivOps, B::float_div);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let dtype = lhs.dtype;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.broadcast(&rhs.shape).unwrap(), lhs.dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Div(desc.clone())),
            DivOps::<B>::new(desc),
        );

        out
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        scalar_float_ops!(DivOps, B::float_div_scalar);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        let dtype = lhs.dtype;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::DivScalar(desc.clone())),
            DivOps::<B>::new(desc),
        );

        out
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(ModOps, B::float_remainder);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let dtype = lhs.dtype;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.broadcast(&rhs.shape).unwrap(), lhs.dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Rem(desc.clone())),
            ModOps::<B>::new(desc),
        );

        out
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> FloatTensor<Self> {
        scalar_float_ops!(ModOps, B::float_remainder_scalar);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        let dtype = lhs.dtype;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::RemScalar(desc.clone())),
            ModOps::<B>::new(desc),
        );

        out
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(MatmulOps, B::float_matmul);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let dtype = lhs.dtype;
        let shape = Shape::matmul(&lhs.shape, &rhs.shape).unwrap();

        let out = lhs.client.tensor_uninitialized(shape, dtype);
        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Matmul(desc.clone())),
            MatmulOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let dtype = lhs.dtype;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.broadcast(&rhs.shape).unwrap(), dtype);
        let desc = CrossOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
            dim,
        };

        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Cross(desc.clone())),
            CrossOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);

        let dtype = tensor.dtype;
        let shape = tensor.shape.clone().swap(dim1, dim2).unwrap();
        let mut out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = SwapDimsOpIr {
            input: tensor.into_ir(),
            dim1,
            dim2,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::SwapDims(desc.clone())),
            SwapDimsOps::<B>::new(desc),
        );
        out.stream = StreamId::current();

        out
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        if tensor.shape == shape {
            return tensor;
        }

        #[derive(new, Debug)]
        struct ReshapeDimsOps<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ReshapeDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_reshape(input, self.desc.out.shape.clone());
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::Reshape(desc.clone())),
            ReshapeDimsOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        streams.tensor(&indices);

        let dtype = tensor.dtype;
        let shape = indices.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = GatherOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Gather(desc.clone())),
            GatherOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        streams.tensor(&indices);
        streams.tensor(&value);

        let shape = tensor.shape.clone();
        let dtype = tensor.dtype;
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = ScatterOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };
        // Check that both float tensors have the same type
        check_binary_op_types(&desc.tensor, &desc.value).unwrap();
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Scatter(desc.clone())),
            ScatterOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        streams.tensor(&indices);

        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = indices.shape[0];
        let out = tensor.client.tensor_uninitialized(shape, dtype);
        let desc = SelectOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Select(desc.clone())),
            SelectOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        streams.tensor(&indices);
        streams.tensor(&value);

        let dtype = tensor.dtype;
        let shape = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = SelectAssignOpIr {
            tensor: tensor.into_ir(),
            dim,
            indices: indices.into_ir(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };
        // Check that both float tensors have the same type
        check_binary_op_types(&desc.tensor, &desc.value).unwrap();
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::SelectAssign(desc.clone())),
            SelectAssignOps::<B>::new(desc),
        );

        out
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
        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let shape = tensor.shape.clone().slice(slices).unwrap();

        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = SliceOpIr {
            tensor: tensor.into_ir(),
            ranges: slices.to_vec(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::Slice(desc.clone())),
            SliceOps::<B>::new(desc),
        );

        out
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        ranges: &[burn_tensor::Slice],
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        streams.tensor(&value);

        let dtype = tensor.dtype;
        let shape = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = SliceAssignOpIr {
            tensor: tensor.into_ir(),
            ranges: ranges.to_vec(),
            value: value.into_ir(),
            out: out.to_ir_out(),
        };
        // Check that both float tensors have the same type
        check_binary_op_types(&desc.tensor, &desc.value).unwrap();
        out.client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::SliceAssign(desc.clone())),
            SliceAssignOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        streams.tensor(&mask);
        streams.tensor(&value);

        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.broadcast(&mask.shape).unwrap(), dtype);

        let desc = MaskWhereOpIr {
            tensor: tensor.into_ir(),
            value: value.into_ir(),
            mask: mask.into_ir(),
            out: out.to_ir_out(),
        };
        // Check that both float tensors have the same type
        check_binary_op_types(&desc.tensor, &desc.value).unwrap();
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::MaskWhere(desc.clone())),
            MaskWhereOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        streams.tensor(&mask);

        let dtype = tensor.dtype;
        let shape = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape, dtype);
        let desc = MaskFillOpIr {
            tensor: tensor.into_ir(),
            value: ScalarIr::with_dtype(value, &dtype),
            mask: mask.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::MaskFill(desc.clone())),
            MaskFillOps::<B>::new(desc),
        );

        out
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float_cmp_ops!(EqualOps, B::float_equal);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let out = lhs.client.tensor_uninitialized(
            lhs.shape.broadcast(&rhs.shape).unwrap(),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::Equal(desc.clone())),
            EqualOps::<B>::new(desc),
        );

        out
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        scalar_float_cmp_ops!(EqualElemOps, B::float_equal_elem);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        let dtype = lhs.dtype;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::EqualElem(desc.clone())),
            EqualElemOps::<B>::new(desc),
        );

        out
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float_cmp_ops!(GreaterOps, B::float_greater);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let dtype = lhs.dtype;
        let out = lhs.client.tensor_uninitialized(
            lhs.shape.broadcast(&rhs.shape).unwrap(),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Greater(desc.clone())),
            GreaterOps::<B>::new(desc),
        );

        out
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        scalar_float_cmp_ops!(GreaterElemOps, B::float_greater_elem);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        let dtype = lhs.dtype;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::GreaterElem(desc.clone())),
            GreaterElemOps::<B>::new(desc),
        );

        out
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float_cmp_ops!(GreaterEqualOps, B::float_greater_equal);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let dtype = lhs.dtype;
        let out = lhs.client.tensor_uninitialized(
            lhs.shape.broadcast(&rhs.shape).unwrap(),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::GreaterEqual(desc.clone())),
            GreaterEqualOps::<B>::new(desc),
        );

        out
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        scalar_float_cmp_ops!(GreaterEqualElemOps, B::float_greater_equal_elem);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        let dtype = lhs.dtype;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::GreaterEqualElem(desc.clone())),
            GreaterEqualElemOps::<B>::new(desc),
        );

        out
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float_cmp_ops!(LowerOps, B::float_lower);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let dtype = lhs.dtype;
        let out = lhs.client.tensor_uninitialized(
            lhs.shape.broadcast(&rhs.shape).unwrap(),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Lower(desc.clone())),
            LowerOps::<B>::new(desc),
        );

        out
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        scalar_float_cmp_ops!(LowerElemOps, B::float_lower_elem);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        let dtype = lhs.dtype;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::LowerElem(desc.clone())),
            LowerElemOps::<B>::new(desc),
        );

        out
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> BoolTensor<Self> {
        binary_float_cmp_ops!(LowerEqualOps, B::float_lower_equal);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let dtype = lhs.dtype;
        let out = lhs.client.tensor_uninitialized(
            lhs.shape.broadcast(&rhs.shape).unwrap(),
            B::BoolElem::dtype(),
        );

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::LowerEqual(desc.clone())),
            LowerEqualOps::<B>::new(desc),
        );

        out
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: FloatElem<Self>) -> BoolTensor<Self> {
        scalar_float_cmp_ops!(LowerEqualElemOps, B::float_lower_equal_elem);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        let dtype = lhs.dtype;
        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.clone(), B::BoolElem::dtype());

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::LowerEqualElem(desc.clone())),
            LowerEqualElemOps::<B>::new(desc),
        );

        out
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(SumOps, B::float_sum, reduce);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor.client.tensor_uninitialized(Shape::new([1]), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Sum(desc.clone())),
            SumOps::<B>::new(desc),
        );

        out
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, axis: usize) -> FloatTensor<Self> {
        reduce_float_ops!(SumDimOps, B::float_sum_dim);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[axis] = 1;
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            axis,
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::SumDim(desc.clone())),
            SumDimOps::<B>::new(desc),
        );

        out
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(ProdOps, B::float_prod, reduce);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor.client.tensor_uninitialized(Shape::new([1]), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Prod(desc.clone())),
            ProdOps::<B>::new(desc),
        );

        out
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce_float_ops!(ProdDimOps, B::float_prod_dim);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::ProdDim(desc.clone())),
            ProdDimOps::<B>::new(desc),
        );

        out
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(MeanOps, B::float_mean, reduce);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor.client.tensor_uninitialized(Shape::new([1]), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Mean(desc.clone())),
            MeanOps::<B>::new(desc),
        );

        out
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce_float_ops!(MeanDimOps, B::float_mean_dim);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::MeanDim(desc.clone())),
            MeanDimOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let shape = tensor.shape.clone();
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = DimOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            axis: dim,
        };

        out.client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::CumSum(desc.clone())),
            CumsumOps::<B>::new(desc),
        );

        out
    }

    fn float_exp(lhs: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(ExpOps, B::float_exp);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        let dtype = lhs.dtype;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: lhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Exp(desc.clone())),
            ExpOps::<B>::new(desc),
        );

        out
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(LogOps, B::float_log);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Log(desc.clone())),
            LogOps::<B>::new(desc),
        );

        out
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(Log1pOps, B::float_log1p);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Log1p(desc.clone())),
            Log1pOps::<B>::new(desc),
        );

        out
    }

    fn float_powf_scalar(lhs: FloatTensor<Self>, rhs: f32) -> FloatTensor<Self> {
        scalar_float_ops!(PowfOps, B::float_powf_scalar);

        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        let dtype = lhs.dtype;
        let out = lhs.client.tensor_uninitialized(lhs.shape.clone(), dtype);

        let desc = ScalarOpIr {
            lhs: lhs.into_ir(),
            rhs: ScalarIr::with_dtype(rhs, &dtype),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::PowfScalar(desc.clone())),
            PowfOps::<B>::new(desc),
        );

        out
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(SqrtOps, B::float_sqrt);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Sqrt(desc.clone())),
            SqrtOps::<B>::new(desc),
        );

        out
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(AbsOps, B::float_abs);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Abs(desc.clone())),
            AbsOps::<B>::new(desc),
        );

        out
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(CosOps, B::float_cos);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Cos(desc.clone())),
            CosOps::<B>::new(desc),
        );

        out
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(SinOps, B::float_sin);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Sin(desc.clone())),
            SinOps::<B>::new(desc),
        );

        out
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(TanhOps, B::float_tanh);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Tanh(desc.clone())),
            TanhOps::<B>::new(desc),
        );

        out
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(Recip, B::float_recip);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);
        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Recip(desc.clone())),
            Recip::<B>::new(desc),
        );

        out
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(TanhOps, B::float_erf);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Erf(desc.clone())),
            TanhOps::<B>::new(desc),
        );

        out
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

        let tensor_first = tensors.first().unwrap();
        let dtype = tensor_first.dtype;
        let client = tensor_first.client.clone();

        let mut streams = OperationStreams::default();
        tensors.iter().for_each(|tensor| streams.tensor(tensor));

        // Calculate the output shape
        let shape = Shape::cat(tensors.iter().map(|t| &t.shape), dim).unwrap();

        let out = client.tensor_uninitialized(shape, dtype);

        let desc = CatOpIr {
            tensors: tensors.into_iter().map(|t| t.into_ir()).collect(),
            dim,
            out: out.to_ir_out(),
        };
        desc.tensors
            .windows(2)
            .for_each(|desc| check_binary_op_types(&desc[0], &desc[1]).unwrap());
        client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::Cat(desc.clone())),
            CatOps::<B>::new(desc),
        );

        out
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce_float2int_ops!(ArgMaxOps, B::float_argmax);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::ArgMax(desc.clone())),
            ArgMaxOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let shape = tensor.shape.clone().repeat(dim, times);
        let out = tensor.client.tensor_uninitialized(shape, tensor.dtype);

        let desc = RepeatDimOpIr {
            tensor: tensor.into_ir(),
            dim,
            times,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::RepeatDim(desc.clone())),
            RepeatDimOps::<B>::new(desc),
        );

        out
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> IntTensor<Self> {
        reduce_float2int_ops!(ArgMinOps, B::float_argmin);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::ArgMin(desc.clone())),
            ArgMinOps::<B>::new(desc),
        );

        out
    }

    fn float_max(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(MaxOps, B::float_max, reduce);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor.client.tensor_uninitialized(Shape::new([1]), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Max(desc.clone())),
            MaxOps::<B>::new(desc),
        );

        out
    }

    fn float_max_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce_float_ops!(MaxDimOps, B::float_max_dim);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let mut shape = tensor.shape.clone();
        let dtype = tensor.dtype;
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::MaxDim(desc.clone())),
            MaxDimOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);

        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let dtype = tensor.dtype;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone(), dtype);
        let out_indices = client.tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ReduceDimWithIndicesOpIr {
            tensor: tensor.into_ir(),
            dim,
            out: out.to_ir_out(),
            out_indices: out_indices.to_ir_out(),
        };
        client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::MaxDimWithIndices(desc.clone())),
            MaxDimWithIndicesOps::<B>::new(desc),
        );

        (out, out_indices)
    }

    fn float_min(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(MinOps, B::float_min, reduce);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor.client.tensor_uninitialized(Shape::new([1]), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Min(desc.clone())),
            MinOps::<B>::new(desc),
        );

        out
    }

    fn float_min_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce_float_ops!(MinDimOps, B::float_min_dim);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::MinDim(desc.clone())),
            MinDimOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim] = 1;
        let client = tensor.client.clone();
        let out = client.tensor_uninitialized(shape.clone(), dtype);
        let out_indices = client.tensor_uninitialized(shape, B::IntElem::dtype());

        let desc = ReduceDimWithIndicesOpIr {
            tensor: tensor.into_ir(),
            dim,
            out: out.to_ir_out(),
            out_indices: out_indices.to_ir_out(),
        };
        client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::MinDimWithIndices(desc.clone())),
            MinDimWithIndicesOps::<B>::new(desc),
        );

        (out, out_indices)
    }

    fn float_max_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(MaxAbsOps, B::float_max_abs, reduce);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor.client.tensor_uninitialized(Shape::new([1]), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::MaxAbs(desc.clone())),
            MaxAbsOps::<B>::new(desc),
        );

        out
    }

    fn float_max_abs_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        reduce_float_ops!(MaxAbsDimOps, B::float_max_abs_dim);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let mut shape = tensor.shape.clone();
        let dtype = tensor.dtype;
        shape[dim] = 1;
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = ReduceDimOpIr {
            input: tensor.into_ir(),
            axis: dim,
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::MaxAbsDim(desc.clone())),
            MaxAbsDimOps::<B>::new(desc),
        );

        out
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        binary_float_ops!(PowOps, B::float_powf);
        let mut streams = OperationStreams::default();
        streams.tensor(&lhs);
        streams.tensor(&rhs);
        let dtype = lhs.dtype;

        let out = lhs
            .client
            .tensor_uninitialized(lhs.shape.broadcast(&rhs.shape).unwrap(), dtype);

        let desc = BinaryOpIr {
            lhs: lhs.into_ir(),
            rhs: rhs.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::NumericFloat(dtype, NumericOperationIr::Powf(desc.clone())),
            PowOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);

        // Change the shape of the tensor to match the new axes
        let shape = tensor.shape.clone().permute(axes).unwrap();
        let out = tensor.client.tensor_uninitialized(shape, tensor.dtype);

        let desc = PermuteOpIr {
            input: tensor.into_ir(),
            axes: axes.to_vec(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::BaseInt(BaseOperationIr::Permute(desc.clone())),
            PermuteDimsOps::<B>::new(desc),
        );

        out
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct ExpandOps<B: FusionBackend> {
            desc: ExpandOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ExpandOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_float_tensor::<B>(&self.desc.input);
                let output = B::float_expand(input, self.desc.shape.clone());

                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);

        let out = tensor
            .client
            .tensor_uninitialized(shape.clone(), tensor.dtype);

        let desc = ExpandOpIr {
            input: tensor.into_ir(),
            shape,
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::Expand(desc.clone())),
            ExpandOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), tensor.dtype);

        let desc = FlipOpIr {
            input: tensor.into_ir(),
            axes: axes.to_vec(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::BaseInt(BaseOperationIr::Flip(desc.clone())),
            FlipOps::<B>::new(desc),
        );

        out
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(RoundOps, B::float_round);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Round(desc.clone())),
            RoundOps::<B>::new(desc),
        );

        out
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(FloorOps, B::float_floor);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Floor(desc.clone())),
            FloorOps::<B>::new(desc),
        );

        out
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(CeilOps, B::float_ceil);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Ceil(desc.clone())),
            CeilOps::<B>::new(desc),
        );

        out
    }

    fn float_trunc(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        unary_float_ops!(TruncOps, B::float_ceil);

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype);

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Trunc(desc.clone())),
            TruncOps::<B>::new(desc),
        );

        out
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: burn_tensor::FloatDType) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct CastOps<B: FusionBackend> {
            desc: UnaryOpIr,
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), dtype.into());

        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::Cast(desc.clone())),
            CastOps::<B>::new(desc, dtype),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);

        let shape = calculate_unfold_shape(tensor.shape(), dim, size, step);

        let out = tensor
            .client
            .tensor_uninitialized(Shape::from(shape), tensor.dtype);

        let desc = UnfoldOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
            dim,
            size,
            step,
        };

        out.client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::Unfold(desc.clone())),
            UnfoldOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::BoolElem::dtype());

        let dtype = tensor.dtype;
        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::IsNan(desc.clone())),
            IsNanOps::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let out = tensor
            .client
            .tensor_uninitialized(tensor.shape.clone(), B::BoolElem::dtype());

        let dtype = tensor.dtype;
        let desc = UnaryOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::IsInf(desc.clone())),
            IsInfOps::<B>::new(desc),
        );

        out
    }
}
