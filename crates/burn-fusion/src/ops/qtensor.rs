use std::marker::PhantomData;

use burn_ir::{
    BaseOperationIr, DequantizeOpIr, FlipOpIr, FloatOperationIr, GatherOpIr, HandleContainer,
    InitOperationIr, MatmulOpIr, NumericOperationIr, OperationIr, OperationOutput, PermuteOpIr,
    QuantizationParametersIr, QuantizeOpIr, SelectOpIr, ShapeOpIr, SliceOpIr, SwapDimsOpIr,
};
use burn_tensor::{
    DType, Device, Element, Shape, Slice, TensorData, TensorPrimitive,
    backend::DeferedError,
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{
        QTensorPrimitive, QuantPropagation, QuantScheme, QuantizationParametersPrimitive,
    },
};

use crate::{
    Fusion, FusionBackend, get_client,
    stream::{OperationStreams, execution::Operation},
};

use super::NoOp;

impl<B: FusionBackend> QTensorOps<Self> for Fusion<B> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        let client = get_client::<B>(device);
        let dtype = data.dtype;
        let tensor = B::q_from_data(data, device);
        let shape = burn_tensor::TensorMetadata::shape(&tensor);

        let handle = B::quantized_tensor_handle(tensor);
        let desc = InitOperationIr::create(shape, dtype, || client.register_tensor_handle(handle));

        client
            .register(
                OperationStreams::default(),
                OperationIr::Init(desc),
                NoOp::<B>::new(),
            )
            .output()
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        #[derive(new, Debug)]
        struct QuantizeOp<B: FusionBackend> {
            desc: QuantizeOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for QuantizeOp<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let scales = handles.get_float_tensor::<B>(&self.desc.qparams.scales);

                let qparams = QuantizationParametersPrimitive { scales };
                let output = B::quantize(tensor, &self.desc.scheme, qparams);
                handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor, &qparams.scales]);

        let client = tensor.client.clone();
        let qparams = QuantizationParametersIr {
            scales: qparams.scales.into_ir(),
        };
        let desc = QuantizeOpIr::create(tensor.into_ir(), qparams, *scheme, || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::Float(desc.tensor.dtype, FloatOperationIr::Quantize(desc.clone())),
                QuantizeOp::<B>::new(desc),
            )
            .output()
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        #[derive(new, Debug)]
        struct DequantizeOp<B: FusionBackend> {
            desc: DequantizeOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for DequantizeOp<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_quantized_tensor::<B>(&self.desc.input);

                let output = B::dequantize(tensor);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let dtype = B::FloatElem::dtype();
        let desc = DequantizeOpIr::create(tensor.into_ir(), dtype, || client.create_empty_handle());

        client
            .register(
                streams,
                OperationIr::Float(dtype, FloatOperationIr::Dequantize(desc.clone())),
                DequantizeOp::<B>::new(desc),
            )
            .output()
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        tensor.client.device().clone()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        let device_original: &B::Device = tensor.client.device();
        let device_target: B::Device = device.clone();

        if device_original == &device_target {
            return tensor;
        }

        let id = tensor.stream;
        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.client.clone();

        client_original.change_client_quantized::<B>(tensor.into_ir(), client_target, id)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
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
                let input = handles.get_quantized_tensor::<B>(&self.desc.input);
                let output = B::q_reshape(input, self.desc.out.shape.clone());
                handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
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

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, DeferedError> {
        tensor.q_into_data::<B>().await
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        #[derive(new, Debug)]
        struct SwapDimsOps<B: FusionBackend> {
            desc: SwapDimsOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SwapDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_quantized_tensor::<B>(&self.desc.input);
                let output = B::q_swap_dims(input, self.desc.dim1, self.desc.dim2);
                handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
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

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        #[derive(new, Debug)]
        struct PermuteDimsOps<B: FusionBackend> {
            desc: PermuteOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for PermuteDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_quantized_tensor::<B>(&self.desc.input);
                let output = B::q_permute(input, self.desc.axes.as_slice());
                handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
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
                OperationIr::BaseFloat(BaseOperationIr::Permute(desc.clone())),
                PermuteDimsOps::<B>::new(desc),
            )
            .output()
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        #[derive(new, Debug)]
        struct FlipOps<B: FusionBackend> {
            desc: FlipOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for FlipOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_quantized_tensor::<B>(&self.desc.input);
                let output = B::q_flip(input, &self.desc.axes);
                handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
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
                OperationIr::BaseFloat(BaseOperationIr::Flip(desc.clone())),
                FlipOps::<B>::new(desc),
            )
            .output()
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        #[derive(new, Debug)]
        struct GatherOps<B: FusionBackend> {
            desc: GatherOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for GatherOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_quantized_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);

                let output = B::q_gather(self.desc.dim, tensor, indices);
                handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = GatherOpIr::create(tensor.into_ir(), dim, indices.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.tensor.dtype,
                    NumericOperationIr::Gather(desc.clone()),
                ),
                GatherOps::<B>::new(desc),
            )
            .output()
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        #[derive(new, Debug)]
        struct SelectOps<B: FusionBackend> {
            desc: SelectOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SelectOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_quantized_tensor::<B>(&self.desc.tensor);
                let indices = handles.get_int_tensor::<B>(&self.desc.indices);

                let output = B::q_select(tensor, self.desc.dim, indices);

                handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let streams = OperationStreams::with_inputs([&tensor]);

        let client = tensor.client.clone();
        let desc = SelectOpIr::create(tensor.into_ir(), dim, indices.into_ir(), || {
            client.create_empty_handle()
        });

        client
            .register(
                streams,
                OperationIr::NumericFloat(
                    desc.tensor.dtype,
                    NumericOperationIr::Select(desc.clone()),
                ),
                SelectOps::<B>::new(desc),
            )
            .output()
    }

    fn q_slice(tensor: QuantizedTensor<Self>, slices: &[Slice]) -> QuantizedTensor<Self> {
        #[derive(new, Debug)]
        struct SliceOps<B: FusionBackend> {
            desc: SliceOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for SliceOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_quantized_tensor::<B>(&self.desc.tensor);

                let output = B::q_slice(tensor, self.desc.ranges.as_slice());

                handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
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

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        #[derive(new, Debug)]
        struct ExpandOps<B: FusionBackend> {
            desc: ShapeOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ExpandOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_quantized_tensor::<B>(&self.desc.input);
                let output = B::q_expand(input, self.desc.out.shape.clone());

                handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
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

    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        #[derive(new, Debug)]
        struct MatmulOps<B: FusionBackend> {
            desc: MatmulOpIr,
            lhs_quantized: bool,
            rhs_quantized: bool,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for MatmulOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let lhs = match self.lhs_quantized {
                    true => {
                        TensorPrimitive::QFloat(handles.get_quantized_tensor::<B>(&self.desc.lhs))
                    }
                    false => TensorPrimitive::Float(handles.get_float_tensor::<B>(&self.desc.lhs)),
                };
                let rhs = match self.rhs_quantized {
                    true => {
                        TensorPrimitive::QFloat(handles.get_quantized_tensor::<B>(&self.desc.rhs))
                    }
                    false => TensorPrimitive::Float(handles.get_float_tensor::<B>(&self.desc.rhs)),
                };
                let output = B::q_matmul(lhs, rhs);
                match output {
                    TensorPrimitive::Float(output) => {
                        handles.register_float_tensor::<B>(&self.desc.out.id, output);
                    }
                    TensorPrimitive::QFloat(output) => {
                        handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
                    }
                }
            }
        }

        let mut propagation = QuantPropagation::Inhibit;
        let mut scheme = QuantScheme::default();
        let mut streams = OperationStreams::default();
        let mut lhs_quantized = false;
        let mut rhs_quantized = false;
        match &lhs {
            TensorPrimitive::QFloat(lhs) => {
                propagation = lhs.propagation();
                scheme = *lhs.scheme();
                lhs_quantized = true;
                streams.tensor(lhs);
            }
            TensorPrimitive::Float(lhs) => {
                streams.tensor(lhs);
            }
        }
        match &rhs {
            TensorPrimitive::QFloat(rhs) => {
                propagation = rhs.propagation();
                scheme = *rhs.scheme();
                rhs_quantized = true;
                streams.tensor(rhs);
            }
            TensorPrimitive::Float(rhs) => {
                streams.tensor(rhs);
            }
        }

        let dtype = match propagation {
            QuantPropagation::Propagate => DType::QFloat(scheme),
            QuantPropagation::Inhibit => B::FloatElem::dtype(),
        };

        let client = match &lhs {
            TensorPrimitive::Float(lhs) => lhs.client.clone(),
            TensorPrimitive::QFloat(lhs) => lhs.client.clone(),
        };

        let lhs = match lhs {
            TensorPrimitive::Float(lhs) => lhs.into_ir(),
            TensorPrimitive::QFloat(lhs) => lhs.into_ir(),
        };
        let rhs = match rhs {
            TensorPrimitive::Float(rhs) => rhs.into_ir(),
            TensorPrimitive::QFloat(rhs) => rhs.into_ir(),
        };

        let desc = MatmulOpIr::create_mixed(lhs, rhs, dtype, || client.create_empty_handle());

        let out = client
            .register(
                streams,
                OperationIr::Float(dtype, FloatOperationIr::Matmul(desc.clone())),
                MatmulOps::<B>::new(desc, lhs_quantized, rhs_quantized),
            )
            .output();

        match propagation {
            QuantPropagation::Propagate => TensorPrimitive::QFloat(out),
            QuantPropagation::Inhibit => TensorPrimitive::Float(out),
        }
    }
}
