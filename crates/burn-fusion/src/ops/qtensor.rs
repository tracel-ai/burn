use std::{marker::PhantomData, ops::Range};

use burn_ir::{
    BaseOperationIr, DequantizeOpIr, ExpandOpIr, FlipOpIr, FloatOperationIr, GatherOpIr,
    HandleContainer, InitOperationIr, NumericOperationIr, OperationIr, PermuteOpIr,
    QuantizationParametersIr, QuantizeOpIr, SelectOpIr, SliceOpIr, SwapDimsOpIr, UnaryOpIr,
};
use burn_tensor::{
    DType, Device, Element, Shape, TensorData, TensorMetadata,
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{QuantScheme, QuantizationParametersPrimitive},
};

use crate::{
    Fusion, FusionBackend,
    client::FusionClient,
    get_client,
    stream::{OperationStreams, StreamId, execution::Operation},
};

use super::NoOp;

impl<B: FusionBackend> QTensorOps<Self> for Fusion<B> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        let stream = StreamId::current();
        let client = get_client::<B>(&device.clone());
        let dtype = data.dtype;
        let tensor = B::q_from_data(data, device);
        let shape = tensor.shape();

        let handle = B::quantized_tensor_handle(tensor);
        let out = client.register_tensor(handle, shape.dims, stream, dtype);
        let desc = out.to_ir_out();

        client.register(
            OperationStreams::default(),
            OperationIr::Init(InitOperationIr { out: desc }),
            NoOp::<B>::new(),
        );

        out
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

        let shape: Vec<usize> = tensor.shape.clone();
        let dtype = tensor.dtype;
        let out = tensor
            .client
            .tensor_uninitialized(shape, DType::QFloat(*scheme));

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        streams.tensor(&qparams.scales);

        let desc = QuantizeOpIr {
            tensor: tensor.into_ir(),
            qparams: QuantizationParametersIr {
                scales: qparams.scales.clone().into_ir(),
            },
            scheme: *scheme,
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Quantize(desc.clone())),
            QuantizeOp::<B>::new(desc),
        );

        out
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);

        let shape: Vec<usize> = tensor.shape.clone();
        let dtype = B::FloatElem::dtype();
        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = DequantizeOpIr {
            input: tensor.into_ir(),
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::Float(dtype, FloatOperationIr::Dequantize(desc.clone())),
            DequantizeOp::<B>::new(desc),
        );

        out
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
        if tensor.shape == shape.dims {
            return tensor;
        }

        #[derive(new, Debug)]
        struct ReshapeDimsOps<B: FusionBackend> {
            desc: UnaryOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ReshapeDimsOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_quantized_tensor::<B>(&self.desc.input);
                let output = B::q_reshape(input, Shape::from(&self.desc.out.shape));
                handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);

        let dtype = tensor.dtype;
        let out = tensor.client.tensor_uninitialized(shape.dims, dtype);

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

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> TensorData {
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);

        let dtype = tensor.dtype;
        let mut shape = tensor.shape.clone();
        shape[dim1] = tensor.shape[dim2];
        shape[dim2] = tensor.shape[dim1];

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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);

        // Change the shape of the tensor to match the new axes
        let shape = axes.iter().map(|x| tensor.shape[*x]).collect();

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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        streams.tensor(&indices);

        let dtype = tensor.dtype;
        let shape: Vec<usize> = indices.shape.clone();
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

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        streams.tensor(&indices);

        let dtype = tensor.dtype;
        let mut shape: Vec<usize> = tensor.shape.clone();
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

    fn q_slice(tensor: QuantizedTensor<Self>, ranges: &[Range<usize>]) -> QuantizedTensor<Self> {
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
        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);
        let dtype = tensor.dtype;
        let ndims = tensor.shape().num_dims();
        let mut shape: Vec<usize> = ranges.iter().map(|range| range.end - range.start).collect();

        for i in shape.len()..ndims {
            shape.push(tensor.shape[i]);
        }

        let out = tensor.client.tensor_uninitialized(shape, dtype);

        let desc = SliceOpIr {
            tensor: tensor.into_ir(),
            ranges: ranges.into(),
            out: out.to_ir_out(),
        };
        out.client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::Slice(desc.clone())),
            SliceOps::<B>::new(desc),
        );

        out
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        #[derive(new, Debug)]
        struct ExpandOps<B: FusionBackend> {
            desc: ExpandOpIr,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for ExpandOps<B> {
            fn execute(&self, handles: &mut HandleContainer<B::Handle>) {
                let input = handles.get_quantized_tensor::<B>(&self.desc.input);
                let output = B::q_expand(input, self.desc.shape.clone().into());

                handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let mut streams = OperationStreams::default();
        streams.tensor(&tensor);

        let out = tensor
            .client
            .tensor_uninitialized(shape.dims.clone(), tensor.dtype);

        let desc = ExpandOpIr {
            input: tensor.into_ir(),
            shape: shape.dims,
            out: out.to_ir_out(),
        };

        out.client.register(
            streams,
            OperationIr::BaseFloat(BaseOperationIr::Expand(desc.clone())),
            ExpandOps::<B>::new(desc),
        );

        out
    }
}
