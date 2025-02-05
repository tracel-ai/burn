use std::{marker::PhantomData, ops::Range};

use burn_tensor::{
    ops::{FloatElem, FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{QuantizationParametersPrimitive, QuantizationScheme},
    repr::{
        DequantizeOperationDescription, FloatOperationDescription, HandleContainer,
        InitOperationDescription, OperationDescription, QuantizationParametersDescription,
        QuantizeOperationDescription,
    },
    DType, Device, Element, Shape, TensorData, TensorMetadata,
};

use crate::{
    client::FusionClient,
    get_client,
    stream::{execution::Operation, StreamId},
    Fusion, FusionBackend,
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
        let desc = out.to_description_out();

        client.register(
            vec![stream],
            OperationDescription::Init(InitOperationDescription { out: desc }),
            NoOp::<B>::new(),
        );

        out
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantizationScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        #[derive(new)]
        struct QuantizeOp<B: FusionBackend> {
            desc: QuantizeOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for QuantizeOp<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_float_tensor::<B>(&self.desc.tensor);
                let scale = handles.get_float_tensor::<B>(&self.desc.qparams.scale);
                let offset = self
                    .desc
                    .qparams
                    .offset
                    .as_ref()
                    .map(|x| handles.get_int_tensor::<B>(x));

                let qparams = QuantizationParametersPrimitive { scale, offset };
                let output = B::quantize(tensor, &self.desc.scheme, qparams);
                handles.register_quantized_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, DType::QFloat(*scheme));

        let streams = if let Some(offset) = &qparams.offset {
            vec![tensor.stream, qparams.scale.stream, offset.stream]
        } else {
            vec![tensor.stream, qparams.scale.stream]
        };

        let desc = QuantizeOperationDescription {
            tensor: tensor.into_description(),
            qparams: QuantizationParametersDescription {
                scale: qparams.scale.clone().into_description(),
                offset: qparams.offset.clone().map(|x| x.into_description()),
            },
            scheme: *scheme,
            out: out.to_description_out(),
        };

        out.client.register(
            streams,
            OperationDescription::Float(
                FloatElem::<Self>::dtype(),
                FloatOperationDescription::Quantize(desc.clone()),
            ),
            QuantizeOp::<B>::new(desc),
        );

        out
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        #[derive(new)]
        struct DequantizeOp<B: FusionBackend> {
            desc: DequantizeOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for DequantizeOp<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_quantized_tensor::<B>(&self.desc.input);

                let output = B::dequantize(tensor);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let stream = tensor.stream;
        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

        let desc = DequantizeOperationDescription {
            input: tensor.into_description(),
            out: out.to_description_out(),
        };

        out.client.register(
            vec![stream],
            OperationDescription::Float(
                FloatElem::<Self>::dtype(),
                FloatOperationDescription::Dequantize(desc.clone()),
            ),
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

        client_original.change_client_quantized::<B>(tensor.into_description(), client_target, id)
    }

    fn q_reshape(_tensor: QuantizedTensor<Self>, _shape: Shape) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> TensorData {
        tensor.q_into_data::<B>().await
    }

    fn q_swap_dims(
        _tensor: QuantizedTensor<Self>,
        _dim1: usize,
        _dim2: usize,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_permute(_tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_flip(_tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_gather(
        _dim: usize,
        _tensor: QuantizedTensor<Self>,
        _indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_select(
        _tensor: QuantizedTensor<Self>,
        _dim: usize,
        _indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_slice(_tensor: QuantizedTensor<Self>, _ranges: &[Range<usize>]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_expand(_tensor: QuantizedTensor<Self>, _shape: Shape) -> QuantizedTensor<Self> {
        unimplemented!()
    }
}
