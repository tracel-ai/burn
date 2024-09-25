use std::{marker::PhantomData, ops::Range};

use burn_tensor::{
    ops::{FloatElem, FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{QuantizationParametersPrimitive, QuantizationScheme, QuantizationStrategy},
    repr::{
        DequantizeOperationDescription, FloatOperationDescription, HandleContainer,
        OperationDescription, QuantizationParametersDescription, QuantizeOperationDescription,
    },
    DType, Device, Element, Shape, TensorData,
};

use crate::{
    client::FusionClient,
    get_client,
    stream::{execution::Operation, StreamId},
    Fusion, FusionBackend, FusionQuantizationParameters, QFusionTensor,
};

impl<B: FusionBackend> QTensorOps<Self> for Fusion<B> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        match data.dtype {
            DType::QFloat(strategy) => {
                let client = get_client::<B>(device);
                let tensor = B::q_from_data(data, device);
                let shape = B::q_shape(&tensor);

                let mut handles = B::quantized_tensor_handle(tensor);
                let qparams = match strategy {
                    QuantizationStrategy::PerTensorAffineInt8(_) => {
                        let num_handles = handles.len();
                        assert_eq!(
                            num_handles, 3,
                            "Expected 3 handles for quantized tensor, got {num_handles}"
                        );
                        let offset = handles.pop().unwrap();
                        let scale = handles.pop().unwrap();
                        FusionQuantizationParameters {
                            scale: client.register_tensor(
                                scale,
                                vec![1],
                                StreamId::current(),
                                B::FloatElem::dtype(),
                            ),
                            offset: Some(client.register_tensor(
                                offset,
                                vec![1],
                                StreamId::current(),
                                B::IntElem::dtype(),
                            )),
                        }
                    }
                    QuantizationStrategy::PerTensorSymmetricInt8(_) => {
                        let num_handles = handles.len();
                        assert_eq!(
                            num_handles, 2,
                            "Expected 2 handles for quantized tensor, got {num_handles}"
                        );
                        let scale = handles.pop().unwrap();
                        FusionQuantizationParameters {
                            scale: client.register_tensor(
                                scale,
                                vec![1],
                                StreamId::current(),
                                B::FloatElem::dtype(),
                            ),
                            offset: None,
                        }
                    }
                };
                let qtensor = client.register_tensor(
                    handles.pop().unwrap(),
                    shape.dims,
                    StreamId::current(),
                    B::QuantizedEncoding::dtype(),
                );
                QFusionTensor {
                    qtensor,
                    qparams,
                    scheme: strategy.scheme(),
                }
            }
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        }
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
                if let Some(offset) = &self.desc.qparams.offset {
                    handles.register_quantized_tensor::<B>(
                        &[&self.desc.out.id, &self.desc.qparams.scale.id, &offset.id],
                        output,
                    );
                } else {
                    handles.register_quantized_tensor::<B>(
                        &[&self.desc.out.id, &self.desc.qparams.scale.id],
                        output,
                    );
                }
            }
        }

        let shape: Vec<usize> = tensor.shape.clone();
        let out = tensor
            .client
            .tensor_uninitialized(shape, B::QuantizedEncoding::dtype());

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
            scheme: scheme.clone(),
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

        QFusionTensor {
            qtensor: out,
            scheme: scheme.clone(),
            qparams: qparams.into(),
        }
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        #[derive(new)]
        struct DequantizeOp<B: FusionBackend> {
            desc: DequantizeOperationDescription,
            _b: PhantomData<B>,
        }

        impl<B: FusionBackend> Operation<B::FusionRuntime> for DequantizeOp<B> {
            fn execute(self: Box<Self>, handles: &mut HandleContainer<B::Handle>) {
                let tensor = handles.get_quantized_tensor::<B>(&self.desc.qtensor);

                let output = B::dequantize(tensor);
                handles.register_float_tensor::<B>(&self.desc.out.id, output);
            }
        }

        let shape: Vec<usize> = tensor.qtensor.shape.clone();
        let out = tensor
            .qtensor
            .client
            .tensor_uninitialized(shape, B::FloatElem::dtype());

        let streams = if let Some(offset) = &tensor.qparams.offset {
            vec![
                tensor.qtensor.stream,
                tensor.qparams.scale.stream,
                offset.stream,
            ]
        } else {
            vec![tensor.qtensor.stream, tensor.qparams.scale.stream]
        };

        let desc = DequantizeOperationDescription {
            qtensor: tensor.into_description(),
            out: out.to_description_out(),
        };

        out.client.register(
            streams,
            OperationDescription::Float(
                FloatElem::<Self>::dtype(),
                FloatOperationDescription::Dequantize(desc.clone()),
            ),
            DequantizeOp::<B>::new(desc),
        );

        out
    }

    fn q_shape(tensor: &QuantizedTensor<Self>) -> Shape {
        tensor.qtensor.shape()
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        tensor.qtensor.client.device().clone()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        // Quantization parameters are on the same device as the qtensor
        let device_original: &B::Device = tensor.qtensor.client.device();
        let device_target: B::Device = device.clone();

        if device_original == &device_target {
            return tensor;
        }
        println!("q_to_device {:?} {:?}", device_original, device_target);

        let client_target = get_client::<B>(&device_target);
        let client_original = tensor.qtensor.client.clone();

        let ids = if let Some(offset) = &tensor.qparams.offset {
            vec![
                tensor.qtensor.stream,
                tensor.qparams.scale.stream,
                offset.stream,
            ]
        } else {
            vec![tensor.qtensor.stream, tensor.qparams.scale.stream]
        };

        client_original.change_client_quantized::<B>(tensor.into_description(), client_target, ids)
    }

    fn q_reshape(_tensor: QuantizedTensor<Self>, _shape: Shape) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> TensorData {
        tensor.into_data::<B>().await
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
