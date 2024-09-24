use crate::{client::FusionClient, stream::StreamId, Client, Fusion, FusionBackend, FusionRuntime};
use burn_tensor::{
    quantization::{
        QTensorPrimitive, QuantizationParametersPrimitive, QuantizationScheme, QuantizationStrategy,
    },
    repr::{
        QuantizationParametersDescription, QuantizedTensorDescription, TensorDescription, TensorId,
        TensorStatus,
    },
    DType, Shape, TensorData,
};
use std::sync::Arc;

/// Tensor primitive for the [fusion backend](crate::FusionBackend) for all kind.
pub struct FusionTensor<R: FusionRuntime> {
    /// Tensor id.
    pub id: Arc<TensorId>,
    /// The shape of the tensor.
    pub shape: Vec<usize>,
    /// The [fusion client](FusionClient).
    pub client: Client<R>,
    /// The datatype of the tensor.
    pub dtype: DType,
    // Orphan means that a tensor is never converted into a description when it becomes `ReadWrite`.
    //
    // When a tensor is dropped and is still an orphan, we need to register it as such to avoid
    // memory leak. Otherwise, the cleanup is going to happen during a graph execution.
    pub(crate) is_orphan: bool,
    pub(crate) stream: StreamId,
}

impl<R: FusionRuntime> Clone for FusionTensor<R> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            shape: self.shape.clone(),
            client: self.client.clone(),
            dtype: self.dtype,
            is_orphan: self.is_orphan,
            stream: self.stream,
        }
    }
}

impl<R: FusionRuntime> core::fmt::Debug for FusionTensor<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            format!(
                "{{ id: {:?}, shape: {:?}, should_drop: {:?}, device: {:?} }}",
                self.id,
                self.shape,
                self.is_orphan,
                self.client.device().clone(),
            )
            .as_str(),
        )
    }
}

impl<R: FusionRuntime> FusionTensor<R> {
    pub(crate) fn new(
        id: Arc<TensorId>,
        shape: Vec<usize>,
        dtype: DType,
        client: Client<R>,
        stream: StreamId,
    ) -> Self {
        Self {
            id,
            shape,
            client,
            dtype,
            is_orphan: true,
            stream,
        }
    }
    pub(crate) fn shape(&self) -> Shape {
        Shape::from(self.shape.clone())
    }

    fn status(&self) -> TensorStatus {
        if Arc::strong_count(&self.id) <= 1 {
            TensorStatus::ReadWrite
        } else {
            TensorStatus::ReadOnly
        }
    }

    /// Description to be used when using an uninitialized tensor as output.
    pub(crate) fn to_description_out(&self) -> TensorDescription {
        TensorDescription {
            status: TensorStatus::NotInit,
            shape: self.shape.clone(),
            id: *self.id.as_ref(),
            dtype: self.dtype,
        }
    }

    /// Description to be used when using an initialized tensor used as input.
    pub(crate) fn into_description(mut self) -> TensorDescription {
        let status = self.status();
        let mut shape_out = Vec::new();
        core::mem::swap(&mut self.shape, &mut shape_out);

        if let TensorStatus::ReadWrite = status {
            self.is_orphan = false;
        }

        TensorDescription {
            status,
            shape: shape_out,
            id: *self.id.as_ref(),
            dtype: self.dtype,
        }
    }

    pub(crate) async fn into_data<B>(self) -> TensorData
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let id = self.stream;
        self.client
            .clone()
            .read_tensor_float::<B>(self.into_description(), id)
            .await
    }

    pub(crate) async fn int_into_data<B>(self) -> TensorData
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let id = self.stream;
        self.client
            .clone()
            .read_tensor_int::<B>(self.into_description(), id)
            .await
    }

    pub(crate) async fn bool_into_data<B>(self) -> TensorData
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let id = self.stream;
        self.client
            .clone()
            .read_tensor_bool::<B>(self.into_description(), id)
            .await
    }
}

impl<R: FusionRuntime> Drop for FusionTensor<R> {
    fn drop(&mut self) {
        if !self.is_orphan {
            return;
        }

        match self.status() {
            TensorStatus::ReadWrite => {
                self.client.register_orphan(&self.id);
            }
            TensorStatus::ReadOnly => {}
            TensorStatus::NotInit => {}
        }
    }
}

/// A quantized tensor primitive for fusion backends.
#[derive(Debug)]
pub struct QFusionTensor<R: FusionRuntime> {
    /// The quantized tensor.
    pub qtensor: FusionTensor<R>,
    /// The quantization scheme.
    pub scheme: QuantizationScheme,
    /// The quantization parameters.
    pub qparams: FusionQuantizationParameters<R>,
}

impl<R: FusionRuntime> QTensorPrimitive for QFusionTensor<R> {
    fn scheme(&self) -> &QuantizationScheme {
        &self.scheme
    }

    fn strategy(&self) -> QuantizationStrategy {
        // TODO
        todo!()
    }
}

impl<R: FusionRuntime> Clone for QFusionTensor<R> {
    fn clone(&self) -> Self {
        Self {
            qtensor: self.qtensor.clone(),
            scheme: self.scheme.clone(),
            qparams: self.qparams.clone(),
        }
    }
}

impl<R: FusionRuntime> QFusionTensor<R> {
    pub(crate) async fn into_data<B>(self) -> TensorData
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let streams = if let Some(offset) = &self.qparams.offset {
            vec![
                self.qtensor.stream,
                self.qparams.scale.stream,
                offset.stream,
            ]
        } else {
            vec![self.qtensor.stream, self.qparams.scale.stream]
        };

        // Quantized tensor and qparams tensors client are the same
        self.qtensor
            .client
            .clone()
            .read_tensor_quantized::<B>(self.into_description(), streams)
            .await
    }

    /// Description to be used when using an initialized tensor used as input.
    pub(crate) fn into_description(self) -> QuantizedTensorDescription {
        QuantizedTensorDescription {
            tensor: self.qtensor.into_description(),
            qparams: QuantizationParametersDescription {
                scale: self.qparams.scale.into_description(),
                offset: self.qparams.offset.map(|x| x.into_description()),
            },
            scheme: self.scheme,
        }
    }
}

/// The quantization parameters.
#[derive(Debug)]
pub struct FusionQuantizationParameters<R: FusionRuntime> {
    /// The scaling factor.
    pub scale: FusionTensor<R>,
    /// The zero-point offset.
    pub offset: Option<FusionTensor<R>>,
}

impl<R: FusionRuntime> Clone for FusionQuantizationParameters<R> {
    fn clone(&self) -> Self {
        Self {
            scale: self.scale.clone(),
            offset: self.offset.clone(),
        }
    }
}

impl<B: FusionBackend> From<QuantizationParametersPrimitive<Fusion<B>>>
    for FusionQuantizationParameters<B::FusionRuntime>
{
    fn from(value: QuantizationParametersPrimitive<Fusion<B>>) -> Self {
        FusionQuantizationParameters {
            scale: value.scale,
            offset: value.offset,
        }
    }
}
