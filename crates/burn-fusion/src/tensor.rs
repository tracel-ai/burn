use crate::{
    Client, FusionBackend, FusionRuntime,
    client::FusionClient,
    stream::{Operation, OperationStreams, StreamId},
};
use burn_ir::{OperationIr, TensorId, TensorIr, TensorStatus};
use burn_tensor::{
    DType, Shape, TensorData, TensorMetadata,
    quantization::{QTensorPrimitive, QuantScheme},
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
    /// The current stream id this tensor is on.
    pub stream: StreamId,
    // Orphan means that a tensor is never converted into a representation when it becomes `ReadWrite`.
    //
    // When a tensor is dropped and is still an orphan, we need to register it as such to avoid
    // memory leak. Otherwise, the cleanup is going to happen during a graph execution.
    pub(crate) is_orphan: bool,
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

impl<R: FusionRuntime> TensorMetadata for FusionTensor<R> {
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> Shape {
        Shape::from(self.shape.clone())
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

    fn status(&self) -> TensorStatus {
        if Arc::strong_count(&self.id) <= 1 {
            TensorStatus::ReadWrite
        } else {
            TensorStatus::ReadOnly
        }
    }

    /// Intermediate representation to be used when using an uninitialized tensor as output.
    pub fn to_ir_out(&self) -> TensorIr {
        TensorIr {
            status: TensorStatus::NotInit,
            shape: self.shape.clone(),
            id: *self.id.as_ref(),
            dtype: self.dtype,
        }
    }

    /// Intermediate representation to be used when using an initialized tensor used as input.
    pub fn into_ir(mut self) -> TensorIr {
        let status = self.status();
        let mut shape_out = Vec::new();
        core::mem::swap(&mut self.shape, &mut shape_out);

        if let TensorStatus::ReadWrite = status {
            self.is_orphan = false;
        }

        TensorIr {
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
        let client = self.client.clone();
        let desc = self.into_ir();
        client.read_tensor_float::<B>(desc, id).await
    }

    pub(crate) async fn q_into_data<B>(self) -> TensorData
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        if let DType::QFloat(_scheme) = self.dtype {
            let id = self.stream;
            let client = self.client.clone();
            let desc = self.into_ir();
            client.read_tensor_quantized::<B>(desc, id).await
        } else {
            panic!("Expected quantized float dtype, got {:?}", self.dtype)
        }
    }

    pub(crate) async fn int_into_data<B>(self) -> TensorData
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let id = self.stream;
        let client = self.client.clone();
        let desc = self.into_ir();
        client.read_tensor_int::<B>(desc, id).await
    }

    pub(crate) async fn bool_into_data<B>(self) -> TensorData
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let id = self.stream;
        let client = self.client.clone();
        let desc = self.into_ir();
        client.read_tensor_bool::<B>(desc, id).await
    }
}

pub(crate) struct DropOp {
    pub(crate) id: TensorId,
}

impl<RO: FusionRuntime> Operation<RO> for DropOp {
    fn execute(&self, handles: &mut burn_ir::HandleContainer<RO::FusionHandle>) {
        handles.remove_handle(self.id);
    }
}

impl<R: FusionRuntime> Drop for FusionTensor<R> {
    fn drop(&mut self) {
        if !self.is_orphan {
            return;
        }
        self.is_orphan = false;

        match self.status() {
            TensorStatus::ReadWrite => {
                println!("Droping {self:?}");
                let id = *self.id.as_ref();
                let mut shape = Vec::new();
                core::mem::swap(&mut shape, &mut self.shape);

                let ir = TensorIr {
                    id,
                    shape,
                    status: TensorStatus::ReadWrite,
                    dtype: self.dtype,
                };
                let mut streams = OperationStreams::default();
                streams.tensor(&self);

                self.client
                    .register(streams, OperationIr::Drop(ir), DropOp { id });
            }
            TensorStatus::ReadOnly => {
                println!("Cant drop readonly {self:?}");
            }
            TensorStatus::NotInit => {
                println!("Cant drop noinit {self:?}");
            }
        }
    }
}

impl<R: FusionRuntime> QTensorPrimitive for FusionTensor<R> {
    fn scheme(&self) -> &QuantScheme {
        if let DType::QFloat(scheme) = &self.dtype {
            scheme
        } else {
            panic!(
                "Quantization scheme is not valid for dtype {:?}",
                self.dtype,
            )
        }
    }
}
