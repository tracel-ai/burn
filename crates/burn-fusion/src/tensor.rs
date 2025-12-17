use crate::{
    Client, FusionBackend, FusionRuntime,
    stream::{Operation, OperationStreams, StreamId},
};
use burn_backend::{
    DType, ExecutionError, QTensorPrimitive, Shape, TensorData, TensorMetadata,
    quantization::QuantScheme,
};
use burn_ir::{OperationIr, TensorId, TensorIr, TensorStatus};
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

/// Tensor primitive for the [fusion backend](crate::FusionBackend) for all kind.
pub struct FusionTensor<R: FusionRuntime> {
    /// Tensor id.
    pub id: TensorId,
    /// The shape of the tensor.
    pub shape: Shape,
    /// The fusion client.
    pub client: Client<R>,
    /// The datatype of the tensor.
    pub dtype: DType,
    /// The current stream id this tensor is on.
    pub stream: StreamId,
    pub(crate) count: Arc<AtomicU32>,
}

impl<R: FusionRuntime> Clone for FusionTensor<R> {
    fn clone(&self) -> Self {
        self.count.fetch_add(1, Ordering::Relaxed);

        Self {
            id: self.id,
            shape: self.shape.clone(),
            client: self.client.clone(),
            dtype: self.dtype,
            stream: self.stream,
            count: self.count.clone(),
        }
    }
}

impl<R: FusionRuntime> core::fmt::Debug for FusionTensor<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            format!(
                "{{ id: {:?}, shape: {:?}, device: {:?} }}",
                self.id,
                self.shape,
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
        self.shape.clone()
    }

    fn rank(&self) -> usize {
        self.shape.num_dims()
    }
}

impl<R: FusionRuntime> FusionTensor<R> {
    pub(crate) fn new(
        id: TensorId,
        shape: Shape,
        dtype: DType,
        client: Client<R>,
        stream: StreamId,
    ) -> Self {
        Self {
            id,
            shape,
            client,
            dtype,
            stream,
            count: Arc::new(AtomicU32::new(1)),
        }
    }

    fn status(&self, count: u32) -> TensorStatus {
        if count <= 1 {
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
            id: self.id,
            dtype: self.dtype,
        }
    }

    /// Intermediate representation to be used when using an initialized tensor used as input.
    pub fn into_ir(mut self) -> TensorIr {
        let count = self.count.load(Ordering::Relaxed);
        let status = self.status(count);

        let mut shape_out = Shape::from(Vec::<usize>::new());
        core::mem::swap(&mut self.shape, &mut shape_out);

        if let TensorStatus::ReadWrite = status {
            // Avoids an unwanted drop on the same thread.
            //
            // Since `drop` is called after `into_ir`, we must not register a drop if the tensor
            // was consumed with a `ReadWrite` status.
            self.count.fetch_add(1, Ordering::Relaxed);
        }

        TensorIr {
            status,
            shape: shape_out,
            id: self.id,
            dtype: self.dtype,
        }
    }

    pub(crate) async fn into_data<B>(self) -> Result<TensorData, ExecutionError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let id = self.stream;
        let client = self.client.clone();
        let desc = self.into_ir();
        client.read_tensor_float::<B>(desc, id).await
    }

    pub(crate) async fn q_into_data<B>(self) -> Result<TensorData, ExecutionError>
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

    pub(crate) async fn int_into_data<B>(self) -> Result<TensorData, ExecutionError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let id = self.stream;
        let client = self.client.clone();
        let desc = self.into_ir();
        client.read_tensor_int::<B>(desc, id).await
    }

    pub(crate) async fn bool_into_data<B>(self) -> Result<TensorData, ExecutionError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let id = self.stream;
        let client = self.client.clone();
        let desc = self.into_ir();
        client.read_tensor_bool::<B>(desc, id).await
    }
}

#[derive(new, Debug)]
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
        let count = self.count.fetch_sub(1, Ordering::Relaxed);

        // Workaround to prevent segfaults when an operation panics
        if std::thread::panicking() {
            return;
        }

        match self.status(count) {
            TensorStatus::ReadWrite => {
                let mut shape = Shape::from(Vec::<usize>::new());
                core::mem::swap(&mut shape, &mut self.shape);

                let ir = TensorIr {
                    id: self.id,
                    shape,
                    status: TensorStatus::ReadWrite,
                    dtype: self.dtype,
                };
                let mut streams = OperationStreams::default();
                streams.tensor(self);

                self.client
                    .register(streams, OperationIr::Drop(ir), DropOp { id: self.id });
            }
            TensorStatus::ReadOnly => {}
            TensorStatus::NotInit => {}
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
