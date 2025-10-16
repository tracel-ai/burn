use super::FusionClient;
use crate::{
    FusionBackend, FusionDevice, FusionHandle, FusionRuntime, FusionServer, FusionTensor,
    stream::{OperationStreams, StreamId, execution::Operation},
};
use burn_ir::{OperationIr, TensorId, TensorIr};
use burn_tensor::TensorData;
use spin::Mutex;
use std::sync::Arc;

/// Use a mutex to communicate with the fusion server.
pub struct MutexFusionClient<R: FusionRuntime> {
    server: Arc<Mutex<FusionServer<R>>>,
    device: FusionDevice<R>,
}

impl<R> Clone for MutexFusionClient<R>
where
    R: FusionRuntime,
{
    fn clone(&self) -> Self {
        Self {
            server: self.server.clone(),
            device: self.device.clone(),
        }
    }
}

impl<R> FusionClient<R> for MutexFusionClient<R>
where
    R: FusionRuntime<FusionClient = Self> + 'static,
{
    fn new(device: FusionDevice<R>) -> Self {
        Self {
            device: device.clone(),
            server: Arc::new(Mutex::new(FusionServer::new(device))),
        }
    }

    fn register<O>(
        &self,
        streams: OperationStreams,
        repr: OperationIr,
        operation: O,
    ) -> Vec<FusionTensor<R>>
    where
        O: Operation<R> + 'static,
    {
        // Create output tensors returned by this operation
        let outputs = repr
            .outputs()
            .map(|output| {
                FusionTensor::new(
                    output.id,
                    output.shape.clone(),
                    output.dtype,
                    self.clone(),
                    StreamId::current(),
                )
            })
            .collect();

        self.server
            .lock()
            .register(streams, repr, Arc::new(operation));

        outputs
    }

    fn drain(&self) {
        let id = StreamId::current();
        self.server.lock().drain_stream(id);
    }

    fn create_empty_handle(&self) -> TensorId {
        self.server.lock().create_empty_handle()
    }

    fn device(&self) -> &FusionDevice<R> {
        &self.device
    }

    fn register_tensor_handle(&self, handle: FusionHandle<R>) -> TensorId {
        let mut server = self.server.lock();
        let id = server.create_empty_handle();
        server.handles.register_handle(id, handle);
        core::mem::drop(server);

        id
    }

    fn read_tensor_float<B>(
        self,
        tensor: TensorIr,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_float::<B>(tensor, stream)
    }

    fn read_tensor_int<B>(
        self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_int::<B>(tensor, id)
    }

    fn read_tensor_bool<B>(
        self,
        tensor: TensorIr,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_bool::<B>(tensor, stream)
    }

    fn read_tensor_quantized<B>(
        self,
        tensor: TensorIr,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_quantized::<B>(tensor, stream)
    }

    fn change_client_float<B>(
        &self,
        tensor: TensorIr,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let mut server_other = client.server.lock();
        let id = server_current.change_server_float::<B>(
            &tensor,
            stream,
            &client.device,
            &mut server_other,
        );

        core::mem::drop(server_current);
        core::mem::drop(server_other);

        FusionTensor::new(id, tensor.shape, tensor.dtype, client, StreamId::current())
    }

    fn change_client_int<B>(
        &self,
        tensor: TensorIr,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let mut server_other = client.server.lock();
        let id = server_current.change_server_int::<B>(
            &tensor,
            stream,
            &client.device,
            &mut server_other,
        );

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        FusionTensor::new(id, tensor.shape, tensor.dtype, client, StreamId::current())
    }

    fn change_client_bool<B>(
        &self,
        tensor: TensorIr,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let mut server_other = client.server.lock();
        let id = server_current.change_server_bool::<B>(
            &tensor,
            stream,
            &client.device,
            &mut server_other,
        );

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        FusionTensor::new(id, tensor.shape, tensor.dtype, client, StreamId::current())
    }

    fn change_client_quantized<B>(
        &self,
        tensor: TensorIr,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let mut server_other = client.server.lock();
        let id =
            server_current.change_server_quantized::<B>(&tensor, &client.device, &mut server_other);

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        FusionTensor::new(id, tensor.shape, tensor.dtype, client, StreamId::current())
    }

    fn resolve_tensor_float<B>(&self, tensor: FusionTensor<R>) -> B::FloatTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server = self.server.lock();
        server.drain_stream(tensor.stream);
        server.resolve_server_float::<B>(&tensor.into_ir())
    }

    fn resolve_tensor_int<B>(&self, tensor: FusionTensor<R>) -> B::IntTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server = self.server.lock();
        server.drain_stream(tensor.stream);
        server.resolve_server_int::<B>(&tensor.into_ir())
    }

    fn resolve_tensor_bool<B>(&self, tensor: FusionTensor<R>) -> B::BoolTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server = self.server.lock();
        server.drain_stream(tensor.stream);
        server.resolve_server_bool::<B>(&tensor.into_ir())
    }
}
