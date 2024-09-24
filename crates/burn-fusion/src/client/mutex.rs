use super::FusionClient;
use crate::{
    stream::{execution::Operation, StreamId},
    FusionBackend, FusionDevice, FusionHandle, FusionQuantizationParameters, FusionRuntime,
    FusionServer, FusionTensor, QFusionTensor,
};
use burn_tensor::{
    repr::{OperationDescription, QuantizedTensorDescription, TensorDescription, TensorId},
    DType,
};
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
    R: FusionRuntime<FusionClient = Self>,
{
    fn new(device: FusionDevice<R>) -> Self {
        Self {
            device: device.clone(),
            server: Arc::new(Mutex::new(FusionServer::new(device))),
        }
    }

    fn register<O>(&self, streams: Vec<StreamId>, description: OperationDescription, operation: O)
    where
        O: Operation<R> + 'static,
    {
        self.server
            .lock()
            .register(streams, description, Box::new(operation))
    }

    fn drain(&self) {
        let id = StreamId::current();
        self.server.lock().drain_stream(id);
    }

    fn tensor_uninitialized(&self, shape: Vec<usize>, dtype: DType) -> FusionTensor<R> {
        let id = self.server.lock().create_empty_handle();

        FusionTensor::new(id, shape, dtype, self.clone(), StreamId::current())
    }

    fn device(&self) -> &FusionDevice<R> {
        &self.device
    }

    fn register_tensor(
        &self,
        handle: FusionHandle<R>,
        shape: Vec<usize>,
        stream: StreamId,
        dtype: DType,
    ) -> FusionTensor<R> {
        let mut server = self.server.lock();
        let id = server.create_empty_handle();
        server.handles.register_handle(*id.as_ref(), handle);
        core::mem::drop(server);

        FusionTensor::new(id, shape, dtype, self.clone(), stream)
    }

    async fn read_tensor_float<B>(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> burn_tensor::TensorData
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_float::<B>(tensor, stream).await
    }

    async fn read_tensor_int<B>(
        &self,
        tensor: TensorDescription,
        id: StreamId,
    ) -> burn_tensor::TensorData
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_int::<B>(tensor, id).await
    }

    async fn read_tensor_bool<B>(
        &self,
        tensor: TensorDescription,
        stream: StreamId,
    ) -> burn_tensor::TensorData
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_bool::<B>(tensor, stream).await
    }

    async fn read_tensor_quantized<B>(
        &self,
        tensor: QuantizedTensorDescription,
        streams: Vec<StreamId>,
    ) -> burn_tensor::TensorData
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server
            .lock()
            .read_quantized::<B>(tensor, streams)
            .await
    }

    fn change_client_float<B>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server_other = client.server.lock();
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let id =
            server_current.change_server_float::<B>(&tensor, &client.device, &mut server_other);

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        FusionTensor::new(id, tensor.shape, tensor.dtype, client, StreamId::current())
    }

    fn change_client_int<B>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server_other = client.server.lock();
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let id = server_current.change_server_int::<B>(&tensor, &client.device, &mut server_other);

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        FusionTensor::new(id, tensor.shape, tensor.dtype, client, StreamId::current())
    }

    fn change_client_bool<B>(
        &self,
        tensor: TensorDescription,
        client: Self,
        stream: StreamId,
    ) -> FusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server_other = client.server.lock();
        let mut server_current = self.server.lock();
        server_current.drain_stream(stream);

        let id = server_current.change_server_bool::<B>(&tensor, &client.device, &mut server_other);

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        FusionTensor::new(id, tensor.shape, tensor.dtype, client, StreamId::current())
    }

    fn change_client_quantized<B>(
        &self,
        tensor: QuantizedTensorDescription,
        client: Self,
        streams: Vec<StreamId>,
    ) -> QFusionTensor<R>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server_other = client.server.lock();
        let mut server_current = self.server.lock();
        for stream in streams {
            server_current.drain_stream(stream);
        }

        let mut ids =
            server_current.change_server_quantized::<B>(&tensor, &client.device, &mut server_other);

        core::mem::drop(server_other);
        core::mem::drop(server_current);

        // NOTE: the expected order is known [qtensor, scale, <offset>]
        let offset = tensor.qparams.offset.map(|desc| {
            FusionTensor::new(
                ids.pop().unwrap(),
                desc.shape,
                desc.dtype,
                client.clone(),
                StreamId::current(),
            )
        });
        let scale = FusionTensor::new(
            ids.pop().unwrap(),
            tensor.qparams.scale.shape,
            tensor.qparams.scale.dtype,
            client.clone(),
            StreamId::current(),
        );
        let qtensor = FusionTensor::new(
            ids.pop().unwrap(),
            tensor.tensor.shape,
            tensor.tensor.dtype,
            client,
            StreamId::current(),
        );

        QFusionTensor {
            qtensor,
            scheme: tensor.scheme,
            qparams: FusionQuantizationParameters { scale, offset },
        }
    }

    fn register_orphan(&self, id: &TensorId) {
        self.server.lock().drop_tensor_handle(*id);
    }
}
