use crate::{
    FusionBackend, FusionDevice, FusionHandle, FusionRuntime, FusionServer, FusionTensor,
    stream::{OperationStreams, StreamId, execution::Operation},
};
use burn_common::device::{Device, DeviceContext, DeviceState};
use burn_ir::{OperationIr, TensorIr};
use burn_tensor::{DType, Shape, TensorData};
use std::sync::Arc;

/// Use a mutex to communicate with the fusion server.
pub struct GlobalFusionClient<R: FusionRuntime> {
    server: DeviceContext<FusionServer<R>>,
    device: FusionDevice<R>,
}

impl<R: FusionRuntime> DeviceState for FusionServer<R> {
    fn init(device_id: burn_common::device::DeviceId) -> Self {
        let device = FusionDevice::<R>::from_id(device_id);
        FusionServer::new(device)
    }
}

impl<R> Clone for GlobalFusionClient<R>
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
impl<R> GlobalFusionClient<R>
where
    R: FusionRuntime + 'static,
{
    /// Loads the client from the given device.
    pub fn load(device: &FusionDevice<R>) -> Self {
        Self {
            device: device.clone(),
            server: DeviceContext::locate(device),
        }
    }
}

impl<R> GlobalFusionClient<R>
where
    R: FusionRuntime + 'static,
{
    /// Create a new client for the given [device](FusionRuntime::FusionDevice).
    pub fn new(device: FusionDevice<R>) -> Self {
        Self {
            device: device.clone(),
            server: DeviceContext::locate(&device),
        }
    }

    /// Register a new [tensor operation intermediate representation](OperationIr).
    pub fn register<O>(&self, streams: OperationStreams, repr: OperationIr, operation: O)
    where
        O: Operation<R> + 'static,
    {
        let mut server = self.server.lock();
        server.register(streams, repr, Arc::new(operation));
    }

    /// Register all lazy computation.
    pub fn drain(&self) {
        let id = StreamId::current();
        self.server.lock().drain_stream(id);
    }

    /// Create a new [fusion tensor](FusionTensor), but with no resources allocated to it.
    pub fn tensor_uninitialized(&self, shape: Shape, dtype: DType) -> FusionTensor<R> {
        let id = self.server.lock().create_empty_handle();

        FusionTensor::new(id, shape, dtype, self.clone(), StreamId::current())
    }

    /// Get the current device used by all operations handled by this client.
    pub fn device(&self) -> &FusionDevice<R> {
        &self.device
    }

    /// Create a tensor with the given handle and shape.
    pub fn register_tensor(
        &self,
        handle: FusionHandle<R>,
        shape: Shape,
        stream: StreamId,
        dtype: DType,
    ) -> FusionTensor<R> {
        let mut server = self.server.lock();
        let id = server.create_empty_handle();
        server.handles.register_handle(id, handle);
        core::mem::drop(server);

        FusionTensor::new(id, shape, dtype, self.clone(), stream)
    }

    /// Read the values contained by a float tensor.
    pub fn read_tensor_float<B>(
        self,
        tensor: TensorIr,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_float::<B>(tensor, stream)
    }

    /// Read the values contained by an int tensor.
    pub fn read_tensor_int<B>(
        self,
        tensor: TensorIr,
        id: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_int::<B>(tensor, id)
    }

    /// Read the values contained by a bool tensor.
    pub fn read_tensor_bool<B>(
        self,
        tensor: TensorIr,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_bool::<B>(tensor, stream)
    }

    /// Read the values contained by a quantized tensor.
    pub fn read_tensor_quantized<B>(
        self,
        tensor: TensorIr,
        stream: StreamId,
    ) -> impl Future<Output = TensorData> + Send
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        self.server.lock().read_quantized::<B>(tensor, stream)
    }

    /// Change the client of the given float tensor.
    pub fn change_client_float<B>(
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

    /// Change the client of the given int tensor.
    pub fn change_client_int<B>(
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

    /// Change the client of the given bool tensor.
    pub fn change_client_bool<B>(
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

    /// Change the client of the given quantized tensor.
    pub fn change_client_quantized<B>(
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

    /// Resolve the given float tensor to a primitive tensor.
    pub fn resolve_tensor_float<B>(&self, tensor: FusionTensor<R>) -> B::FloatTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server = self.server.lock();
        server.drain_stream(tensor.stream);
        server.resolve_server_float::<B>(&tensor.into_ir())
    }

    /// Resolve the given int tensor to a primitive tensor.
    pub fn resolve_tensor_int<B>(&self, tensor: FusionTensor<R>) -> B::IntTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server = self.server.lock();
        server.drain_stream(tensor.stream);
        server.resolve_server_int::<B>(&tensor.into_ir())
    }

    /// Resolve the given bool tensor to a primitive tensor.
    pub fn resolve_tensor_bool<B>(&self, tensor: FusionTensor<R>) -> B::BoolTensorPrimitive
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let mut server = self.server.lock();
        server.drain_stream(tensor.stream);
        server.resolve_server_bool::<B>(&tensor.into_ir())
    }
}
