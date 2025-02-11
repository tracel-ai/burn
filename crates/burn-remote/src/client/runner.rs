use burn_router::{MultiBackendBridge, RouterTensor, RunnerClient};
use burn_tensor::{
    backend::{DeviceId, DeviceOps},
    DType, TensorData,
};
use std::{future::Future, sync::Arc};

use crate::shared::{ComputeTask, TaskResponseContent};

use super::WsClient;

// It is very important to block on any request made with the sender, since ordering is crucial
// when registering operation or creating tensors.
//
// The overhead is minimal, since we only wait for the task to be sent to the async
// channel, but not sent to the websocket server and even less processed by the server.
impl RunnerClient for WsClient {
    type Device = WsDevice;

    fn register(&self, op: burn_ir::OperationIr) {
        self.sender
            .send(ComputeTask::RegisterOperation(Box::new(op)));
    }

    fn read_tensor(
        &self,
        tensor: burn_ir::TensorIr,
    ) -> impl std::future::Future<Output = TensorData> + Send {
        // Important for ordering to call the creation of the future sync.
        let fut = self.sender.send_callback(ComputeTask::ReadTensor(tensor));

        async move {
            match fut.await {
                TaskResponseContent::ReadTensor(data) => data,
                _ => panic!("Invalid message type"),
            }
        }
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let id = self.sender.new_tensor_id();
        let shape = data.shape.clone();
        let dtype = data.dtype;

        self.sender.send(ComputeTask::RegisterTensor(id, data));

        RouterTensor::new(Arc::new(id), shape, dtype, self.clone())
    }

    fn register_empty_tensor(
        &self,
        shape: Vec<usize>,
        dtype: burn_tensor::DType,
    ) -> RouterTensor<Self> {
        let id = self.sender.new_tensor_id();

        RouterTensor::new(Arc::new(id), shape, dtype, self.clone())
    }

    fn register_float_tensor(
        &self,
        shape: Vec<usize>,
        _dtype: burn_tensor::FloatDType,
    ) -> RouterTensor<Self> {
        self.register_empty_tensor(shape, DType::F32)
    }

    fn device(&self) -> Self::Device {
        self.device.clone()
    }

    fn register_orphan(&self, id: &burn_ir::TensorId) {
        self.sender.send(ComputeTask::RegisterOrphan(*id));
    }

    fn sync(&self) -> impl Future<Output = ()> + Send + 'static {
        // Important for ordering to call the creation of the future sync.
        let fut = self.sender.send_callback(ComputeTask::SyncBackend);
        let runtime = self.runtime.clone();

        async move {
            match runtime.block_on(fut) {
                TaskResponseContent::SyncBackend => {}
                _ => panic!("Invalid message type"),
            };
        }
    }

    fn seed(&self, _seed: u64) {
        // TODO
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
/// The device contains the connection information of the server.
pub struct WsDevice {
    pub(crate) address: Arc<String>,
}

impl WsDevice {
    /// Create a device from an url.
    pub fn new(url: &str) -> Self {
        let mut address = String::new();

        if !url.starts_with("ws://") {
            address += "ws://";
            address += url;
        } else {
            address += url;
        };

        Self {
            address: Arc::new(address),
        }
    }
}

impl Default for WsDevice {
    fn default() -> Self {
        let address = match std::env::var("BURN_REMOTE_ADDRESS") {
            Ok(address) => address,
            Err(_) => String::from("ws://127.0.0.1:3000"),
        };

        Self {
            address: Arc::new(address),
        }
    }
}

impl DeviceOps for WsDevice {
    fn id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: 0,
        }
    }
}

pub struct WsBridge;

impl MultiBackendBridge for WsBridge {
    type TensorHandle = TensorData;
    type Device = WsDevice;

    fn change_backend_float(
        tensor: Self::TensorHandle,
        _shape: burn_tensor::Shape,
        _target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor
    }

    fn change_backend_int(
        tensor: Self::TensorHandle,
        _shape: burn_tensor::Shape,
        _target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor
    }

    fn change_backend_bool(
        tensor: Self::TensorHandle,
        _shape: burn_tensor::Shape,
        _target_device: &Self::Device,
    ) -> Self::TensorHandle {
        tensor
    }
}
