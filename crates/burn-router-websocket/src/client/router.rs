use std::sync::Arc;

use burn_router::{MultiBackendBridge, RouterTensor, RunnerClient};
use burn_tensor::{
    backend::{DeviceId, DeviceOps},
    DType, TensorData,
};

use crate::shared::{TaskContent, TaskResponseContent};

use super::WsClient;

impl RunnerClient for WsClient {
    type Device = WsDevice;

    fn register(&self, op: burn_tensor::repr::OperationDescription) {
        let fut = self.sender.send(TaskContent::RegisterOperation(op));
        self.runtime.block_on(fut);
    }

    fn read_tensor(
        &self,
        tensor: burn_tensor::repr::TensorDescription,
    ) -> impl std::future::Future<Output = TensorData> + Send {
        let fut = self.sender.send_callback(TaskContent::ReadTensor(tensor));

        let fut = async move {
            match fut.await {
                TaskResponseContent::ReadTensor(data) => data,
                _ => panic!("Invalid message type"),
            }
        };

        fut
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let id = self.sender.new_tensor_id();
        let shape = data.shape.clone();
        let dtype = data.dtype;

        let fut = self.sender.send(TaskContent::RegisterTensor(id, data));

        self.runtime.block_on(fut);

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
        _full_precision: bool,
    ) -> RouterTensor<Self> {
        self.register_empty_tensor(shape, DType::F32)
    }

    fn device(&self) -> Self::Device {
        self.device.clone()
    }

    fn register_orphan(&self, id: &burn_tensor::repr::TensorId) {
        let fut = self.sender.send(TaskContent::RegisterOrphan(id.clone()));
        self.runtime.block_on(fut);
    }

    fn sync(&self) {
        let fut = self.sender.send_callback(TaskContent::SyncBackend);

        let fut = async move {
            match fut.await {
                TaskResponseContent::SyncBackend => {}
                _ => panic!("Invalid message type"),
            };
        };

        self.runtime.block_on(fut)
    }

    fn seed(&self, _seed: u64) {
        // Skip
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WsDevice {
    pub(crate) address: Arc<String>,
}

impl Default for WsDevice {
    fn default() -> Self {
        Self {
            address: Arc::new(String::from("ws://127.0.0.1:3000")),
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
