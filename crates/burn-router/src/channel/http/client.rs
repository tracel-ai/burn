use core::sync::atomic::AtomicU64;
use std::sync::Arc;

use super::{
    ReadTensor, RegisterOperation, RegisterOrphan, RegisterTensor, RegisterTensorEmpty, SyncBackend,
};
use crate::{MultiBackendBridge, RouterTensor, RunnerChannel, RunnerClient};
use burn_tensor::{
    backend::{DeviceId, DeviceOps},
    repr::{OperationDescription, TensorDescription},
    DType, TensorData,
};

/// A local channel with direct connection to the backend runner clients.
#[derive(Clone)]
pub struct HttpChannel;

/// TODO:
#[derive(Clone)]
pub struct HttpClient {
    runtime: Arc<tokio::runtime::Runtime>,
    order: Arc<AtomicU64>,
    client: reqwest::Client,
    url: reqwest::Url,
}

/// TODO:
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct HttpDevice {
    url: reqwest::Url,
}

impl Default for HttpDevice {
    fn default() -> Self {
        Self {
            url: reqwest::Url::parse("http://localhost:3000").unwrap(),
        }
    }
}

/// TODO:
pub struct HttpBridge;

impl MultiBackendBridge for HttpBridge {
    type TensorHandle = TensorData;
    type Device = HttpDevice;

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

impl DeviceOps for HttpDevice {
    fn id(&self) -> DeviceId {
        DeviceId {
            type_id: 0,
            index_id: 0,
        }
    }
}

impl RunnerClient for HttpClient {
    type Device = HttpDevice;

    fn register(&self, op: OperationDescription) {
        log::info!("Register Operation.");

        let order = self
            .order
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let res = self
            .client
            .post(self.url.join("register-operation").unwrap())
            .json(&RegisterOperation { op, index: order })
            .send();

        let fut = async move {
            let res = res.await.unwrap();
            let body = res.json::<()>().await;

            body.unwrap()
        };

        self.runtime.block_on(fut);
    }

    fn read_tensor(
        &self,
        tensor: TensorDescription,
    ) -> impl core::future::Future<Output = burn_tensor::TensorData> + Send {
        let client = self.client.clone();
        let url = self.url.join("read-tensor").unwrap();
        let runtime = self.runtime.clone();

        let order = self
            .order
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        let fut = async move {
            let res = client
                .post(url)
                .json(&ReadTensor {
                    tensor,
                    index: order,
                })
                .send()
                .await;

            let res = res.unwrap();
            let body = res.json::<burn_tensor::TensorData>().await;

            body.unwrap()
        };

        let data = runtime.block_on(fut);
        //let fut = async move { runtime.spawn(fut).await.unwrap() };
        let fut = async move { data };

        fut
    }

    fn register_tensor_data(&self, data: burn_tensor::TensorData) -> RouterTensor<Self> {
        log::info!("Register Tensor.");

        let order = self
            .order
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        let fut = async move {
            let res = self
                .client
                .post(self.url.join("register-tensor").unwrap())
                .json(&RegisterTensor { data, index: order })
                .send()
                .await;

            let res = res.unwrap();
            let body = res.json::<TensorDescription>().await;

            body.unwrap()
        };

        let tensor = self.runtime.block_on(fut);

        RouterTensor::new(
            Arc::new(tensor.id),
            tensor.shape,
            tensor.dtype,
            self.clone(),
        )
    }

    fn register_empty_tensor(
        &self,
        shape: Vec<usize>,
        dtype: burn_tensor::DType,
    ) -> RouterTensor<Self> {
        log::info!("Register Tensor Empty.");

        let order = self
            .order
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        let fut = async move {
            let res = self
                .client
                .post(self.url.join("register-tensor-empty").unwrap())
                .json(&RegisterTensorEmpty {
                    shape,
                    dtype,
                    index: order,
                })
                .send()
                .await;

            let res = res.unwrap();
            let body = res.json::<TensorDescription>().await;

            body.unwrap()
        };

        let tensor = self.runtime.block_on(fut);

        RouterTensor::new(
            Arc::new(tensor.id),
            tensor.shape,
            tensor.dtype,
            self.clone(),
        )
    }

    fn register_float_tensor(
        &self,
        shape: Vec<usize>,
        _full_precision: bool,
    ) -> RouterTensor<Self> {
        self.register_empty_tensor(shape, DType::F32)
    }

    fn device(&self) -> Self::Device {
        HttpDevice {
            url: self.url.clone(),
        }
    }

    fn register_orphan(&self, id: &burn_tensor::repr::TensorId) {
        log::info!("Register Orphan.");
        let id = id.clone();

        let order = self
            .order
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        let res = self
            .client
            .post(self.url.join("register-orphan").unwrap())
            .json(&RegisterOrphan { id, index: order })
            .send();

        let fut = async move {
            let res = res.await.unwrap();
            let body = res.json::<()>().await;

            body.unwrap()
        };

        self.runtime.block_on(fut);
    }

    fn sync(&self) {
        log::info!("Sync");
        let order = self
            .order
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let fut = async move {
            let res = self
                .client
                .post(self.url.join("sync").unwrap())
                .json(&SyncBackend { index: order })
                .send()
                .await;

            let res = res.unwrap();
            let body = res.json::<()>().await;

            body.unwrap()
        };

        self.runtime.block_on(fut);
    }

    fn seed(&self, _seed: u64) {
        // TODO
    }
}

impl RunnerChannel for HttpChannel {
    type Device = HttpDevice;
    type Bridge = HttpBridge;
    type Client = HttpClient;

    type FloatElem = f32;

    type IntElem = i32;

    fn name() -> String {
        "http".into()
    }

    fn init_client(device: &Self::Device) -> Self::Client {
        log::info!("HERE");
        let client = reqwest::Client::new();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        HttpClient {
            runtime: Arc::new(runtime),
            order: Arc::new(AtomicU64::new(0)),
            client,
            url: device.url.clone(),
        }
    }

    fn get_tensor_handle(
        _tensor: &TensorDescription,
        _client: &Self::Client,
    ) -> crate::TensorHandle<Self::Bridge> {
        panic!("Unsupported")
    }

    fn register_tensor(
        _client: &Self::Client,
        _handle: crate::TensorHandle<Self::Bridge>,
        _shape: Vec<usize>,
        _dtype: burn_tensor::DType,
    ) -> RouterTensor<Self::Client> {
        panic!("Unsupported")
    }
}
