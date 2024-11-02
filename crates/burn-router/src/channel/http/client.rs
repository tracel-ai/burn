use core::{future::Future, sync::atomic::AtomicU64};
use std::sync::Arc;

use super::{
    CloseConnection, ReadTensor, RegisterOperation, RegisterOrphan, RegisterTensor,
    RegisterTensorEmpty, SyncBackend,
};
use crate::{MultiBackendBridge, RouterTensor, RunnerChannel, RunnerClient};
use burn_tensor::{
    backend::{DeviceId, DeviceOps},
    repr::{OperationDescription, TensorDescription},
    DType, TensorData,
};
use reqwest::Url;

/// A local channel with direct connection to the backend runner clients.
#[derive(Clone)]
pub struct HttpChannel;

/// TODO:
#[derive(Clone)]
pub struct HttpClient {
    state: Arc<HttpClientState>,
}

struct HttpClientState {
    runtime: tokio::runtime::Runtime,
    position: AtomicU64,
    client: reqwest::Client,
    url: reqwest::Url,
    id: u64,
}

impl Drop for HttpClientState {
    fn drop(&mut self) {
        let client = self.client.clone();
        let position = self
            .position
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let id = self.id;
        let url = self.url.join("close").unwrap();

        let fut = async move {
            let res = client
                .post(url)
                .json(&CloseConnection {
                    position,
                    client_id: id,
                })
                .send()
                .await;

            let res = res.unwrap();
            let body = res.json::<burn_tensor::TensorData>().await;

            body.unwrap()
        };

        self.runtime.block_on(fut);
    }
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

impl HttpClient {
    fn position(&self) -> u64 {
        self.state
            .position
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed)
    }

    fn url(&self, path: &str) -> Url {
        self.state.url.join(path).unwrap()
    }

    fn spawn<F>(&self, fut: F)
    where
        F: Future + Send + Sync + 'static,
        F::Output: Send + Sync + 'static,
    {
        self.state.runtime.spawn(fut);
    }
}

impl RunnerClient for HttpClient {
    type Device = HttpDevice;

    fn register(&self, op: OperationDescription) {
        log::info!("Register Operation.");

        let res = self
            .state
            .client
            .post(self.url("register-operation"))
            .json(&RegisterOperation {
                op,
                position: self.position(),
                client_id: self.state.id,
            })
            .send();

        let fut = async move {
            let res = res.await.unwrap();
            let body = res.json::<()>().await;

            body.unwrap()
        };

        self.spawn(fut);
    }

    fn read_tensor(
        &self,
        tensor: TensorDescription,
    ) -> impl core::future::Future<Output = burn_tensor::TensorData> + Send {
        let url = self.url("read-tensor");
        let position = self.position();
        let state = self.state.clone();
        let state2 = self.state.clone();

        let fut = async move {
            let res = state
                .client
                .post(url)
                .json(&ReadTensor {
                    tensor,
                    position,
                    client_id: self.state.id,
                })
                .send()
                .await;

            let res = res.unwrap();
            let body = res.json::<burn_tensor::TensorData>().await;

            body.unwrap()
        };

        let data = state2.runtime.block_on(fut);
        let fut = async move { data };

        fut
    }

    fn register_tensor_data(&self, data: burn_tensor::TensorData) -> RouterTensor<Self> {
        log::info!("Register Tensor.");

        let position = self.position();

        let fut = async move {
            let res = self
                .state
                .client
                .post(self.url("register-tensor"))
                .json(&RegisterTensor {
                    data,
                    position,
                    client_id: self.state.id,
                })
                .send()
                .await;

            let res = res.unwrap();
            let body = res.json::<TensorDescription>().await;

            body.unwrap()
        };

        let tensor = self.state.runtime.block_on(fut);

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

        let position = self.position();

        let fut = async move {
            let res = self
                .state
                .client
                .post(self.url("register-tensor-empty"))
                .json(&RegisterTensorEmpty {
                    shape,
                    dtype,
                    position,
                    client_id: self.state.id,
                })
                .send()
                .await;

            let res = res.unwrap();
            let body = res.json::<TensorDescription>().await;

            body.unwrap()
        };

        let tensor = self.state.runtime.block_on(fut);

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
            url: self.state.url.clone(),
        }
    }

    fn register_orphan(&self, id: &burn_tensor::repr::TensorId) {
        log::info!("Register Orphan.");
        let id = id.clone();

        let position = self.position();

        let res = self
            .state
            .client
            .post(self.url("register-orphan"))
            .json(&RegisterOrphan {
                id,
                position,
                client_id: self.state.id,
            })
            .send();

        let fut = async move {
            let res = res.await.unwrap();
            let body = res.json::<()>().await;

            body.unwrap()
        };

        self.spawn(fut);
    }

    fn sync(&self) {
        log::info!("Sync");
        let position = self.position();

        let fut = async move {
            let res = self
                .state
                .client
                .post(self.url("sync"))
                .json(&SyncBackend {
                    position,
                    client_id: self.state.id,
                })
                .send()
                .await;

            let res = res.unwrap();
            let body = res.json::<()>().await;

            body.unwrap()
        };

        self.state.runtime.block_on(fut);
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
        let client = reqwest::Client::new();
        let id = burn_common::id::IdGenerator::generate();
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let state = HttpClientState {
            runtime,
            position: AtomicU64::new(0),
            client,
            url: device.url.clone(),
            id,
        };

        HttpClient {
            state: Arc::new(state),
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
