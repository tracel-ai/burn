use burn_tensor::{TensorData, backend::Backend};
use std::{collections::HashMap, marker::PhantomData, sync::Arc};
use tokio::sync::Mutex;
use tokio::{runtime::Runtime, sync::Notify};
use tokio_util::sync::CancellationToken;

use crate::global::shared::base::NodeAddress;

use burn_network::{ClientNetworkStream, Server, ServerNetworkStream};

#[derive(Clone)]
pub struct TensorDataClient<B: Backend> {
    state: Arc<TensorDataService<B>>,
}

struct TensorDataService<B: Backend> {
    /// Maps tensor IDs to their exposed state.
    pub exposed_tensors: Mutex<HashMap<u32, TensorExposeState>>,
    /// Maps node addresses to their WebSocket streams.
    pub streams: Mutex<HashMap<NodeAddress, Arc<Mutex<ClientNetworkStream>>>>,
    /// Notify when a new tensor is exposed.
    pub new_tensor_notify: Arc<Notify>,

    cancel_token: CancellationToken,

    _phantom_data: PhantomData<B>,
}

pub struct TensorExposeState {
    /// The bytes of the tensor data
    pub bytes: bytes::Bytes,
    /// How many times the tensor will be downloaded
    pub max_downloads: u32,
    /// How man times the tensor has been downloaded
    pub cur_download_count: u32,
    /// Unique identifier between two nodes for the transfer of a tensor.
    pub transfer_id: u32,
}

impl<B: Backend> TensorDataClient<B> {
    pub fn new(runtime: &Runtime, cancel_token: CancellationToken, data_server_port: u16) -> Self {
        let state = Arc::new(TensorDataService::new(cancel_token.clone()));
        runtime.spawn(Self::start(state.clone(), cancel_token, data_server_port));

        Self { state }
    }

    /// Start the server on the given address.
    /// This will block until the server is stopped with the `cancel_token`.
    async fn start(state: Arc<TensorDataService<B>>, cancel_token: CancellationToken, port: u16) {
        let mut server = Server::<Arc<TensorDataService<B>>>::new(port);

        server = server.route("/data", async |state, stream| {
            state.handle_stream(stream).await;
        });

        let cancel_token = cancel_token.clone();
        let shutdown = async move {
            cancel_token.cancelled().await;
        };

        server.serve(state, shutdown).await;
    }

    /// Exposes a tensor to the data server, allowing it to be downloaded by other nodes.
    pub(crate) async fn expose(
        &self,
        tensor: <B as Backend>::FloatTensorPrimitive,
        max_downloads: u32,
        transfer_id: u32,
    ) {
        self.state.expose(tensor, max_downloads, transfer_id).await
    }

    /// Downloads a tensor that is exposed on another server. Requires a Tokio 1.x runtime
    pub(crate) async fn download_next_tensor(
        &self,
        remote: &NodeAddress,
        tensor_id: u32,
    ) -> Option<TensorData> {
        self.state.download_tensor(remote, tensor_id).await
    }

    pub(crate) async fn close(&mut self) {
        self.state.close().await;
    }
}

impl<B: Backend> TensorDataService<B> {
    pub fn new(cancel_token: CancellationToken) -> Self {
        Self {
            exposed_tensors: Mutex::new(HashMap::new()),
            streams: Mutex::new(HashMap::new()),
            new_tensor_notify: Arc::new(Notify::new()),
            cancel_token,
            _phantom_data: PhantomData::<B>,
        }
    }

    /// Exposes a tensor to the data server, allowing it to be downloaded by other nodes.
    pub(crate) async fn expose(
        &self,
        tensor: <B as Backend>::FloatTensorPrimitive,
        max_downloads: u32,
        transfer_id: u32,
    ) {
        let data = B::float_into_data(tensor).await;
        let bytes: bytes::Bytes = rmp_serde::to_vec(&data).unwrap().into();
        let mut exposed_tensors = self.exposed_tensors.lock().await;
        exposed_tensors.insert(
            transfer_id,
            TensorExposeState {
                bytes,
                max_downloads,
                cur_download_count: 0,
                transfer_id,
            },
        );
        core::mem::drop(exposed_tensors);
        self.new_tensor_notify.notify_waiters();
    }

    pub(crate) async fn close(&self) {
        // Send a closing message to every open WebSocket stream

        let mut streams = self.streams.lock().await;
        for (_, stream) in streams.drain() {
            let mut stream = stream.lock().await;

            stream
                .close()
                .await
                .expect("Failed to close WebSocket stream");
        }
    }

    /// Downloads a tensor that is exposed on another server. Requires a Tokio 1.x runtime
    pub(crate) async fn download_tensor(
        &self,
        remote: &NodeAddress,
        tensor_id: u32,
    ) -> Option<TensorData> {
        log::info!("Downloading next tensor from {:?}", remote.0.as_str());

        let stream = self.get_data_stream(remote).await;
        let mut stream = stream.lock().await;

        // Send the download request with the download id
        let bytes: bytes::Bytes = rmp_serde::to_vec(&tensor_id).unwrap().into();
        stream
            .send(bytes)
            .await
            .expect("Failed to send download id");

        if let Ok(msg) = stream.recv().await {
            let Some(msg) = msg else {
                log::warn!("Received None message from the websocket, closing connection.");
                return None;
            };

            let data: TensorData = rmp_serde::from_slice(&msg.data)
                .expect("Can deserialize messages from the websocket.");
            return Some(data);
        }
        log::warn!("Closed connection");
        None
    }

    /// Get the WebSocket stream for the given address, or create a new one if it doesn't exist.
    async fn get_data_stream(&self, address: &NodeAddress) -> Arc<Mutex<ClientNetworkStream>> {
        let mut streams = self.streams.lock().await;
        match streams.get(address) {
            Some(stream) => stream.clone(),
            None => {
                // Open a new WebSocket connection to the address
                let address_request = format!("{}/{}", address.0.as_str(), "data");
                let stream = burn_network::connect(address_request).await;
                let Some(stream) = stream else {
                    panic!("Failed to connect to data server at {}", address.0.as_str());
                };

                let stream = Arc::new(Mutex::new(stream));
                streams.insert(address.clone(), stream.clone());

                stream
            }
        }
    }

    /// Handle incoming connections for downloading tensors.
    pub async fn handle_stream(&self, mut stream: ServerNetworkStream) {
        log::info!("[Data Handler] New connection for download.");

        while !self.cancel_token.is_cancelled() {
            match stream.recv().await {
                Ok(message) => {
                    if let Some(msg) = message {
                        let bytes = msg.data;
                        let tensor_id: u32 = rmp_serde::from_slice(&bytes)
                            .expect("Can deserialize messages from the websocket.");

                        let bytes = self.get_exposed_tensor_bytes(tensor_id).await.unwrap();

                        stream.send(bytes).await.unwrap();
                    } else {
                        eprintln!("Closed connection");
                        return;
                    }
                }
                Err(err) => panic!("Failed to receive message from websocket: {err:?}"),
            };
        }
        eprintln!("[Data Service] Closing connection for download.");
    }

    /// Get the requested exposed tensor data, and update download counter
    async fn get_exposed_tensor_bytes(&self, tensor_id: u32) -> Option<bytes::Bytes> {
        loop {
            {
                let mut exposed_tensors = self.exposed_tensors.lock().await;
                // take the tensor out of the hashmap while we download
                if let Some(mut exposed_state) = exposed_tensors.remove(&tensor_id) {
                    exposed_state.cur_download_count += 1;
                    let bytes = if exposed_state.cur_download_count == exposed_state.max_downloads {
                        exposed_state.bytes
                    } else {
                        let bytes = exposed_state.bytes.clone();
                        exposed_tensors.insert(tensor_id, exposed_state);
                        bytes
                    };
                    return Some(bytes);
                }
            }
            // No matching tensor, wait for a new one to come in.
            self.new_tensor_notify.notified().await;
        }
    }
}
