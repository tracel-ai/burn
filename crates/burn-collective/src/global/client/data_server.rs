use burn_network::network::{NetworkClient, NetworkServer, NetworkStream};
use burn_tensor::{TensorData, backend::Backend};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, marker::PhantomData, sync::Arc};
use tokio::sync::Mutex;
use tokio::{runtime::Runtime, sync::Notify};
use tokio_util::sync::CancellationToken;

use crate::global::shared::base::NodeAddress;

pub(crate) struct TensorDataClient<B, C, S>
where
    B: Backend,
    C: NetworkClient,
    S: NetworkServer<State = Arc<TensorDataService<B, C>>>,
{
    state: Arc<TensorDataService<B, C>>,
    _phantom_data: PhantomData<S>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct TensorTransferId(u32);

impl From<u32> for TensorTransferId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

#[derive(Debug, Serialize, Deserialize)]
enum Message {
    Request(TensorTransferId),
    Data(TensorData),
}

pub(crate) struct TensorDataService<B: Backend, C: NetworkClient> {
    /// Maps tensor transfer IDs to their exposed state.
    pub exposed_tensors: Mutex<HashMap<TensorTransferId, TensorExposeState>>,
    /// Maps node addresses to their WebSocket streams.
    pub streams: Mutex<HashMap<NodeAddress, Arc<Mutex<C::ClientStream>>>>,
    /// Notify when a new tensor is exposed.
    pub new_tensor_notify: Arc<Notify>,

    cancel_token: CancellationToken,

    _phantom_data: PhantomData<B>,
}

pub struct TensorExposeState {
    /// The bytes of the tensor data message. Message::Data(...) serialized with rmp_serde
    pub bytes: bytes::Bytes,
    /// How many times the tensor will be downloaded
    pub max_downloads: u32,
    /// How man times the tensor has been downloaded
    pub cur_download_count: u32,
}

impl<B, N, S> TensorDataClient<B, N, S>
where
    B: Backend,
    N: NetworkClient,
    S: NetworkServer<State = Arc<TensorDataService<B, N>>>,
{
    pub fn new(runtime: &Runtime, cancel_token: CancellationToken, data_server_port: u16) -> Self {
        let state = Arc::new(TensorDataService::new(cancel_token.clone()));
        runtime.spawn(Self::start(state.clone(), cancel_token, data_server_port));

        Self {
            state,
            _phantom_data: PhantomData,
        }
    }

    /// Start the server on the given address.
    /// This will block until the server is stopped with the `cancel_token`.
    async fn start(
        state: Arc<TensorDataService<B, N>>,
        cancel_token: CancellationToken,
        port: u16,
    ) {
        let mut server = S::new(port);

        server = server.route(
            "/data",
            async |state: Arc<TensorDataService<B, N>>, stream: S::ServerStream| {
                Self::handle_stream(&state, stream).await;
            },
        );

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
        transfer_id: TensorTransferId,
    ) {
        self.state.expose(tensor, max_downloads, transfer_id).await
    }

    /// Downloads a tensor that is exposed on another server. Requires a Tokio 1.x runtime
    pub(crate) async fn download_next_tensor(
        &self,
        remote: &NodeAddress,
        transfer_id: TensorTransferId,
    ) -> Option<TensorData> {
        self.state.download_tensor(remote, transfer_id).await
    }

    pub(crate) async fn close(&mut self) {
        self.state.close().await;
    }

    /// Handle incoming connections for downloading tensors.
    pub async fn handle_stream(state: &TensorDataService<B, N>, mut stream: S::ServerStream) {
        log::info!("[Data Handler] New connection for download.");

        while !state.cancel_token.is_cancelled() {
            match stream.recv().await {
                Ok(message) => {
                    if let Some(msg) = message {
                        let bytes = msg.data;
                        let msg: Message = rmp_serde::from_slice(&bytes)
                            .expect("Can deserialize messages from the websocket.");
                        let Message::Request(transfer_id) = msg else {
                            panic!("Received a message that wasn't a tensor request! {:?}", msg);
                        };

                        let bytes = state.get_exposed_tensor_bytes(transfer_id).await.unwrap();

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
}

impl<B: Backend, N: NetworkClient> TensorDataService<B, N> {
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
        transfer_id: TensorTransferId,
    ) {
        let data = B::float_into_data(tensor).await;
        let bytes: bytes::Bytes = rmp_serde::to_vec(&Message::Data(data)).unwrap().into();
        let mut exposed_tensors = self.exposed_tensors.lock().await;
        exposed_tensors.insert(
            transfer_id,
            TensorExposeState {
                bytes,
                max_downloads,
                cur_download_count: 0,
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
        transfer_id: TensorTransferId,
    ) -> Option<TensorData> {
        log::info!("Downloading next tensor from {:?}", remote.0.as_str());

        let stream = self.get_data_stream(remote).await;
        let mut stream = stream.lock().await;

        // Send the download request with the download id
        let bytes: bytes::Bytes = rmp_serde::to_vec(&Message::Request(transfer_id))
            .unwrap()
            .into();
        stream
            .send(bytes)
            .await
            .expect("Failed to send download id");

        if let Ok(msg) = stream.recv().await {
            let Some(msg) = msg else {
                log::warn!("Received None message from the websocket, closing connection.");
                return None;
            };

            let Message::Data(data) = rmp_serde::from_slice(&msg.data)
                .expect("Can deserialize messages from the websocket.")
            else {
                panic!("Message should have been TensorData")
            };
            return Some(data);
        }
        log::warn!("Closed connection");
        None
    }

    /// Get the WebSocket stream for the given address, or create a new one if it doesn't exist.
    async fn get_data_stream(&self, address: &NodeAddress) -> Arc<Mutex<N::ClientStream>> {
        let mut streams = self.streams.lock().await;
        match streams.get(address) {
            Some(stream) => stream.clone(),
            None => {
                // Open a new WebSocket connection to the address
                let address_request = format!("{}/{}", address.0.as_str(), "data");
                let stream = N::connect(address_request).await;
                let Some(stream) = stream else {
                    panic!("Failed to connect to data server at {}", address.0.as_str());
                };

                let stream = Arc::new(Mutex::new(stream));
                streams.insert(address.clone(), stream.clone());

                stream
            }
        }
    }

    /// Get the requested exposed tensor data, and update download counter
    async fn get_exposed_tensor_bytes(
        &self,
        transfer_id: TensorTransferId,
    ) -> Option<bytes::Bytes> {
        loop {
            {
                let mut exposed_tensors = self.exposed_tensors.lock().await;
                // take the tensor out of the hashmap while we download
                if let Some(mut exposed_state) = exposed_tensors.remove(&transfer_id) {
                    exposed_state.cur_download_count += 1;
                    let bytes = if exposed_state.cur_download_count == exposed_state.max_downloads {
                        exposed_state.bytes
                    } else {
                        let bytes = exposed_state.bytes.clone();
                        exposed_tensors.insert(transfer_id, exposed_state);
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

impl<B, C, S> Clone for TensorDataClient<B, C, S>
where
    B: Backend,
    C: NetworkClient,
    S: NetworkServer<State = Arc<TensorDataService<B, C>>>,
{
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            _phantom_data: PhantomData,
        }
    }
}
