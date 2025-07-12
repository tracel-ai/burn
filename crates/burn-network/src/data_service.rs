use crate::network::Network;
use crate::network::{NetworkAddress, NetworkClient, NetworkServer, NetworkStream};
use burn_tensor::{TensorData, backend::Backend};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, marker::PhantomData, sync::Arc};
use tokio::sync::Mutex;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorTransferId(Option<u32>);

impl From<u32> for TensorTransferId {
    fn from(value: u32) -> Self {
        Self(Some(value))
    }
}

impl TensorTransferId {
    pub fn none() -> Self {
        Self(None)
    }

    pub fn next(&mut self) {
        if let Some(id) = &mut self.0 {
            *id += 1;
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
enum Message {
    Request(TensorTransferId),
    Data(TensorData),
}

type ClientStreamRef<C> = Arc<Mutex<<C as NetworkClient>::Stream>>;

pub struct TensorDataService<B: Backend, N: Network<Client: NetworkClient>> {
    /// Maps tensor transfer IDs to their exposed state.
    pub exposed_tensors: Mutex<HashMap<TensorTransferId, TensorExposeState>>,
    /// Maps node addresses to their WebSocket streams.
    pub streams: Mutex<HashMap<NetworkAddress, ClientStreamRef<N::Client>>>,
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

pub trait TensorDataServer<B: Backend, N: Network> {
    fn route_tensor_data_service(self, state: Arc<TensorDataService<B, N>>) -> Self;
}

impl<B: Backend, S: NetworkServer + Sized, N: Network<Server = S> + 'static> TensorDataServer<B, N>
    for S
{
    fn route_tensor_data_service(self, state: Arc<TensorDataService<B, N>>) -> Self {
        self.route("/data", async move |stream: S::Stream| {
            state.handle_data_stream(stream).await;
        })
    }
}

impl<B: Backend, N: Network> TensorDataService<B, N> {
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
    pub async fn expose(
        &self,
        tensor: B::FloatTensorPrimitive,
        max_downloads: u32,
        transfer_id: TensorTransferId,
    ) {
        let data = B::float_into_data(tensor).await;
        self.expose_data(data, max_downloads, transfer_id).await
    }

    /// Exposes a tensor data to the data server, allowing it to be downloaded by other nodes.
    pub async fn expose_data(
        &self,
        tensor_data: TensorData,
        max_downloads: u32,
        transfer_id: TensorTransferId,
    ) {
        let bytes: bytes::Bytes = rmp_serde::to_vec(&Message::Data(tensor_data))
            .unwrap()
            .into();
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

    pub async fn close(&self) {
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
    pub async fn download_tensor(
        &self,
        remote: NetworkAddress,
        transfer_id: TensorTransferId,
    ) -> Option<TensorData> {
        log::info!("Downloading tensor from {remote:?}");

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
    async fn get_data_stream(
        &self,
        address: NetworkAddress,
    ) -> Arc<Mutex<<N::Client as NetworkClient>::Stream>> {
        let mut streams = self.streams.lock().await;
        match streams.get(&address) {
            Some(stream) => stream.clone(),
            None => {
                // Open a new WebSocket connection to the address
                let stream = N::Client::connect(address.clone(), "data").await;

                let Some(stream) = stream else {
                    panic!("Failed to connect to data server at {address:?}");
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

    /// Handle incoming connections for downloading tensors.
    pub(crate) async fn handle_data_stream(
        &self,
        mut stream: <N::Server as NetworkServer>::Stream,
    ) {
        log::info!("[Data Handler] New connection for download.");

        while !self.cancel_token.is_cancelled() {
            match stream.recv().await {
                Ok(message) => {
                    if let Some(msg) = message {
                        let bytes = msg.data;
                        let msg: Message = rmp_serde::from_slice(&bytes)
                            .expect("Can deserialize messages from the websocket.");
                        let Message::Request(transfer_id) = msg else {
                            panic!("Received a message that wasn't a tensor request! {msg:?}");
                        };

                        let bytes = self.get_exposed_tensor_bytes(transfer_id).await.unwrap();

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
