use super::{RemoteClient, runner::RemoteDevice};
use crate::shared::{ConnectionId, SessionId, Task, TaskResponse, TaskResponseContent};
use burn_communication::{CommunicationChannel, Message, ProtocolClient};
use std::{collections::HashMap, marker::PhantomData, sync::Arc};

pub type CallbackSender = async_channel::Sender<TaskResponseContent>;

#[derive(Debug)]
pub enum ClientRequest {
    WithSyncCallback(Task, CallbackSender),
    WithoutCallback(Task),
}

pub(crate) struct ClientWorker<C: ProtocolClient> {
    requests: HashMap<ConnectionId, CallbackSender>,
    _p: PhantomData<C>,
}

impl<C: ProtocolClient> ClientWorker<C> {
    async fn on_response(&mut self, response: TaskResponse) {
        match self.requests.remove(&response.id) {
            Some(request) => {
                request.send(response.content).await.unwrap();
            }
            None => {
                panic!("Can't ignore message from the server.");
            }
        }
    }

    fn register_callback(&mut self, id: ConnectionId, callback: CallbackSender) {
        self.requests.insert(id, callback);
    }
}

impl<C: ProtocolClient> ClientWorker<C> {
    pub fn start(device: RemoteDevice) -> RemoteClient {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_io()
                .build()
                .unwrap(),
        );

        let (sender, rec) = async_channel::bounded(10);

        let session_id = SessionId::new();
        let address = device.address.clone();

        // Connect synchronously *before* returning the client. Doing this in a `runtime.spawn`
        // task would swallow any connection failure inside the worker (panicking the worker but
        // leaving the main thread blocked forever on the request channel). Surfacing the error
        // here means the very first op on the device fails with a clear message.
        log::info!("Connecting to {address} ...");
        let (mut stream_request, mut stream_response) = runtime.block_on(async {
            let request = match C::connect(address.clone(), "request").await {
                Ok(stream) => stream,
                Err(err) => panic!(
                    "Failed to open remote 'request' channel to {address}: {err:?}. \
                     Is a `burn-remote` server running at that address?"
                ),
            };
            let response = match C::connect(address.clone(), "response").await {
                Ok(stream) => stream,
                Err(err) => panic!(
                    "Failed to open remote 'response' channel to {address}: {err:?}. \
                     Is a `burn-remote` server running at that address?"
                ),
            };
            (request, response)
        });

        // Send the session init handshake synchronously too — a failure here is a clear
        // diagnostic, and the worker loops below assume the handshake is already done.
        let init_bytes: bytes::Bytes = rmp_serde::to_vec(&Task::Init(session_id))
            .expect("Can serialize tasks to bytes.")
            .into();

        let device_settings = runtime
            .block_on(async {
                stream_request
                    .send(Message::new(init_bytes.clone()))
                    .await?;
                stream_response.send(Message::new(init_bytes)).await?;

                // Block on the very first frame to catch the Init response payload
                let msg = stream_response
                    .recv()
                    .await?
                    .expect("Server disconnected during connection initialization");

                let response: TaskResponse = rmp_serde::from_slice(&msg.data)
                    .expect("Can deserialize init handshake payload from server");

                match response.content {
                    TaskResponseContent::Init(settings) => Ok(settings),
                    other => panic!("Expected Init handshake response, got: {:?}", other),
                }
            })
            .unwrap_or_else(|err: C::Error| {
                panic!("Failed to initialize remote session at {address}: {err:?}")
            });

        // Init device settings
        device.settings.set(device_settings).unwrap();

        let state = Arc::new(tokio::sync::Mutex::new(ClientWorker::<C>::default()));

        // Async worker loading callbacks from the server.
        let state_ws = state.clone();
        runtime.spawn(async move {
            while let Ok(msg) = stream_response.recv().await {
                let msg = match msg {
                    Some(msg) => msg,
                    None => {
                        log::warn!("Closed connection");
                        return;
                    }
                };

                let response: TaskResponse = rmp_serde::from_slice(&msg.data)
                    .expect("Can deserialize messages from the websocket.");
                let mut state = state_ws.lock().await;
                state.on_response(response).await;
            }
        });

        // Channel async worker sending operations to the server.
        runtime.spawn(async move {
            while let Ok(req) = rec.recv().await {
                let task = match req {
                    ClientRequest::WithSyncCallback(task, callback) => {
                        if let Task::Compute(_content, id) = &task {
                            let mut state = state.lock().await;
                            state.register_callback(*id, callback);
                        }
                        task
                    }
                    ClientRequest::WithoutCallback(task) => task,
                };
                let bytes = rmp_serde::to_vec(&task)
                    .expect("Can serialize tasks to bytes.")
                    .into();
                stream_request
                    .send(Message::new(bytes))
                    .await
                    .expect("Can send the message on the websocket.");
            }
        });

        RemoteClient::new(device, sender, runtime, session_id)
    }
}

impl<C: ProtocolClient> Default for ClientWorker<C> {
    fn default() -> Self {
        Self {
            requests: Default::default(),
            _p: PhantomData,
        }
    }
}
