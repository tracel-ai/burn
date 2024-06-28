use burn_common::{reader::Reader, sync_type::SyncType};
use std::{sync::Arc, thread};

use super::ComputeChannel;
use crate::{
    server::{Binding, ComputeServer, Handle},
    storage::ComputeStorage,
};

/// Create a channel using a [multi-producer, single-consumer channel to communicate with
/// the compute server spawn on its own thread.
#[derive(Debug)]
pub struct MpscComputeChannel<Server>
where
    Server: ComputeServer,
{
    state: Arc<MpscComputeChannelState<Server>>,
}

#[derive(Debug)]
struct MpscComputeChannelState<Server>
where
    Server: ComputeServer,
{
    _handle: thread::JoinHandle<()>,
    sender: async_channel::Sender<Message<Server>>,
}

type Callback<Response> = async_channel::Sender<Response>;

enum Message<Server>
where
    Server: ComputeServer,
{
    Read(Binding<Server>, Callback<Vec<u8>>),
    GetResource(
        Binding<Server>,
        Callback<<Server::Storage as ComputeStorage>::Resource>,
    ),
    Create(Vec<u8>, Callback<Handle<Server>>),
    Empty(usize, Callback<Handle<Server>>),
    ExecuteKernel(Server::Kernel, Vec<Binding<Server>>),
    Sync(SyncType, Callback<()>),
}

impl<Server> MpscComputeChannel<Server>
where
    Server: ComputeServer + 'static,
{
    /// Create a new mpsc compute channel.
    pub fn new(mut server: Server) -> Self {
        let (sender, receiver) = async_channel::unbounded();

        let _handle = thread::spawn(move || {
            // Run the whole procedure as one blocking future. This is much simpler than trying
            // to use some multithreaded executor.
            pollster::block_on(async {
                while let Ok(message) = receiver.recv().await {
                    match message {
                        Message::Read(binding, callback) => {
                            let data = server.read(binding).await;
                            callback.send(data).await.unwrap();
                        }
                        Message::GetResource(binding, callback) => {
                            let data = server.get_resource(binding);
                            callback.send(data).await.unwrap();
                        }
                        Message::Create(data, callback) => {
                            let handle = server.create(&data);
                            callback.send(handle).await.unwrap();
                        }
                        Message::Empty(size, callback) => {
                            let handle = server.empty(size);
                            callback.send(handle).await.unwrap();
                        }
                        Message::ExecuteKernel(kernel, bindings) => {
                            server.execute(kernel, bindings);
                        }
                        Message::Sync(sync_type, callback) => {
                            server.sync(sync_type);
                            callback.send(()).await.unwrap();
                        }
                    };
                }
            });
        });

        let state = Arc::new(MpscComputeChannelState { sender, _handle });

        Self { state }
    }
}

impl<Server: ComputeServer> Clone for MpscComputeChannel<Server> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<Server> ComputeChannel<Server> for MpscComputeChannel<Server>
where
    Server: ComputeServer + 'static,
{
    fn read(&self, binding: Binding<Server>) -> Reader {
        let sender = self.state.sender.clone();

        Box::pin(async move {
            let (callback, response) = async_channel::unbounded();
            sender.send(Message::Read(binding, callback)).await.unwrap();
            handle_response(response.recv().await)
        })
    }

    fn get_resource(
        &self,
        binding: Binding<Server>,
    ) -> <Server::Storage as ComputeStorage>::Resource {
        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::GetResource(binding, callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn create(&self, data: &[u8]) -> Handle<Server> {
        let (callback, response) = async_channel::unbounded();

        self.state
            .sender
            .send_blocking(Message::Create(data.to_vec(), callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn empty(&self, size: usize) -> Handle<Server> {
        let (callback, response) = async_channel::unbounded();
        self.state
            .sender
            .send_blocking(Message::Empty(size, callback))
            .unwrap();

        handle_response(response.recv_blocking())
    }

    fn execute(&self, kernel: Server::Kernel, bindings: Vec<Binding<Server>>) {
        self.state
            .sender
            .send_blocking(Message::ExecuteKernel(kernel, bindings))
            .unwrap()
    }

    fn sync(&self, sync_type: SyncType) {
        let (callback, response) = async_channel::unbounded();
        self.state
            .sender
            .send_blocking(Message::Sync(sync_type, callback))
            .unwrap();
        handle_response(response.recv_blocking())
    }
}

fn handle_response<Response, Err: core::fmt::Debug>(response: Result<Response, Err>) -> Response {
    match response {
        Ok(val) => val,
        Err(err) => panic!("Can't connect to the server correctly {err:?}"),
    }
}
