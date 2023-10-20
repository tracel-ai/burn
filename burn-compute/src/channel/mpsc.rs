use std::{
    sync::{mpsc, Arc},
    thread,
};

use burn_common::reader::Reader;

use super::ComputeChannel;
use crate::{
    server::{ComputeServer, Handle},
    tune::{AutotuneOperation, AutotuneServer},
};

/// Create a channel using the [multi-producer, single-consumer channel](mpsc) to communicate with
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
    sender: mpsc::SyncSender<Message<Server>>,
}

type Callback<Response> = mpsc::SyncSender<Response>;

enum Message<Server>
where
    Server: ComputeServer,
{
    Read(Handle<Server>, Callback<Reader<Vec<u8>>>),
    Create(Vec<u8>, Callback<Handle<Server>>),
    Empty(usize, Callback<Handle<Server>>),
    ExecuteKernel(Server::Kernel, Vec<Handle<Server>>),
    Sync(Callback<()>),
    ExecuteAutotune(Box<dyn AutotuneOperation<Server>>, Vec<Handle<Server>>),
}

impl<Server> MpscComputeChannel<Server>
where
    Server: ComputeServer + 'static,
{
    /// Create a new mpsc compute channel.
    pub fn new(server: Server, bound: usize) -> Self {
        let (sender, receiver) = mpsc::sync_channel(bound);
        let mut autotune_server = AutotuneServer::new(server);

        let _handle = thread::spawn(move || {
            while let Ok(message) = receiver.recv() {
                match message {
                    Message::Read(handle, callback) => {
                        let data = autotune_server.server.read(&handle);
                        core::mem::drop(handle);
                        callback.send(data).unwrap();
                    }
                    Message::Create(data, callback) => {
                        let handle = autotune_server.server.create(&data);
                        callback.send(handle).unwrap();
                    }
                    Message::Empty(size, callback) => {
                        let handle = autotune_server.server.empty(size);
                        callback.send(handle).unwrap();
                    }
                    Message::ExecuteKernel(kernel, handles) => {
                        autotune_server
                            .server
                            .execute(kernel, &handles.iter().collect::<Vec<_>>());
                    }
                    Message::Sync(callback) => {
                        autotune_server.server.sync();
                        callback.send(()).unwrap();
                    }
                    Message::ExecuteAutotune(autotune_kernel, handles) => {
                        autotune_server
                            .execute_autotune(autotune_kernel, &handles.iter().collect::<Vec<_>>());
                    }
                };
            }
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
    fn read(&self, handle: &Handle<Server>) -> Reader<Vec<u8>> {
        let (callback, response) = mpsc::sync_channel(1);

        self.state
            .sender
            .send(Message::Read(handle.clone(), callback))
            .unwrap();

        self.response(response)
    }

    fn create(&self, data: &[u8]) -> Handle<Server> {
        let (callback, response) = mpsc::sync_channel(1);

        self.state
            .sender
            .send(Message::Create(data.to_vec(), callback))
            .unwrap();

        self.response(response)
    }

    fn empty(&self, size: usize) -> Handle<Server> {
        let (callback, response) = mpsc::sync_channel(1);

        self.state
            .sender
            .send(Message::Empty(size, callback))
            .unwrap();

        self.response(response)
    }

    fn execute(&self, kernel: Server::Kernel, handles: &[&Handle<Server>]) {
        self.state
            .sender
            .send(Message::ExecuteKernel(
                kernel,
                handles
                    .iter()
                    .map(|h| (*h).clone())
                    .collect::<Vec<Handle<Server>>>(),
            ))
            .unwrap()
    }

    fn execute_autotune(
        &self,
        autotune_kernel: Box<dyn AutotuneOperation<Server>>,
        handles: &[&Handle<Server>],
    ) {
        self.state
            .sender
            .send(Message::ExecuteAutotune(
                autotune_kernel,
                handles
                    .iter()
                    .map(|h| (*h).clone())
                    .collect::<Vec<Handle<Server>>>(),
            ))
            .unwrap();
    }

    fn sync(&self) {
        let (callback, response) = mpsc::sync_channel(1);

        self.state.sender.send(Message::Sync(callback)).unwrap();

        self.response(response)
    }
}

impl<Server: ComputeServer> MpscComputeChannel<Server> {
    fn response<Response>(&self, response: mpsc::Receiver<Response>) -> Response {
        match response.recv() {
            Ok(val) => val,
            Err(err) => panic!("Can't connect to the server correctly {err:?}"),
        }
    }
}
