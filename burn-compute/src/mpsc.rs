use crate::{ComputeChannel, ComputeServer, Handle, MemoryHandle};
use std::{
    sync::{mpsc, Arc},
    thread,
};

pub struct MpscComputeChannel<Server>
where
    Server: ComputeServer,
{
    state: Arc<MpscComputeChannelState<Server>>,
}

pub struct MpscComputeChannelState<Server>
where
    Server: ComputeServer,
{
    _handle: thread::JoinHandle<()>,
    sender: mpsc::SyncSender<Message<Server>>,
}

type Callback<Response> = mpsc::SyncSender<Response>;

pub enum Message<Server>
where
    Server: ComputeServer,
{
    Read(Handle<Server>, Callback<Vec<u8>>),
    Create(Vec<u8>, Callback<Handle<Server>>),
    Empty(usize, Callback<Handle<Server>>),
    Execute(Server::Kernel, Vec<Handle<Server>>),
    Sync(Callback<()>),
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
    fn new(mut server: Server) -> Self {
        let (sender, receiver) = mpsc::sync_channel(10);

        let _handle = thread::spawn(move || {
            while let Ok(message) = receiver.recv() {
                match message {
                    Message::Read(handle, callback) => {
                        let data = server.read(&handle);
                        core::mem::drop(handle);
                        callback.send(data).unwrap();
                    }
                    Message::Create(data, callback) => {
                        let handle = server.create(data);
                        callback.send(handle).unwrap();
                    }
                    Message::Empty(size, callback) => {
                        let handle = server.empty(size);
                        callback.send(handle).unwrap();
                    }
                    Message::Execute(kernel, handles) => {
                        server.execute(kernel, &handles.iter().map(|h| h).collect::<Vec<_>>());
                    }
                    Message::Sync(callback) => {
                        server.sync();
                        callback.send(()).unwrap();
                    }
                };
            }
        });

        let state = Arc::new(MpscComputeChannelState { sender, _handle });

        Self { state }
    }

    fn read(&self, handle: &Handle<Server>) -> Vec<u8> {
        let (callback, response) = mpsc::sync_channel(1);

        self.state
            .sender
            .send(Message::Read(handle.compute_reference(), callback))
            .unwrap();

        self.response(response)
    }

    fn create(&self, data: Vec<u8>) -> Handle<Server> {
        let (callback, response) = mpsc::sync_channel(1);

        self.state
            .sender
            .send(Message::Create(data, callback))
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
            .send(Message::Execute(
                kernel,
                handles
                    .into_iter()
                    .map(|h| (*h).compute_reference())
                    .collect::<Vec<Handle<Server>>>(),
            ))
            .unwrap()
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
