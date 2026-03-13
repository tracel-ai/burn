use std::{
    sync::{Arc, Condvar, Mutex, mpsc::Sender},
    thread::spawn,
};

use crate::{Backend, ShardedParams, ops::TensorRef, server::GradientSyncServer, tensor::Device};

pub(crate) enum MessageAction<B: Backend> {
    Message(GradientSyncMessage<B>),
    Close(Sender<()>),
}

pub(crate) enum GradientSyncMessage<B: Backend> {
    RegisterDevice(Vec<ShardedParams>),
    Register((TensorRef<B>, ShardedParams)),
    Sync((Device<B>, Arc<(Mutex<bool>, Condvar)>)),
}

#[derive(Clone)]
pub struct GradientSyncClient<B: Backend> {
    sender: Sender<MessageAction<B>>,
    is_finished_fence: Arc<(Mutex<bool>, Condvar)>,
}

impl<B: Backend> GradientSyncClient<B> {
    pub(crate) fn new(devices: Vec<B::Device>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let is_finished_fence = Arc::new((Mutex::new(false), Condvar::new()));

        let mut server = GradientSyncServer::new(devices, is_finished_fence.clone());
        spawn(move || {
            loop {
                match rx.recv().expect("Gradient sync server disconnected.") {
                    MessageAction::Message(msg) => server.process_message(msg),
                    MessageAction::Close(tx) => {
                        server.close(tx);
                        break;
                    }
                }
            }
        });
        Self {
            sender: tx,
            is_finished_fence,
        }
    }

    pub fn register_device(&self, sharded_params: Vec<ShardedParams>) {
        self.sender
            .send(MessageAction::Message(GradientSyncMessage::RegisterDevice(
                sharded_params,
            )))
            .unwrap();
    }

    pub fn on_register(&self, tensor: TensorRef<B>, params: ShardedParams) {
        self.sender
            .send(MessageAction::Message(GradientSyncMessage::Register((
                tensor, params,
            ))))
            .unwrap();
    }

    pub fn wait_gradients_sync(&self, device: Device<B>) {
        let is_synced = Arc::new((Mutex::new(false), Condvar::new()));
        self.sender
            .send(MessageAction::Message(GradientSyncMessage::Sync((
                device.clone(),
                is_synced.clone(),
            ))))
            .unwrap();

        let (lock, cvar) = &*is_synced;
        let mut synced = lock.lock().unwrap();
        while !*synced {
            synced = cvar.wait(synced).unwrap();
        }
        println!("synced client {device:?}");
    }

    pub(crate) fn close(&self) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.sender.send(MessageAction::Close(tx)).unwrap();
        rx.recv()
            .expect("Should receive `closed` callback from server");
    }
}
