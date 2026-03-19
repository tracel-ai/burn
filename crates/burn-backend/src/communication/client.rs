use std::{sync::mpsc::Sender, thread::spawn};

use crate::{
    Backend, DistributedParams, ops::TensorRef, server::GradientSyncServer, tensor::Device,
};

pub(crate) enum MessageAction<B: Backend> {
    Message(GradientSyncMessage<B>),
    Close(),
}

pub(crate) enum GradientSyncMessage<B: Backend> {
    RegisterDevice(Vec<DistributedParams>),
    Register((TensorRef<B>, DistributedParams)),
    Sync((Device<B>, oneshot::Sender<Box<dyn FnOnce() + Send>>)),
}

#[derive(Clone)]
pub struct GradientSyncClient<B: Backend> {
    sender: Sender<MessageAction<B>>,
}

impl<B: Backend> GradientSyncClient<B> {
    pub(crate) fn new(devices: Vec<B::Device>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();

        let mut server = GradientSyncServer::new(devices);
        spawn(move || {
            loop {
                match rx.recv().expect("Gradient sync server disconnected.") {
                    MessageAction::Message(msg) => server.process_message(msg),
                    MessageAction::Close() => {
                        break;
                    }
                }
            }
        });
        Self { sender: tx }
    }

    pub fn register_device(&self, sharded_params: Vec<DistributedParams>) {
        self.sender
            .send(MessageAction::Message(GradientSyncMessage::RegisterDevice(
                sharded_params,
            )))
            .unwrap();
    }

    pub fn submit_gradient_sync(&self, tensor: TensorRef<B>, params: DistributedParams) {
        self.sender
            .send(MessageAction::Message(GradientSyncMessage::Register((
                tensor, params,
            ))))
            .unwrap();
    }

    pub fn wait_gradients_sync(&self, device: Device<B>) {
        let (tx, rx) = oneshot::channel();

        self.sender
            .send(MessageAction::Message(GradientSyncMessage::Sync((
                device.clone(),
                tx,
            ))))
            .unwrap();

        let task = rx.recv().expect("Can receive callback");
        task();
    }

    pub(crate) fn close(&self) {
        self.sender.send(MessageAction::Close()).unwrap();
    }
}
