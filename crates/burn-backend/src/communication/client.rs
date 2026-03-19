use std::{sync::mpsc::Sender, thread::spawn};

use crate::{
    Backend, DistributedParams, ops::TensorRef, server::GradientSyncServer, tensor::Device,
};

pub(crate) enum ActionMessage<B: Backend> {
    Message(GradientSyncMessage<B>),
    Close(),
}

pub(crate) enum GradientSyncMessage<B: Backend> {
    RegisterSyncParameters(Vec<DistributedParams>),
    GradientSync((TensorRef<B>, DistributedParams)),
    CollectiveSync((Device<B>, oneshot::Sender<Box<dyn FnOnce() + Send>>)),
}

#[derive(Clone)]
pub struct GradientSyncClient<B: Backend> {
    sender: Sender<ActionMessage<B>>,
}

impl<B: Backend> GradientSyncClient<B> {
    pub(crate) fn new(devices: Vec<B::Device>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();

        let mut server = GradientSyncServer::new(devices);
        spawn(move || {
            loop {
                match rx.recv().expect("Gradient sync server disconnected.") {
                    ActionMessage::Message(msg) => server.process_message(msg),
                    ActionMessage::Close() => break,
                }
            }
        });
        Self { sender: tx }
    }

    pub fn register_sync_parameters(&self, sharded_params: Vec<DistributedParams>) {
        self.sender
            .send(ActionMessage::Message(
                GradientSyncMessage::RegisterSyncParameters(sharded_params),
            ))
            .unwrap();
    }

    pub fn submit_gradient_sync(&self, tensor: TensorRef<B>, params: DistributedParams) {
        self.sender
            .send(ActionMessage::Message(GradientSyncMessage::GradientSync((
                tensor, params,
            ))))
            .unwrap();
    }

    pub fn submit_sync_collective(&self, device: Device<B>) {
        let (tx, rx) = oneshot::channel();

        self.sender
            .send(ActionMessage::Message(GradientSyncMessage::CollectiveSync(
                (device.clone(), tx),
            )))
            .unwrap();

        let sync = rx.recv().expect("Can receive callback");
        sync();
    }

    pub(crate) fn close(&self) {
        self.sender.send(ActionMessage::Close()).unwrap();
    }
}
