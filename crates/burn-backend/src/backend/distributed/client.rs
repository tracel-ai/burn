use std::{sync::mpsc::Sender, thread::spawn};

use crate::tensor::Device;

use crate::distributed::{
    DistributedBackend, DistributedConfig, DistributedParams, TensorRef,
    server::DistributedSyncServer,
};

pub(crate) enum ActionMessage<B: DistributedBackend> {
    Message(DistributedSyncMessage<B>),
    Close(),
}

pub(crate) enum DistributedSyncMessage<B: DistributedBackend> {
    RegisterSyncParameters(Vec<DistributedParams>),
    TensorSync((TensorRef<B>, DistributedParams)),
    #[allow(clippy::type_complexity)]
    CollectiveSync((Device<B>, oneshot::Sender<Box<dyn FnOnce() + Send>>)),
}

#[derive(Clone)]
pub struct DistributedSyncClient<B: DistributedBackend> {
    sender: Sender<ActionMessage<B>>,
}

impl<B: DistributedBackend> DistributedSyncClient<B> {
    pub(crate) fn new(num_devices: usize, config: DistributedConfig) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();

        let mut server = DistributedSyncServer::new(num_devices, config);
        spawn(move || {
            while let ActionMessage::Message(msg) =
                rx.recv().expect("Gradient sync server disconnected.")
            {
                server.process_message(msg)
            }
        });
        Self { sender: tx }
    }

    pub fn register_sync_parameters(&self, sharded_params: Vec<DistributedParams>) {
        self.sender
            .send(ActionMessage::Message(
                DistributedSyncMessage::RegisterSyncParameters(sharded_params),
            ))
            .unwrap();
    }

    pub fn submit_gradient_sync(&self, tensor: TensorRef<B>, params: DistributedParams) {
        self.sender
            .send(ActionMessage::Message(DistributedSyncMessage::TensorSync(
                (tensor, params),
            )))
            .unwrap();
    }

    pub fn submit_sync_collective(&self, device: Device<B>) {
        let (tx, rx) = oneshot::channel();

        self.sender
            .send(ActionMessage::Message(
                DistributedSyncMessage::CollectiveSync((device.clone(), tx)),
            ))
            .unwrap();

        let sync = rx.recv().expect("Can receive callback");

        sync();
    }

    pub(crate) fn close(&self) {
        self.sender.send(ActionMessage::Close()).unwrap();
    }
}
