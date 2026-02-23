use std::{sync::mpsc::Sender, thread::spawn};

use burn_backend::{
    Backend, ShardedParams,
    tensor::{CommunicationTensor, FloatTensor},
};

use crate::{NodeId, collections::HashMap, grad_sync::server::GradientSyncServer};

pub(crate) enum MessageAction<B: Backend> {
    Message(GradientSyncMessage<B>),
    Close(),
}

pub(crate) enum GradientSyncMessage<B: Backend> {
    RegisterDevice((HashMap<NodeId, usize>, HashMap<NodeId, ShardedParams>)),
    Register((NodeId, CommunicationTensor<B>)),
    IsFinished(Sender<bool>),
}

#[derive(Clone)]
pub struct GradientSyncClient<B: Backend> {
    sender: Sender<MessageAction<B>>,
}

impl<B: Backend> GradientSyncClient<B> {
    pub(crate) fn new(num_devices: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut server = GradientSyncServer::new(num_devices);
        spawn(move || {
            loop {
                match rx.recv() {
                    Ok(action) => match action {
                        MessageAction::Message(msg) => server.process_message(msg),
                        MessageAction::Close() => break,
                    },
                    Err(err) => panic!("Gradient sync server failed to receive message: {err}."),
                }
            }
        });
        Self { sender: tx }
    }

    pub fn register_device(
        &self,
        n_required_map: HashMap<NodeId, usize>,
        sharded_params_map: HashMap<NodeId, ShardedParams>,
    ) {
        self.sender
            .send(MessageAction::Message(GradientSyncMessage::RegisterDevice(
                (n_required_map, sharded_params_map),
            )))
            .unwrap();
    }

    pub fn on_register(&self, id: NodeId, tensor: &mut FloatTensor<B>) {
        self.sender
            .send(MessageAction::Message(GradientSyncMessage::Register((
                id,
                B::comm_duplicated(tensor),
            ))))
            .unwrap();
    }

    pub fn wait_gradients_sync(&self) {
        let (tx, rx) = std::sync::mpsc::channel();
        // TODO: Efficiency?
        loop {
            self.sender
                .send(MessageAction::Message(GradientSyncMessage::IsFinished(
                    tx.clone(),
                )))
                .unwrap();
            if rx.recv().unwrap() {
                break;
            }
        }
    }

    pub(crate) fn close(&self) {
        self.sender.send(MessageAction::Close()).unwrap();
    }
}
