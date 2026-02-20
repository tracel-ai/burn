use std::{sync::mpsc::Sender, thread::spawn};

use burn_backend::{
    Backend, ShardedParams,
    tensor::{CommunicationTensor, FloatTensor},
};

use crate::{NodeId, collections::HashMap, grad_sync::server::GradientSyncServer};

pub(crate) enum GradientSyncMessage<B: Backend> {
    RegisterDevice((HashMap<NodeId, usize>, HashMap<NodeId, ShardedParams>)),
    Register((NodeId, CommunicationTensor<B>)),
    IsFinished(Sender<bool>),
}

#[derive(Clone)]
pub struct GradientSyncClient<B: Backend> {
    sender: Sender<GradientSyncMessage<B>>,
}

impl<B: Backend> GradientSyncClient<B> {
    pub(crate) fn new() -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut server = GradientSyncServer::default();
        spawn(move || {
            loop {
                match rx.recv() {
                    Ok(msg) => server.process_message(msg),
                    Err(err) => panic!("Gradient sync server failed to receive message: {err}."),
                }
                // TODO: server_stop in api.
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
            .send(GradientSyncMessage::RegisterDevice((
                n_required_map,
                sharded_params_map,
            )))
            .unwrap();
    }

    pub fn on_register(&self, id: NodeId, tensor: &mut FloatTensor<B>) {
        self.sender
            .send(GradientSyncMessage::Register((
                id,
                B::comm_duplicated(tensor),
            )))
            .unwrap();
    }

    pub fn wait_gradients_sync(&self) {
        let (tx, rx) = std::sync::mpsc::channel();
        // TODO: Efficiency?
        loop {
            self.sender
                .send(GradientSyncMessage::IsFinished(tx.clone()))
                .unwrap();
            if rx.recv().unwrap() {
                break;
            }
        }
    }
}
