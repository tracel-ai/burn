use std::{
    sync::{
        Arc, Condvar, Mutex,
        mpsc::{self, Sender},
    },
    thread::spawn,
};

use burn_backend::{Backend, ShardedParams, ops::TensorRef};

use crate::{NodeId, collections::HashMap, grad_sync::server::GradientSyncServer};

pub(crate) enum MessageAction<B: Backend> {
    Message(GradientSyncMessage<B>),
    Close(),
}

pub(crate) enum GradientSyncMessage<B: Backend> {
    RegisterDevice((HashMap<NodeId, usize>, HashMap<NodeId, ShardedParams>)),
    Register((NodeId, TensorRef<B>)),
}

#[derive(Clone)]
pub struct GradientSyncClient<B: Backend> {
    sender: Sender<MessageAction<B>>,
    is_finished_fence: Arc<(Mutex<bool>, Condvar)>,
}

impl<B: Backend> GradientSyncClient<B> {
    pub(crate) fn new(num_devices: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        let is_finished_fence = Arc::new((Mutex::new(false), Condvar::new()));

        let mut server = GradientSyncServer::new(num_devices, is_finished_fence.clone());
        let fence_clone = is_finished_fence.clone();
        spawn(move || {
            loop {
                match rx.try_recv() {
                    Ok(action) => match action {
                        MessageAction::Message(msg) => server.process_message(msg),
                        MessageAction::Close() => break,
                    },
                    Err(mpsc::TryRecvError::Empty) => {
                        if server.is_finished() {
                            let (lock, cvar) = &*fence_clone;
                            let mut finished = lock.lock().unwrap();
                            *finished = true;
                            cvar.notify_all();
                        }
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        panic!("Gradient sync server disconnected.")
                    }
                }
            }
        });
        Self {
            sender: tx,
            is_finished_fence,
        }
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

    pub fn on_register(&self, id: NodeId, tensor: TensorRef<B>) {
        self.sender
            .send(MessageAction::Message(GradientSyncMessage::Register((
                id, tensor,
            ))))
            .unwrap();
    }

    pub fn wait_gradients_sync(&self) {
        let (lock, cvar) = &*self.is_finished_fence;
        let mut finished = lock.lock().unwrap();
        while !*finished {
            finished = cvar.wait(finished).unwrap();
        }
    }

    pub(crate) fn close(&self) {
        self.sender.send(MessageAction::Close()).unwrap();
    }
}
