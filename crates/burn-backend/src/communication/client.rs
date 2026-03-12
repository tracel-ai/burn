use std::{
    sync::{
        Arc, Condvar, Mutex,
        mpsc::{self, Sender},
    },
    thread::spawn,
};

use crate::{Backend, ShardedParams, ops::TensorRef, server::GradientSyncServer, tensor::Device};

pub(crate) enum MessageAction<B: Backend> {
    Message(GradientSyncMessage<B>),
    Close(Sender<()>),
}

pub(crate) enum GradientSyncMessage<B: Backend> {
    RegisterDevice(Vec<ShardedParams>),
    // RegisterDevice((HashMap<u64, usize>, HashMap<u64, ShardedParams>)),
    Register((TensorRef<B>, ShardedParams)),
    Sync(Device<B>),
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
        // let fence_clone = is_finished_fence.clone();
        spawn(move || {
            loop {
                match rx.try_recv() {
                    Ok(action) => match action {
                        MessageAction::Message(msg) => server.process_message(msg),
                        MessageAction::Close(tx) => {
                            server.close(tx);
                            break;
                        }
                    },
                    Err(mpsc::TryRecvError::Empty) => {
                        // if server.is_finished() {
                        //     let (lock, cvar) = &*fence_clone;
                        //     let mut finished = lock.lock().unwrap();
                        //     *finished = true;
                        //     cvar.notify_all();
                        // }
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

    // TODO: Might be more logical to let the autodiff graph/gradients struct decide when to register the gradients to the server.
    // The "unused" parameters in the graph will just be sent right away with 0 gradients and the all_reduce is launched simply when the param_id count reaches num_devices.
    pub fn register_device(
        &self,
        // n_required_map: HashMap<u64, usize>,
        // sharded_params_map: HashMap<u64, ShardedParams>,
        sharded_params: Vec<ShardedParams>,
    ) {
        self.sender
            .send(MessageAction::Message(GradientSyncMessage::RegisterDevice(
                // (n_required_map, sharded_params_map),
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
        self.sender
            .send(MessageAction::Message(GradientSyncMessage::Sync(device)))
            .unwrap();

        let (lock, cvar) = &*self.is_finished_fence;
        let mut finished = lock.lock().unwrap();
        while !*finished {
            finished = cvar.wait(finished).unwrap();
        }
    }

    pub(crate) fn close(&self) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.sender.send(MessageAction::Close(tx)).unwrap();
        rx.recv()
            .expect("Should receive `closed` callback from server");
    }
}
