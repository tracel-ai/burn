use std::{
    collections::HashMap,
    hash::Hash,
    sync::mpsc::{Sender, SyncSender},
};

use burn_ndarray::NdArrayTensor;
use burn_tensor::backend::Backend;

// TODO is "Cluster" a good name? Maybe "CollectiveGroup" is better

/// Operations on a cluster, which is the struct used by the collective backend extention to keep
/// track of the other units during collective operations.
pub trait ClusterOps<B: Backend> {
    fn register(
        &mut self,
        device: B::Device,
        rank: u32,
        cluster_info: ClusterMetadata,
    ) -> Result<(), String>;
    fn sync_op(&self);

    // TODO make generic for any backend
    fn set_tensor_sender(&mut self, sender: Sender<NdArrayTensor<f32>>);
    fn get_tensor_sender(&self) -> Option<Sender<NdArrayTensor<f32>>>;
}

#[derive(Debug, Clone)]
pub struct ClusterMetadata {
    pub cluster_size: usize,
}

pub struct Aggregator {}

pub type AggregationId = u32;

pub enum Message<B: Backend> {
    Aggregate {
        tensor: B::FloatTensorPrimitive,
        callback: SyncSender<B::FloatTensorPrimitive>,
    },
    Register {
        num_nodes: u32,
        callback: SyncSender<()>,
    },
}

#[derive(Clone)]
pub struct AggregatorClient<B: Backend> {
    channel: SyncSender<Message<B>>,
}

impl<B: Backend> AggregatorClient<B> {
    pub fn register(&self, num_nodes: u32) {
        let (callback, rec) = std::sync::mpsc::sync_channel::<Message<B>>(1);

        self.channel.send(Message::Register {
            num_nodes,
            callback,
        });

        if let Some(result) = rec.recv() {
            return;
        }
    }

    pub fn aggregate(&self, tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        let (callback, rec) = std::sync::mpsc::sync_channel::<Message<B>>(1);
        self.channel.send(Message::Aggregate { tensor, callback });

        if let Some(result) = rec.recv() {
            result
        } else {
            panic!("message");
        }
    }
}

impl Aggregator {
    pub fn start<B: Backend>() -> AggregatorClient<B> {
        let (sender, rec) = std::sync::mpsc::sync_channel::<Message<B>>(50);

        let client = AggregatorClient { channel: sender };

        let _handle = std::thread::spawn(move || {
            let mut num_nodes_registered = 0;

            let mut aggregations = HashMap::new();
            let mut tensors = Vec::new();
            let mut callbacks_register = Vec::new();
            let mut callbacks_aggregate = Vec::new();

            for message in rec.recv() {
                match message {
                    Message::Aggregate { tensor, callback } => {
                        tensors.push(message.tensor);
                        callbacks_aggregate.push(message.callback);
                    }
                    Message::Register {
                        num_nodes,
                        callback,
                    } => {
                        num_nodes_registered += 1;
                        callbacks_register.push(callback);
                        if num_nodes_registered == num_nodes {
                            for callback in callbacks_register.drain(..) {
                                callback.send(());
                            }
                        }
                    }
                }
            }

            if tensors.len() == num_nodes_registered {
                let mut base = tensors.pop().unwrap();

                for tensor in tensors.drain(..) {
                    base = B::float_add(base, tensor);
                }

                for callback in callbacks_aggregate.drain(..) {
                    callback.send(base.clone());
                }
                num_nodes_registered = 0;
            }
        });

        client
    }
}
