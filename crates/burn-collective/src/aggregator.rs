use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::{
        Mutex,
        mpsc::{Receiver, SyncSender},
    },
};

use burn_tensor::backend::Backend;

use crate::{
    AggregateParams, AggregateStrategy, centralized::all_reduce_centralized, ring::all_reduce_ring,
    tree::all_reduce_tree,
};

/// Used by the aggregator thread that manages all the collective aggregation operations
/// (like all-reduce).
/// This thread takes in messages from different clients. The clients must register, than they can
/// send an aggregate message. They must all use the same parameters for the same aggregate
/// operation.
pub struct Aggregator<B: Backend> {
    /// Channel receiver for messages from clients
    message_rec: Receiver<Message<B>>,

    /// The ids passed to each register so far
    registered_nodes: Vec<u32>,
    /// The params of the current operation, as defined by the first caller
    cur_params: Option<AggregateParams>,
    /// The tensor primitives passed by each operation call
    tensors: Vec<B::FloatTensorPrimitive>,
    /// Callbacks for when all registers are done
    callbacks_register: Vec<SyncSender<()>>,
    /// Callbacks for when aggregate is done
    callbacks_aggregate: Vec<SyncSender<B::FloatTensorPrimitive>>,
}

#[derive(Debug)]
pub(crate) enum Message<B: Backend> {
    Aggregate {
        tensor: B::FloatTensorPrimitive,
        params: AggregateParams,
        callback: SyncSender<B::FloatTensorPrimitive>,
    },
    Register {
        id: u32,
        num_nodes: u32,
        callback: SyncSender<()>,
    },
    Reset,
}

#[derive(Clone)]
pub(crate) struct AggregatorClient<B: Backend> {
    channel: SyncSender<Message<B>>,
}

static STATE: Mutex<Option<HashMap<TypeId, Box<dyn Any + Send + Sync>>>> = Mutex::new(None);

pub(crate) fn aggregator<B: Backend>() -> AggregatorClient<B> {
    let mut state = STATE.lock().unwrap();

    if state.is_none() {
        *state = Some(HashMap::new());
    }
    let hashmap = state.as_mut().unwrap();

    let typeid = core::any::TypeId::of::<B>();

    let val = match hashmap.get(&typeid) {
        Some(val) => val,
        None => {
            let client = Aggregator::start();
            hashmap.insert(typeid, Box::new(client.clone()));
            return client;
        }
    };

    val.downcast_ref().cloned().unwrap()
}

impl<B: Backend> AggregatorClient<B> {
    pub fn reset(&self) {
        self.channel.send(Message::Reset).unwrap();
    }

    pub fn register(&self, id: u32, num_nodes: u32) {
        let (callback, rec) = std::sync::mpsc::sync_channel::<()>(1);

        self.channel
            .send(Message::Register {
                id,
                num_nodes,
                callback,
            })
            .unwrap();

        rec.recv().unwrap();
    }

    pub fn aggregate(
        &self,
        tensor: B::FloatTensorPrimitive,
        params: AggregateParams,
    ) -> B::FloatTensorPrimitive {
        let (callback, rec) = std::sync::mpsc::sync_channel::<B::FloatTensorPrimitive>(1);

        self.channel
            .send(Message::Aggregate {
                tensor,
                params,
                callback,
            })
            .unwrap();

        // returns a tensor primitive that may or may not be on the correct device,
        // depending on the strategy used.
        rec.recv()
            .expect("Failed to receive callback from aggregator")
    }
}

impl<B: Backend> Aggregator<B> {
    fn new(rec: Receiver<Message<B>>) -> Self {
        Self {
            message_rec: rec,
            registered_nodes: vec![],
            cur_params: None,
            tensors: vec![],
            callbacks_register: vec![],
            callbacks_aggregate: vec![],
        }
    }

    /// Starts the aggregator thread
    pub(crate) fn start() -> AggregatorClient<B> {
        let (sender, rec) = std::sync::mpsc::sync_channel::<Message<B>>(50);

        let client = AggregatorClient { channel: sender };

        let _handle = std::thread::spawn(move || {
            let mut aggregator = Aggregator::new(rec);

            while let Ok(message) = aggregator.message_rec.recv() {
                aggregator.process_message(message);
            }

            log::debug!("Aggregator message failed");
        });

        client
    }

    fn process_message(&mut self, message: Message<B>) {
        match message {
            Message::Aggregate {
                tensor,
                params,
                callback,
            } => {
                if self.tensors.is_empty() || self.cur_params.is_none() {
                    self.cur_params = Some(params);
                } else if *self.cur_params.as_ref().unwrap() != params {
                    panic!(
                        "Trying to aggregate a different way ({:?}) than is currently
                            being done ({:?})",
                        params, self.cur_params,
                    );
                }

                self.tensors.push(tensor);
                self.callbacks_aggregate.push(callback);
            }
            Message::Register {
                id,
                num_nodes,
                callback,
            } => {
                if self.registered_nodes.contains(&id) {
                    panic!("Cannot register a node twice!");
                }
                self.registered_nodes.push(id);
                self.callbacks_register.push(callback);
                if self.registered_nodes.len() == num_nodes as usize {
                    for callback in self.callbacks_register.drain(..) {
                        callback.send(()).unwrap();
                    }
                }
            }
            Message::Reset => {
                self.registered_nodes.clear();
                self.tensors.clear();
                self.cur_params = None;
            }
        }

        let tensor_count = self.tensors.len();
        if tensor_count > 0 && tensor_count == self.registered_nodes.len() {
            // all registered callers have sent a tensor to aggregate
            self.do_aggregation()
        }
    }

    fn do_aggregation(&mut self) {
        let kind = &self.cur_params.as_ref().unwrap().kind;
        let strategy = &self.cur_params.as_ref().unwrap().strategy;
        let tensor_count = self.tensors.len();

        let mut outs = match strategy {
            AggregateStrategy::Centralized => {
                let out = all_reduce_centralized::<B>(&mut self.tensors, kind);
                vec![out; tensor_count]
            }
            AggregateStrategy::Tree(arity) => {
                let out = all_reduce_tree::<B>(&mut self.tensors, kind, *arity);
                vec![out; tensor_count]
            }
            AggregateStrategy::Ring => all_reduce_ring::<B>(&mut self.tensors, kind),
        };

        // Callbacks return results
        for callback in self.callbacks_aggregate.drain(..) {
            let out = outs.remove(0);
            callback.send(out).unwrap();
        }
    }
}
