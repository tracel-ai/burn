use std::{
    cmp::{self},
    sync::mpsc::SyncSender,
};

use burn_tensor::{
    ElementConversion,
    backend::{Backend, DeviceOps},
};

pub struct Aggregator {}

pub type AggregationId = u32;

#[derive(Debug, PartialEq, Clone)]
pub enum AggregateStrategy {
    Centralized,
    Tree(u32),
    Ring,
}

#[derive(Debug, PartialEq, Clone)]
pub enum AggregateKind {
    Sum,
    Mean,
}

#[derive(Debug, PartialEq, Clone)]
pub struct AggregateParams {
    pub kind: AggregateKind,
    pub strategy: AggregateStrategy,
}

#[derive(Debug)]
pub enum Message<B: Backend> {
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
pub struct AggregatorClient<B: Backend> {
    channel: SyncSender<Message<B>>,
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
        kind: AggregateKind,
        strategy: AggregateStrategy,
    ) -> B::FloatTensorPrimitive {
        let (callback, rec) = std::sync::mpsc::sync_channel::<B::FloatTensorPrimitive>(1);
        let params = AggregateParams { kind, strategy };

        self.channel
            .send(Message::Aggregate {
                tensor,
                params,
                callback,
            })
            .unwrap();

        rec.recv()
            .expect("Failed to receive callback from aggregator")
    }
}

impl Aggregator {
    pub fn start<B: Backend>() -> AggregatorClient<B> {
        let (sender, rec) = std::sync::mpsc::sync_channel::<Message<B>>(50);

        let client = AggregatorClient { channel: sender };

        let _handle = std::thread::spawn(move || {
            let mut registered_nodes = vec![];
            let mut cur_params = None;

            let mut tensors = Vec::new();
            let mut callbacks_register = Vec::new();
            let mut callbacks_aggregate = Vec::new();

            while let Ok(message) = rec.recv() {
                match message {
                    Message::Aggregate {
                        tensor,
                        params,
                        callback,
                    } => {
                        if tensors.is_empty() || cur_params.is_none() {
                            cur_params = Some(params);
                        } else if *cur_params.as_ref().unwrap() != params {
                            panic!(
                                "Trying to aggregate a different way ({:?}) than is currently
                                    being done ({:?})",
                                params, cur_params,
                            );
                        }

                        tensors.push(tensor);
                        callbacks_aggregate.push(callback);
                    }
                    Message::Register {
                        id,
                        num_nodes,
                        callback,
                    } => {
                        if registered_nodes.contains(&id) {
                            panic!("Cannot register a node twice!");
                        }
                        registered_nodes.push(id);
                        callbacks_register.push(callback);
                        if registered_nodes.len() == num_nodes as usize {
                            for callback in callbacks_register.drain(..) {
                                callback.send(()).unwrap();
                            }
                        }
                    }
                    Message::Reset => {
                        registered_nodes.clear();
                        tensors.clear();
                        cur_params = None;
                    }
                }

                let tensor_count = tensors.len();
                if tensor_count > 0 && tensor_count == registered_nodes.len() {
                    let kind = &cur_params.as_ref().unwrap().kind;
                    let strategy = &cur_params.as_ref().unwrap().strategy;
                    let out = match &strategy {
                        AggregateStrategy::Centralized => {
                            aggregate_centralized::<B>(&mut tensors, kind)
                        }
                        AggregateStrategy::Tree(arity) => {
                            aggregate_tree::<B>(&mut tensors, kind, *arity)
                        }
                        AggregateStrategy::Ring => aggregate_ring::<B>(&mut tensors, kind),
                    };

                    for callback in callbacks_aggregate.drain(..) {
                        callback.send(out.clone()).unwrap();
                    }
                }
            }

            log::debug!("Aggregator message failed");
        });

        client
    }
}

fn aggregate_centralized<B: Backend>(
    tensors: &mut Vec<B::FloatTensorPrimitive>,
    kind: &AggregateKind,
) -> B::FloatTensorPrimitive {
    let tensor_count = tensors.len();
    let mut base = tensors.pop().unwrap();

    for tensor in tensors.drain(..) {
        let target_device = B::float_device(&base);
        let tensor = B::float_to_device(tensor, &target_device);
        base = B::float_add(base, tensor);
    }

    if *kind == AggregateKind::Mean {
        base = B::float_div_scalar(base, (tensor_count as f32).elem());
    }

    base
}

fn aggregate_tree<B: Backend>(
    tensors: &mut Vec<B::FloatTensorPrimitive>,
    kind: &AggregateKind,
    arity: u32,
) -> B::FloatTensorPrimitive {
    // Sort by device id
    tensors.sort_by(|a, b| {
        let dev_a = B::float_device(a).id();
        let dev_b = B::float_device(b).id();

        dev_a.cmp(&dev_b)
    });

    let tensor_count = tensors.len() as u32;
    let mut result = if tensor_count > arity {
        // Split tensor vec into chunks
        let chunk_count = cmp::min(arity, tensor_count);
        let chunk_size = tensor_count / chunk_count;
        let chunks: Vec<Vec<B::FloatTensorPrimitive>> = tensors
            .chunks(chunk_size as usize)
            .map(|s| s.into())
            .collect();

        // Recursive reduce
        let mut new_tensors = vec![];
        for mut chunk in chunks {
            new_tensors.push(aggregate_tree::<B>(&mut chunk, kind, arity));
        }
        aggregate_centralized::<B>(&mut new_tensors, &AggregateKind::Sum)
    } else {
        aggregate_centralized::<B>(tensors, &AggregateKind::Sum)
    };

    if *kind == AggregateKind::Mean {
        result = B::float_div_scalar(result, (tensor_count as f32).elem());
    }

    result
}

fn aggregate_ring<B: Backend>(
    _tensors: &mut Vec<B::FloatTensorPrimitive>,
    _kind: &AggregateKind,
) -> B::FloatTensorPrimitive {
    unimplemented!()
}
