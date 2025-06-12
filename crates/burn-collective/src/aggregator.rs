use std::sync::mpsc::SyncSender;

use burn_tensor::{ElementConversion, backend::Backend};

pub struct Aggregator {}

pub type AggregationId = u32;

#[derive(Debug, PartialEq)]
pub enum AggregateMethod {
    Sum,
    Mean,
}

#[derive(Debug)]
pub enum Message<B: Backend> {
    Aggregate {
        tensor: B::FloatTensorPrimitive,
        method: AggregateMethod,
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
        method: AggregateMethod,
    ) -> B::FloatTensorPrimitive {
        let (callback, rec) = std::sync::mpsc::sync_channel::<B::FloatTensorPrimitive>(1);
        self.channel
            .send(Message::Aggregate {
                tensor,
                method,
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
            let mut cur_method: Option<AggregateMethod> = None;

            let mut tensors = Vec::new();
            let mut callbacks_register = Vec::new();
            let mut callbacks_aggregate = Vec::new();

            while let Ok(message) = rec.recv() {
                match message {
                    Message::Aggregate {
                        tensor,
                        method,
                        callback,
                    } => {
                        if tensors.is_empty() || cur_method.is_none() {
                            cur_method = Some(method);
                        } else if *cur_method.as_ref().unwrap() != method {
                            panic!(
                                "Trying to aggregate a different way ({:?}) than is currently
                                    being done ({:?})",
                                method, cur_method,
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
                    }
                }

                let tensor_count = tensors.len();
                if tensor_count > 0 && tensor_count == registered_nodes.len() {
                    let mut base = tensors.pop().unwrap();

                    for tensor in tensors.drain(..) {
                        let target_device = B::float_device(&base);
                        let tensor = B::float_to_device(tensor, &target_device);
                        base = B::float_add(base, tensor);
                    }

                    if cur_method == Some(AggregateMethod::Mean) {
                        base = B::float_div_scalar(base, (registered_nodes.len() as f32).elem());
                    }

                    for callback in callbacks_aggregate.drain(..) {
                        callback.send(base.clone()).unwrap();
                    }
                }
            }

            log::debug!("Aggregator message failed");
        });

        client
    }
}
