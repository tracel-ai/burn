use std::sync::mpsc::SyncSender;

use burn_tensor::backend::Backend;

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
        let (callback, rec) = std::sync::mpsc::sync_channel::<()>(1);

        self.channel
            .send(Message::Register {
                num_nodes,
                callback,
            })
            .unwrap();

        rec.recv().unwrap();
    }

    pub fn aggregate(&self, tensor: B::FloatTensorPrimitive) -> B::FloatTensorPrimitive {
        let (callback, rec) = std::sync::mpsc::sync_channel::<B::FloatTensorPrimitive>(1);
        self.channel
            .send(Message::Aggregate { tensor, callback })
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
            let mut num_nodes_registered = 0;

            let mut tensors = Vec::new();
            let mut callbacks_register = Vec::new();
            let mut callbacks_aggregate = Vec::new();

            while let Ok(message) = rec.recv() {
                match message {
                    Message::Aggregate { tensor, callback } => {
                        tensors.push(tensor);
                        callbacks_aggregate.push(callback);
                    }
                    Message::Register {
                        num_nodes,
                        callback,
                    } => {
                        num_nodes_registered += 1;
                        callbacks_register.push(callback);
                        if num_nodes_registered == num_nodes {
                            for callback in callbacks_register.drain(..) {
                                callback.send(()).unwrap();
                            }
                        }
                    }
                }

                if tensors.len() == num_nodes_registered as usize {
                    let mut base = tensors.pop().unwrap();

                    for tensor in tensors.drain(..) {
                        let target_device = B::float_device(&base);
                        let tensor = B::float_to_device(tensor, &target_device);
                        base = B::float_add(base, tensor);
                    }

                    for callback in callbacks_aggregate.drain(..) {
                        callback.send(base.clone()).unwrap();
                    }
                    num_nodes_registered = 0;
                }
            }

            log::debug!("Aggregator message failed");
        });

        client
    }
}
