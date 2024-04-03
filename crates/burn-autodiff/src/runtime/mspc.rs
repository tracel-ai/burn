use super::{server::AutodiffServer, AutodiffClient};
use crate::{
    checkpoint::builder::CheckpointerBuilder,
    grads::Gradients,
    graph::StepBoxed,
    tensor::{AutodiffTensor, NodeRefCount},
    NodeID,
};
use burn_tensor::backend::Backend;
use std::sync::mpsc::Sender;

static INSTANCE: spin::Lazy<ChannelClient> = spin::Lazy::new(ChannelClient::init);

#[derive(Debug, Clone)]
pub struct ChannelClient {
    sender: Sender<Message>,
}

enum Message {
    Register {
        node_id: NodeRefCount,
        step: StepBoxed,
        actions: CheckpointerBuilder,
    },
    Backward {
        node_id: NodeID,
        grads: Gradients,
        callback: Sender<Gradients>,
    },
}
impl ChannelClient {
    pub(crate) fn new() -> Self {
        INSTANCE.clone()
    }

    fn init() -> Self {
        let (sender, receiver) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            let mut server = AutodiffServer::default();

            for message in receiver.iter() {
                match message {
                    Message::Register {
                        node_id,
                        step,
                        actions,
                    } => server.register(node_id, step, actions),
                    Message::Backward {
                        node_id,
                        grads,
                        callback,
                    } => {
                        let grads = server.backward(grads, node_id);
                        callback.send(grads).unwrap();
                    }
                }
            }
        });

        Self { sender }
    }
}

impl AutodiffClient for ChannelClient {
    fn register(&self, node_id: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder) {
        self.sender
            .send(Message::Register {
                node_id,
                step,
                actions,
            })
            .unwrap()
    }

    fn backward<B: Backend, const D: usize>(&self, root: AutodiffTensor<B, D>) -> Gradients {
        let node_id = root.node.id;
        let grads = Gradients::new::<B, D>(root.node, root.primitive);
        let (callback, receiver) = std::sync::mpsc::channel();

        self.sender
            .send(Message::Backward {
                node_id,
                grads,
                callback,
            })
            .unwrap();

        match receiver.recv() {
            Ok(grads) => grads,
            Err(err) => panic!("Error during backward {err:?}"),
        }
    }
}
