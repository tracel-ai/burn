use super::{server::AutodiffServer, AutodiffClient};
use crate::{
    checkpoint::builder::CheckpointerBuilder,
    grads::Gradients,
    graph::StepBoxed,
    tensor::{AutodiffTensor, NodeRefCount},
};
use burn_tensor::backend::Backend;

#[derive(Clone, new)]
pub struct MutexClient;

impl core::fmt::Debug for MutexClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MutexClient")
    }
}

static SERVER: spin::Mutex<Option<AutodiffServer>> = spin::Mutex::new(None);

impl AutodiffClient for MutexClient {
    fn register(&self, node_id: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder) {
        let mut server = SERVER.lock();

        if let Some(server) = server.as_mut() {
            server.register(node_id, step, actions);
            return;
        }

        let mut server_new = AutodiffServer::default();
        server_new.register(node_id, step, actions);
        *server = Some(server_new);
    }
    fn backward<B: Backend>(&self, root: AutodiffTensor<B>) -> Gradients {
        let mut server = SERVER.lock();
        let node_id = root.node.id;
        let grads = Gradients::new::<B>(root.node, root.primitive);

        if let Some(server) = server.as_mut() {
            return server.backward(grads, node_id);
        }

        let mut server_new = AutodiffServer::default();
        let gradients = server_new.backward(grads, node_id);
        *server = Some(server_new);

        gradients
    }
}
