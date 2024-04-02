use super::server::AutodiffServer;
use crate::{
    checkpoint::builder::CheckpointerBuilder, grads::Gradients, graph::StepBoxed,
    tensor::AutodiffTensor, NodeID,
};
use burn_tensor::backend::Backend;

pub trait AutodiffClient: Send + Clone {
    fn register(&self, node_id: NodeID, ops: StepBoxed, actions: CheckpointerBuilder);
    fn backward<B: Backend, const D: usize>(&self, root: AutodiffTensor<B, D>) -> Gradients;
    fn drop_node(&self, node_id: NodeID);
}

#[derive(Clone)]
pub struct MutexClient;

impl core::fmt::Debug for MutexClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MutexClient")
    }
}

static SERVER: spin::Mutex<Option<AutodiffServer>> = spin::Mutex::new(None);

impl AutodiffClient for MutexClient {
    fn register(&self, node_id: NodeID, step: StepBoxed, actions: CheckpointerBuilder) {
        let mut server = SERVER.lock();

        if let Some(server) = server.as_mut() {
            server.register(node_id, step, actions);
            return;
        }

        let mut server_new = AutodiffServer::default();
        server_new.register(node_id, step, actions);
        *server = Some(server_new);
    }
    fn backward<B: Backend, const D: usize>(&self, root: AutodiffTensor<B, D>) -> Gradients {
        let mut server = SERVER.lock();

        if let Some(server) = server.as_mut() {
            return server.backward(root);
        }
        let mut server_new = AutodiffServer::default();
        let gradients = server_new.backward(root);
        *server = Some(server_new);

        gradients
    }

    fn drop_node(&self, node_id: NodeID) {}
}
