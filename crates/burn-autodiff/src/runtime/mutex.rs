use super::{AutodiffClient, server::AutodiffServer};
use crate::{
    checkpoint::builder::CheckpointerBuilder,
    grads::Gradients,
    graph::StepBoxed,
    tensor::{AutodiffTensor, NodeRefCount},
};
use alloc::sync::Arc;
use burn_common::{id::StreamId, stub::Mutex};
use burn_tensor::backend::Backend;
use hashbrown::HashMap;

#[derive(Clone, new)]
pub struct MutexClient;

#[derive(Clone, new, Debug)]
pub struct MultiThreadMutexClient;

pub struct ServerLocator {
    streams: HashMap<StreamId, Arc<Stream>>,
}

struct Stream {
    server: Mutex<AutodiffServer>,
    stream_id: StreamId,
}

impl core::fmt::Debug for MutexClient {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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

static STATE: spin::Mutex<Option<ServerLocator>> = spin::Mutex::new(None);

impl MultiThreadMutexClient {
    fn stream(stream_id: StreamId, streams: impl Iterator<Item = StreamId>) -> Arc<Stream> {
        let mut locator = STATE.lock();

        match locator.as_mut() {
            Some(val) => val.select(stream_id, streams),
            None => {
                let mut l = ServerLocator {
                    streams: HashMap::new(),
                };
                let stream = l.select(stream_id, streams);
                *locator = Some(l);
                stream
            }
        }
    }
}

impl AutodiffClient for MultiThreadMutexClient {
    fn register(&self, node_id: NodeRefCount, step: StepBoxed, actions: CheckpointerBuilder) {
        let stream_id = StreamId::current();
        let stream =
            MultiThreadMutexClient::stream(stream_id, step.parent_streams().iter().map(|s| *s));
        let mut server = stream.server.lock().unwrap();
        server.register(node_id, step, actions);
    }

    fn backward<B: Backend>(&self, root: AutodiffTensor<B>) -> Gradients {
        let stream_id = StreamId::current();

        let stream = MultiThreadMutexClient::stream(stream_id, [].into_iter());

        let node_id = root.node.id;
        let grads = Gradients::new::<B>(root.node, root.primitive);
        let mut server = stream.server.lock().unwrap();

        server.backward(grads, node_id)
    }
}

impl ServerLocator {
    fn select(
        &mut self,
        stream_id: StreamId,
        streams: impl Iterator<Item = StreamId>,
    ) -> Arc<Stream> {
        let mut streams = self.select_many(stream_id, streams);

        if streams.len() == 1 {
            return streams.pop().unwrap();
        }

        self.merge(stream_id, streams)
    }

    fn select_many(
        &mut self,
        stream_id: StreamId,
        streams: impl Iterator<Item = StreamId>,
    ) -> Vec<Arc<Stream>> {
        let mut servers = HashMap::<StreamId, Arc<Stream>>::new();

        for parent_stream in streams {
            match self.streams.get(&parent_stream) {
                Some(val) => servers.insert(parent_stream, val.clone()),
                None => continue,
            };
        }

        if servers.len() == 0 {
            let server = Arc::new(Stream {
                server: Mutex::new(AutodiffServer::default()),
                stream_id,
            });
            self.streams.insert(stream_id, server.clone());
            return vec![server];
        }

        servers.drain().map(|(_, v)| v).collect()
    }

    fn merge(&mut self, stream_id: StreamId, mut streams: Vec<Arc<Stream>>) -> Arc<Stream> {
        let mut stream_ids = Vec::with_capacity(streams.len());
        let main = streams.pop().unwrap();

        let mut server = main.server.lock().unwrap();

        for stream in streams.drain(..) {
            let mut locked = stream.server.lock().unwrap();
            let mut ser = AutodiffServer::default();
            core::mem::swap(&mut ser, &mut locked);
            server.extend(ser);
            stream_ids.push(stream.stream_id);
        }

        core::mem::drop(server);

        for sid in stream_ids {
            self.streams.insert(sid, main.clone());
        }
        self.streams.insert(stream_id, main.clone());

        main
    }
}
