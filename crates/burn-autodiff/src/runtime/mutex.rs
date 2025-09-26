use super::{AutodiffClient, server::AutodiffServer};
use crate::{
    checkpoint::builder::CheckpointerBuilder,
    grads::Gradients,
    graph::{Parent, StepBoxed},
    tensor::{AutodiffTensor, NodeRefCount},
};
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use burn_common::{id::StreamId, stub::Mutex};
use burn_tensor::backend::Backend;
use hashbrown::HashMap;

#[derive(Clone, new, Debug)]
pub struct MultiStreamMutexClient;

pub struct ServerLocator {
    streams: HashMap<StreamId, Arc<Stream>>,
}

struct Stream {
    server: Mutex<AutodiffServer>,
    stream_id: StreamId,
}

static STATE: spin::Mutex<Option<ServerLocator>> = spin::Mutex::new(None);

impl MultiStreamMutexClient {
    fn stream(stream_id: StreamId, parents: &[Parent]) -> Arc<Stream> {
        let mut state = STATE.lock();

        match state.as_mut() {
            Some(locator) => locator.select(stream_id, parents),
            None => {
                let mut locator = ServerLocator {
                    streams: HashMap::new(),
                };
                let stream = locator.select(stream_id, parents);
                *state = Some(locator);
                stream
            }
        }
    }
}

impl AutodiffClient for MultiStreamMutexClient {
    fn register(
        &self,
        stream_id: StreamId,
        node_id: NodeRefCount,
        step: StepBoxed,
        actions: CheckpointerBuilder,
    ) {
        let stream = MultiStreamMutexClient::stream(stream_id, step.parents());
        let mut server = stream.server.lock().unwrap();
        server.register(node_id, step, actions);
    }

    fn backward<B: Backend>(&self, root: AutodiffTensor<B>) -> Gradients {
        let stream_id = StreamId::current();

        let stream = MultiStreamMutexClient::stream(stream_id, &[]);
        println!(
            "Backward on stream {} from stream {stream_id}",
            stream.stream_id
        );

        let node_id = root.node.id;
        let grads = Gradients::new::<B>(root.node, root.primitive);
        let mut server = stream.server.lock().unwrap();

        server.backward(grads, node_id)
    }
}

impl ServerLocator {
    fn select(&mut self, stream_id: StreamId, parents: &[Parent]) -> Arc<Stream> {
        let mut streams = self.select_many(StreamId { value: 0 }, &[]);
        return streams.pop().unwrap().clone();

        let mut streams = self.select_many(stream_id, parents);

        if streams.len() == 1 {
            let stream = streams.pop().unwrap();
            if stream.stream_id != stream_id {
                if let Some(current) = self.streams.get(&stream_id) {
                    assert_eq!(current.stream_id, stream.stream_id);
                }
                println!("Assign {stream_id} to server {}", stream.stream_id);
                self.streams.insert(stream_id, stream.clone());
            }

            return stream;
        }

        self.merge(stream_id, streams)
    }

    fn select_many(&mut self, stream_id: StreamId, parents: &[Parent]) -> Vec<Arc<Stream>> {
        let mut servers = HashMap::<StreamId, Arc<Stream>>::new();

        if let Some(val) = self.streams.get(&stream_id) {
            if parents.is_empty() {
                return vec![val.clone()];
            }
            servers.insert(val.stream_id, val.clone());
        }

        for parent in parents {
            // println!("{parent:?}");
            match self.streams.get(&parent.stream) {
                Some(val) => servers.insert(val.stream_id, val.clone()),
                None => continue,
            };
        }

        if servers.len() == 0 {
            return match self.streams.get(&stream_id) {
                Some(old) => vec![old.clone()],
                None => {
                    let server = Arc::new(Stream {
                        server: Mutex::new(AutodiffServer::default()),
                        stream_id,
                    });

                    println!("New stream {stream_id}");
                    self.streams.insert(stream_id, server.clone());
                    vec![server]
                }
            };
        }

        servers.drain().map(|(_, v)| v).collect()
    }

    fn merge(&mut self, stream_id: StreamId, mut streams: Vec<Arc<Stream>>) -> Arc<Stream> {
        println!("Merge on stream {stream_id}");
        let mut stream_ids = Vec::with_capacity(streams.len());
        let main = streams.pop().unwrap();

        println!("Merge main {}", main.stream_id);
        let mut server = main.server.lock().unwrap();

        for stream in streams.drain(..) {
            println!("Merge next {}", stream.stream_id);
            let mut locked = stream.server.lock().unwrap();
            let mut ser = AutodiffServer::default();
            core::mem::swap(&mut ser, &mut locked);
            server.extend(ser);
            stream_ids.push(stream.stream_id);
        }

        for sid in stream_ids {
            self.streams.insert(sid, main.clone());
        }
        self.streams.insert(stream_id, main.clone());

        println!("Drop main lock ..");
        core::mem::drop(server);

        main
    }
}

pub(crate) mod tmp {
    use super::AutodiffClient;
    use super::*;
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
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.write_str("MutexClient")
        }
    }

    static SERVER: spin::Mutex<Option<AutodiffServer>> = spin::Mutex::new(None);

    impl AutodiffClient for MutexClient {
        fn register(
            &self,
            _stream_id: StreamId,
            node_id: NodeRefCount,
            step: StepBoxed,
            actions: CheckpointerBuilder,
        ) {
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
}
