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

/// A client for managing multiple streams using mutex-based synchronization.
///
/// The biggest benefit of using this client implementation is that each stream can modify its own
/// graph without blocking other streams, which is essential for multi-device training.
#[derive(Clone, new, Debug)]
pub struct MultiStreamMutexClient;

/// Manages a collection of computation streams, mapping stream IDs to their respective servers.
///
/// The `ServerLocator` is responsible for selecting and merging streams based on their IDs and parent
/// dependencies, ensuring proper synchronization and server allocation.
///
/// # Notes
///
/// Multiple stream IDs can point to the same stream, where an autodiff graph is tracked
/// across multiple threads.
pub struct ServerLocator {
    streams: HashMap<StreamId, Arc<Stream>>,
}

/// Represents a single computation stream with a mutex-protected server.
///
/// Each `Stream` contains an [AutodiffServer] and the original [StreamId] where the server was
/// first created.
struct Stream {
    server: Mutex<AutodiffServer>,
    stream_id: StreamId,
}

/// Global static mutex for storing the server locator state.
///
/// This ensures thread-safe access to the [ServerLocator] instance.
static STATE: spin::Mutex<Option<ServerLocator>> = spin::Mutex::new(None);

impl MultiStreamMutexClient {
    /// Retrieves or creates a stream for the given stream ID and parent dependencies.
    ///
    /// # Parameters
    /// - `stream_id`: The unique identifier for the stream.
    /// - `parents`: A slice of parent nodes that the stream depends on.
    ///
    /// # Returns
    /// An `Arc<Stream>` representing the selected or newly created stream.
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
        log::info!(
            "Backward from stream {stream_id} on server {}",
            stream.stream_id
        );

        let node_id = root.node.id;
        let grads = Gradients::new::<B>(root.node, root.primitive);
        let mut server = stream.server.lock().unwrap();

        server.backward(grads, node_id)
    }
}

impl ServerLocator {
    /// Selects a single stream for the given stream ID, considering parent dependencies.
    ///
    /// If multiple streams are found, they are merged into a single stream.
    ///
    /// # Parameters
    /// - `stream_id`: The ID of the stream to select.
    /// - `parents`: A slice of parent nodes that the stream depends on.
    ///
    /// # Returns
    /// An `Arc<Stream>` representing the selected or merged stream.
    fn select(&mut self, stream_id: StreamId, parents: &[Parent]) -> Arc<Stream> {
        let mut streams = self.select_many(stream_id, parents);

        if streams.len() == 1 {
            let stream = streams.pop().unwrap();
            if stream.stream_id != stream_id {
                self.streams.insert(stream_id, stream.clone());
            }

            return stream;
        }

        self.merge(stream_id, streams)
    }

    /// Selects multiple streams based on the stream ID and parent dependencies.
    ///
    /// # Parameters
    /// - `stream_id`: The ID of the stream to select.
    /// - `parents`: A slice of parent nodes that the stream depends on.
    ///
    /// # Returns
    /// A vector of `Arc<Stream>` containing the selected streams.
    fn select_many(&mut self, stream_id: StreamId, parents: &[Parent]) -> Vec<Arc<Stream>> {
        let mut servers = HashMap::<StreamId, Arc<Stream>>::new();

        if let Some(val) = self.streams.get(&stream_id) {
            if parents.is_empty() {
                return vec![val.clone()];
            }
            servers.insert(val.stream_id, val.clone());
        }

        for parent in parents {
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

                    self.streams.insert(stream_id, server.clone());
                    vec![server]
                }
            };
        }

        servers.drain().map(|(_, v)| v).collect()
    }

    /// Merges multiple streams into a single stream and updates the stream mappings.
    ///
    /// # Parameters
    /// - `stream_id`: The ID of the target stream to merge into.
    /// - `streams`: A vector of streams to merge.
    ///
    /// # Returns
    /// An `Arc<Stream>` representing the merged stream.
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

        for sid in stream_ids {
            self.streams.insert(sid, main.clone());
        }
        self.streams.insert(stream_id, main.clone());

        core::mem::drop(server);

        main
    }
}
