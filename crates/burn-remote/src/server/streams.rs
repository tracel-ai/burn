use std::collections::HashMap;

use burn_std::id::StreamId;

/// The server stream each client stream of one session runs on.
///
/// A client numbers its streams with a counter local to its process, so two clients send the
/// same small ids. Run under those ids, one session's operations would share a stream with
/// another's, and be recorded into the graph that session captures on it. Each session's streams
/// are instead allocated by the server the first time the client names them, and never shared
/// with another session.
#[derive(Default)]
pub(crate) struct ServerStreams {
    streams: HashMap<StreamId, StreamId>,
}

impl ServerStreams {
    /// The server stream the client's `stream` runs on.
    pub(crate) fn of(&mut self, stream: StreamId) -> StreamId {
        *self
            .streams
            .entry(stream)
            .or_insert_with(StreamId::allocate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_sessions_naming_one_client_stream_run_on_two_server_streams() {
        let client = StreamId { value: 0 };
        let mut first = ServerStreams::default();
        let mut second = ServerStreams::default();

        let stream = first.of(client);
        assert_eq!(first.of(client), stream, "a client stream moved");
        assert_ne!(second.of(client), stream);
    }
}
