use crate::{http::server::stream::Stream, Runner};
use std::sync::Arc;

use burn_tensor::{
    backend::{Backend, BackendBridge},
    repr::ReprBackend,
};
use hashbrown::HashMap;
use std::sync::Mutex;

#[derive(Clone)]
pub struct SessionManager<B: ReprBackend> {
    state: Arc<Mutex<SessionManagerState<B>>>,
}

struct SessionManagerState<B: ReprBackend> {
    device: B::Device,
    streams: HashMap<u64, Stream<B>>,
}

impl<B: ReprBackend> SessionManager<B>
where
    // Restrict full precision backend handle to be the same
    <<B as Backend>::FullPrecisionBridge as BackendBridge<B>>::Target:
        ReprBackend<Handle = B::Handle>,
{
    pub fn new(device: B::Device) -> Self {
        Self {
            state: Arc::new(Mutex::new(SessionManagerState {
                device,
                streams: HashMap::new(),
            })),
        }
    }
    pub fn get_stream(&self, id: u64) -> Stream<B> {
        let mut state = self.state.lock().unwrap();

        match state.streams.get(&id) {
            Some(stream) => stream.clone(),
            None => {
                let runner = Runner::new(state.device.clone());
                let stream = Stream::new(runner);
                state.streams.insert(id, stream.clone());
                stream
            }
        }
    }

    pub fn pop_stream(&self, id: u64) -> Stream<B> {
        let mut state = self.state.lock().unwrap();
        state.streams.remove(&id).unwrap()
    }
}
