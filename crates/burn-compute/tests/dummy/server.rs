use std::sync::Arc;

use burn_common::reader::Reader;
use burn_compute::{
    memory_management::{MemoryManagement, MemoryTensorBufferHandle, SimpleMemoryManagement},
    server::{ComputeServer, ExecutionBufferHandle, TensorBufferHandle},
    storage::BytesStorage,
};
use derive_new::new;

use super::DummyKernel;

/// The dummy server is used to test the burn-compute infrastructure.
/// It uses simple memory management with a bytes storage on CPU, without asynchronous tasks.
#[derive(new, Debug)]
pub struct DummyServer<MM = SimpleMemoryManagement<BytesStorage>> {
    memory_management: MM,
}

impl<MM> ComputeServer for DummyServer<MM>
where
    MM: MemoryManagement<BytesStorage>,
{
    type Kernel = Arc<dyn DummyKernel>;
    type Storage = BytesStorage;
    type MemoryManagement = MM;
    type AutotuneKey = String;

    fn read(&mut self, handle: ExecutionBufferHandle<Self>) -> Reader<Vec<u8>> {
        let bytes = self.memory_management.get(handle.memory);

        Reader::Concrete(bytes.read().to_vec())
    }

    fn create(&mut self, data: &[u8]) -> TensorBufferHandle<Self> {
        let handle = self.memory_management.reserve(data.len());
        let resource = self.memory_management.get(handle.enqueue());

        let bytes = resource.write();

        for (i, val) in data.iter().enumerate() {
            bytes[i] = *val;
        }

        TensorBufferHandle::new(handle)
    }

    fn empty(&mut self, size: usize) -> TensorBufferHandle<Self> {
        TensorBufferHandle::new(self.memory_management.reserve(size))
    }

    fn execute(&mut self, kernel: Self::Kernel, handles: Vec<ExecutionBufferHandle<Self>>) {
        let mut resources = handles
            .into_iter()
            .map(|handle| self.memory_management.get(handle.memory))
            .collect::<Vec<_>>();

        kernel.compute(&mut resources);
    }

    fn sync(&mut self) {
        // Nothing to do with dummy backend.
    }
}
