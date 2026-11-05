use core::sync::atomic::{AtomicU32, Ordering};

use alloc::format;
use alloc::{sync::Arc, vec::Vec};

use super::RouterClient;
use burn_backend::{DType, Shape, TensorData, TensorMetadata, backend::ExecutionError};
use burn_ir::{TensorId, TensorIr, TensorStatus};

/// Tensor primitive for the [router backend](crate::BackendRouter).
pub struct RouterTensor<C: RouterClient> {
    pub(crate) id: TensorId,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
    /// The client that has this tensor
    pub client: C,
    pub(crate) count: Arc<AtomicU32>,
}

impl<C: RouterClient> TensorMetadata for RouterTensor<C> {
    type Device = C::Device;
    fn dtype(&self) -> DType {
        self.dtype
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn rank(&self) -> usize {
        self.shape.num_dims()
    }

    fn device(&self) -> Self::Device {
        self.client.device()
    }

    fn can_mut(&self) -> bool {
        // Same sharing rule as `into_ir`: a handle cloned on its stream
        // (count > 1) is read-only for the runner, a unique one is read-write.
        self.count.load(Ordering::Acquire) <= 1
    }
}

impl<C: RouterClient> RouterTensor<C> {
    /// The id identifying this tensor on its [client](RouterClient).
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Create a new router tensor.
    pub fn new(id: TensorId, shape: Shape, dtype: DType, client: C) -> Self {
        Self {
            id,
            shape,
            dtype,
            client,
            count: Arc::new(AtomicU32::new(1)),
        }
    }

    pub(crate) async fn into_data(self) -> Result<TensorData, ExecutionError> {
        // `into_ir` gives up this handle before the read is registered. Held, it keeps another
        // thread that drops the last other handle in between from registering the drop first.
        let held = self.clone();
        let read = self.client.clone().read_tensor_async(self.into_ir());
        drop(held);
        read.await
    }

    /// Get the ir for this tensor
    pub fn into_ir(mut self) -> TensorIr {
        let count = self.count.load(Ordering::Relaxed);
        let status = self.status(count);
        let mut shape_out = Shape::from(Vec::<usize>::new());
        core::mem::swap(&mut self.shape, &mut shape_out);

        if let TensorStatus::ReadWrite = status {
            // Avoids an unwanted drop on the same thread.
            //
            // Since `drop` is called after `into_ir`, we must not register a drop if the tensor
            // was consumed with a `ReadWrite` status.
            self.count.fetch_add(1, Ordering::Relaxed);
        }

        TensorIr {
            status,
            shape: shape_out,
            id: self.id,
            dtype: self.dtype,
        }
    }

    pub(crate) fn to_ir_out(&self) -> TensorIr {
        TensorIr {
            status: TensorStatus::NotInit,
            shape: self.shape.clone(),
            id: self.id,
            dtype: self.dtype,
        }
    }

    pub(crate) fn status(&self, count: u32) -> TensorStatus {
        if count <= 1 {
            TensorStatus::ReadWrite
        } else {
            TensorStatus::ReadOnly
        }
    }
}

impl<C: RouterClient> core::fmt::Debug for RouterTensor<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            format!(
                "{{ id: {:?}, shape: {:?}, dtype: {:?}, device: {:?} }}",
                self.id,
                self.shape,
                self.dtype,
                self.client.device(),
            )
            .as_str(),
        )
    }
}

impl<C: RouterClient> Clone for RouterTensor<C> {
    fn clone(&self) -> Self {
        self.count.fetch_add(1, Ordering::Relaxed);

        Self {
            id: self.id,
            shape: self.shape.clone(),
            client: self.client.clone(),
            dtype: self.dtype,
            count: self.count.clone(),
        }
    }
}

impl<C: RouterClient> Drop for RouterTensor<C> {
    fn drop(&mut self) {
        let count = self.count.fetch_sub(1, Ordering::Relaxed);

        match self.status(count) {
            TensorStatus::ReadWrite => {
                let id = self.id;
                let mut shape = Shape::from(Vec::<usize>::new());
                core::mem::swap(&mut shape, &mut self.shape);

                let ir = TensorIr {
                    id,
                    shape,
                    status: TensorStatus::ReadWrite,
                    dtype: self.dtype,
                };
                self.client.register_op(burn_ir::OperationIr::Drop(ir));
            }
            TensorStatus::ReadOnly => {}
            TensorStatus::NotInit => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{DTypeUsageSet, RandomKey};
    use burn_flex::FlexDevice;
    use burn_ir::{GraphBindings, GraphId, OperationIr};
    use burn_std::future::DynFut;
    use std::sync::Mutex;

    #[derive(Debug, PartialEq)]
    enum Event {
        Read(TensorId),
        Drop(TensorId),
    }

    /// Records the reads and drops it is sent. It drops the handle it holds as a read arrives,
    /// before taking the read, as another thread giving up its handle just then would.
    #[derive(Clone, Default)]
    struct DropOnRead {
        events: Arc<Mutex<Vec<Event>>>,
        other: Arc<Mutex<Option<RouterTensor<DropOnRead>>>>,
    }

    impl RouterClient for DropOnRead {
        type Device = FlexDevice;

        fn register_op(&self, op: OperationIr) {
            if let OperationIr::Drop(tensor) = op {
                self.events.lock().unwrap().push(Event::Drop(tensor.id));
            }
        }

        fn read_tensor_async(
            &self,
            tensor: TensorIr,
        ) -> DynFut<Result<TensorData, ExecutionError>> {
            drop(self.other.lock().unwrap().take());
            self.events.lock().unwrap().push(Event::Read(tensor.id));
            Box::pin(async { Ok(TensorData::from([0f32])) })
        }

        fn sync(&self) -> Result<(), ExecutionError> {
            Ok(())
        }

        fn flush(&self) -> Result<(), ExecutionError> {
            Ok(())
        }

        fn create_empty_handle(&self) -> TensorId {
            TensorId::new(0)
        }

        fn register_tensor_data(&self, _data: TensorData) -> RouterTensor<Self> {
            unimplemented!()
        }

        fn device(&self) -> Self::Device {
            FlexDevice
        }

        fn seed(&self, _seed: u64) {}

        fn next_random_key(&self) -> RandomKey {
            unimplemented!()
        }

        fn dtype_usage(&self, _dtype: DType) -> DTypeUsageSet {
            unimplemented!()
        }

        fn execute_graph(&self, _graph_id: GraphId, _bindings: GraphBindings) {}

        fn register_alias(&self, _new_id: TensorId, _src_id: TensorId) {}
    }

    #[test]
    fn a_read_is_registered_before_the_drop_of_a_handle_given_up_meanwhile() {
        let client = DropOnRead::default();
        let id = TensorId::new(7);
        let tensor = RouterTensor::new(id, Shape::from([1]), DType::F32, client.clone());
        *client.other.lock().unwrap() = Some(tensor.clone());

        burn_std::future::block_on(tensor.into_data()).unwrap();

        assert_eq!(
            *client.events.lock().unwrap(),
            [Event::Read(id), Event::Drop(id)]
        );
    }
}
