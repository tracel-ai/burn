use crate::{
    Client, FusionBackend, FusionRuntime,
    stream::{Operation, StreamId},
};
use burn_backend::{DType, ExecutionError, Shape, TensorData, TensorMetadata};
use burn_ir::{OperationIr, TensorId, TensorIr, TensorStatus};
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

/// Tensor primitive for the [fusion backend](crate::FusionBackend) for all kind.
pub struct FusionTensor<R: FusionRuntime> {
    /// Tensor id.
    pub id: TensorId,
    /// The shape of the tensor.
    pub shape: Shape,
    /// The fusion client.
    pub client: Client<R>,
    /// The datatype of the tensor.
    pub dtype: DType,
    /// The current stream id this tensor is on.
    pub stream: StreamId,
    pub(crate) count: Arc<AtomicU32>,
}

impl<R: FusionRuntime> Clone for FusionTensor<R> {
    fn clone(&self) -> Self {
        let current = StreamId::current();
        if self.stream != current {
            return self.shared_view(current);
        }

        self.count.fetch_add(1, Ordering::Acquire);

        Self {
            id: self.id,
            shape: self.shape.clone(),
            client: self.client.clone(),
            dtype: self.dtype,
            stream: self.stream,
            count: self.count.clone(),
        }
    }
}

impl<R: FusionRuntime> core::fmt::Debug for FusionTensor<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            format!(
                "{{ id: {:?}, shape: {:?}, device: {:?} }}",
                self.id,
                self.shape,
                self.client.device(),
            )
            .as_str(),
        )
    }
}

impl<R: FusionRuntime> TensorMetadata for FusionTensor<R> {
    type Device = R::FusionDevice;
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
        self.client.device().clone()
    }

    fn can_mut(&self) -> bool {
        // Same rule as `status` at drain time: a handle shared on its stream
        // (count > 1) is read-only, a unique one is read-write and the fused
        // kernel may write its buffer in place.
        matches!(
            self.status(self.count.load(Ordering::Acquire)),
            TensorStatus::ReadWrite
        )
    }
}

impl<R: FusionRuntime> FusionTensor<R> {
    pub(crate) fn new(
        id: TensorId,
        shape: Shape,
        dtype: DType,
        client: Client<R>,
        stream: StreamId,
    ) -> Self {
        Self {
            id,
            shape,
            client,
            dtype,
            stream,
            count: Arc::new(AtomicU32::new(1)),
        }
    }

    fn status(&self, count: u32) -> TensorStatus {
        if count <= 1 {
            TensorStatus::ReadWrite
        } else {
            TensorStatus::ReadOnly
        }
    }

    /// Intermediate representation to be used when using an uninitialized tensor as output.
    pub fn to_ir_out(&self) -> TensorIr {
        TensorIr {
            status: TensorStatus::NotInit,
            shape: self.shape.clone(),
            id: self.id,
            dtype: self.dtype,
        }
    }

    /// Intermediate representation to be used when using an initialized tensor used as input.
    pub fn into_ir(mut self) -> TensorIr {
        let current = StreamId::current();
        if self.stream != current {
            self = self.shared_view(current);
        }

        let count = self.count.load(Ordering::Acquire);
        let status = self.status(count);

        let mut shape_out = Shape::from(Vec::<usize>::new());
        core::mem::swap(&mut self.shape, &mut shape_out);

        if let TensorStatus::ReadWrite = status {
            // Avoids an unwanted drop on the same thread.
            //
            // Since `drop` is called after `into_ir`, we must not register a drop if the tensor
            // was consumed with a `ReadWrite` status.
            self.count.fetch_add(1, Ordering::Acquire);
        }

        TensorIr {
            status,
            shape: shape_out,
            id: self.id,
            dtype: self.dtype,
        }
    }

    /// Create a fresh `FusionTensor` on `current` that aliases the same backing
    /// handle as `self`. Used by [`Clone`] and [`Self::into_ir`] when the tensor is
    /// crossing stream boundaries — the rest of the pipeline only ever sees ids
    /// whose home stream is the calling stream.
    ///
    /// The cross-stream coordination (draining the source stream so the handle
    /// exists, then aliasing it under a fresh id) is done by
    /// [`MultiStream::tag_shared_view`](crate::stream::MultiStream::tag_shared_view).
    /// See that type's docs for the full strategy — how shares are tagged, how the
    /// buffer's lifetime is managed across the two sides, and why a single drain
    /// per source is enough.
    fn shared_view(&self, current: StreamId) -> Self {
        let new_id = self.client.create_empty_handle();

        self.client.tag_shared_view(self.stream, self.id, new_id);

        Self::new(
            new_id,
            self.shape.clone(),
            self.dtype,
            self.client.clone(),
            current,
        )
    }

    pub(crate) async fn into_data<B>(self) -> Result<TensorData, ExecutionError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let id = self.stream;
        let client = self.client.clone();
        let desc = self.into_ir();
        client.read_tensor_float::<B>(desc, id).await
    }

    pub(crate) async fn q_into_data<B>(self) -> Result<TensorData, ExecutionError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        if let DType::QFloat(_scheme) = self.dtype {
            let id = self.stream;
            let client = self.client.clone();
            let desc = self.into_ir();
            client.read_tensor_quantized::<B>(desc, id).await
        } else {
            panic!("Expected quantized float dtype, got {:?}", self.dtype)
        }
    }

    pub(crate) async fn int_into_data<B>(self) -> Result<TensorData, ExecutionError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let id = self.stream;
        let client = self.client.clone();
        let desc = self.into_ir();
        client.read_tensor_int::<B>(desc, id).await
    }

    pub(crate) async fn bool_into_data<B>(self) -> Result<TensorData, ExecutionError>
    where
        B: FusionBackend<FusionRuntime = R>,
    {
        let id = self.stream;
        let client = self.client.clone();
        let desc = self.into_ir();
        client.read_tensor_bool::<B>(desc, id).await
    }
}

#[derive(new, Debug)]
pub(crate) struct DropOp {
    pub(crate) id: TensorId,
}

impl<RO: FusionRuntime> Operation<RO> for DropOp {
    fn execute(
        &self,
        handles: &mut burn_ir::HandleContainer<RO::FusionHandle>,
    ) -> Result<(), burn_backend::ExecutionError> {
        handles.remove_handle(self.id);
        Ok(())
    }
}

/// Drops that could not be registered when they happened.
///
/// Registering a drop re-enters the client, which can drain the stream and run
/// queued work. Doing that while the thread is unwinding is how this used to
/// abort, so a drop raised during a panic is set aside here instead and
/// replayed by the next registration on this thread — which is a normal call
/// stack, with nothing unwinding through it.
///
/// The alternative, and what this replaces, was to drop the registration
/// entirely: no re-entry, but the tensor's entry in the handle container was
/// never released, and a claim on it outlived every tensor that could report
/// it.
pub(crate) mod deferred {
    use core::cell::RefCell;

    thread_local! {
        static PENDING: RefCell<Vec<Box<dyn FnOnce()>>> = const { RefCell::new(Vec::new()) };
    }

    /// Set a drop aside until this thread is somewhere it can be registered.
    pub(crate) fn defer(drop: impl FnOnce() + 'static) {
        PENDING.with(|pending| pending.borrow_mut().push(Box::new(drop)));
    }

    /// Register everything set aside. Called where a registration already
    /// happens, so the check rides along with work the caller was doing anyway.
    pub(crate) fn flush() {
        // The common case, on the hot path: nothing was ever deferred.
        if PENDING.with(|pending| pending.borrow().is_empty()) {
            return;
        }

        // Taken rather than iterated in place: registering a drop can defer
        // another one, and holding the borrow across that would panic.
        loop {
            let batch: Vec<_> = PENDING.with(|pending| core::mem::take(&mut *pending.borrow_mut()));

            if batch.is_empty() {
                return;
            }

            for drop in batch {
                drop();
            }
        }
    }
}

impl<R: FusionRuntime> Drop for FusionTensor<R> {
    fn drop(&mut self) {
        let count = self.count.fetch_sub(1, Ordering::Acquire);

        // A drop raised while the thread is unwinding is set aside rather than
        // registered: registering re-enters the client, which can drain the
        // stream and run queued work, and doing that mid-unwind is how this
        // used to abort. It is replayed by the next registration on this
        // thread — which is a normal call stack — so the entry is released
        // rather than leaked.
        if std::thread::panicking() {
            if let TensorStatus::ReadWrite = self.status(count) {
                let mut shape = Shape::from(Vec::<usize>::new());
                core::mem::swap(&mut shape, &mut self.shape);

                let ir = TensorIr {
                    id: self.id,
                    shape,
                    status: TensorStatus::ReadWrite,
                    dtype: self.dtype,
                };
                let (client, stream, id) = (self.client.clone(), self.stream, self.id);

                deferred::defer(move || {
                    client.register_foreign_drop(stream, ir, DropOp { id });
                });
            }
            return;
        }

        match self.status(count) {
            TensorStatus::ReadWrite => {
                let mut shape = Shape::from(Vec::<usize>::new());
                core::mem::swap(&mut shape, &mut self.shape);

                let ir = TensorIr {
                    id: self.id,
                    shape,
                    status: TensorStatus::ReadWrite,
                    dtype: self.dtype,
                };

                // A foreign drop interleaves at a nondeterministic point in the home stream's
                // pending fused segment; route it through a path that never touches the queue.
                if StreamId::current() == self.stream {
                    self.client.register(
                        self.stream,
                        OperationIr::Drop(ir),
                        DropOp { id: self.id },
                    );
                } else {
                    self.client
                        .register_foreign_drop(self.stream, ir, DropOp { id: self.id });
                }
            }
            TensorStatus::ReadOnly => {}
            TensorStatus::NotInit => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::deferred;
    use std::cell::Cell;
    use std::rc::Rc;

    /// A drop set aside during an unwind runs at the next flush, rather than
    /// being dropped on the floor as it used to be.
    #[test]
    fn a_deferred_drop_runs_at_the_next_flush() {
        let ran = Rc::new(Cell::new(0));

        let counter = ran.clone();
        deferred::defer(move || counter.set(counter.get() + 1));
        assert_eq!(ran.get(), 0, "not until something flushes");

        deferred::flush();
        assert_eq!(ran.get(), 1);

        deferred::flush();
        assert_eq!(ran.get(), 1, "and only once");
    }

    /// Registering a deferred drop can defer another — a tensor released by the
    /// first going out of scope. The flush has to reach those too, which is why
    /// it takes the queue rather than iterating it in place.
    #[test]
    fn a_flush_reaches_drops_deferred_by_the_flush() {
        let ran = Rc::new(Cell::new(0));

        let counter = ran.clone();
        deferred::defer(move || {
            counter.set(counter.get() + 1);
            let inner = counter.clone();
            deferred::defer(move || inner.set(inner.get() + 1));
        });

        deferred::flush();
        assert_eq!(ran.get(), 2, "both the deferred drop and the one it caused");
    }
}
