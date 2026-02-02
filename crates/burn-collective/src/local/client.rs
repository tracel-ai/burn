use crate::local::all_reduce::AllReduceResult;
use crate::{
    CollectiveConfig, CollectiveError, PeerId, ReduceOperation,
    local::{
        BroadcastResult, ReduceResult,
        server::{FinishResult, Message, RegisterResult},
    },
};
use burn_tensor::backend::Backend;
use std::sync::mpsc::{Receiver, SyncSender};

/// Local client to communicate with the local server. Each thread has a client.
#[derive(Clone)]
pub(crate) struct LocalCollectiveClient<B: Backend> {
    pub channel: SyncSender<Message<B>>,
}

/// A pending operation that can be waited on.
pub(crate) struct PendingCollectiveOperation<T> {
    rx: Receiver<Result<T, CollectiveError>>,
}

impl<T> From<PendingCollectiveOperation<T>> for Receiver<Result<T, CollectiveError>> {
    fn from(value: PendingCollectiveOperation<T>) -> Self {
        value.rx
    }
}

impl<T> PendingCollectiveOperation<T> {
    /// Wait on the operation.
    ///
    /// Given a `Receiver<Result<T, CollectiveError>>`, this function will wait:
    /// - Unwraps `Ok(Result<T, CollectiveError>)` into `Result<T, CollectiveError>`;
    /// - maps `Err(RecvError)` to `Err(CollectiveError::LocalServerMissing)`.
    pub(crate) fn wait(self) -> Result<T, CollectiveError> {
        let tensor = self
            .rx
            .recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))?;

        Ok(tensor)
    }
}

impl<B: Backend> LocalCollectiveClient<B> {
    /// Common logic for starting a collective operation.
    ///
    /// - Allocates `(callback, recv)` channels,
    /// - Passes the `callback` to the `Message<B>` builder,
    /// - Sends the message through the collective channel,
    /// - Returns the `recv`.
    pub(crate) fn start_operation<T, F>(&self, builder: F) -> PendingCollectiveOperation<T>
    where
        F: FnOnce(SyncSender<Result<T, CollectiveError>>) -> Message<B>,
    {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.channel.send((builder)(tx)).unwrap();
        PendingCollectiveOperation { rx }
    }

    /// Common logic for starting a collective operation, with validation.
    ///
    /// When `valid` is `Err`, this function returns a `Receiver<Result<T, CollectiveError>>` that
    /// immediately returns `Err(valid)`;
    /// otherwise, it behaves like [`LocalCollectiveClient::start_operation`].
    pub(crate) fn start_valid_operation<T, F>(
        &self,
        valid: Result<(), CollectiveError>,
        builder: F,
    ) -> PendingCollectiveOperation<T>
    where
        F: FnOnce(SyncSender<Result<T, CollectiveError>>) -> Message<B>,
    {
        match valid {
            Err(e) => {
                let (tx, rx) = std::sync::mpsc::sync_channel(1);
                tx.send(Err(e)).unwrap();
                PendingCollectiveOperation { rx }
            }
            _ => self.start_operation(builder),
        }
    }

    pub(crate) fn reset(&self) {
        self.channel.send(Message::Reset).unwrap();
    }

    pub(crate) fn register(
        &mut self,
        id: PeerId,
        device: B::Device,
        config: CollectiveConfig,
    ) -> RegisterResult {
        self.register_start(id, device, config).wait()
    }

    pub(crate) fn register_start(
        &mut self,
        id: PeerId,
        device: B::Device,
        config: CollectiveConfig,
    ) -> PendingCollectiveOperation<()> {
        self.start_valid_operation(
            match config.is_valid() {
                true => Ok(()),
                false => Err(CollectiveError::InvalidConfig),
            },
            |callback| Message::Register {
                device_id: id,
                device,
                config,
                callback,
            },
        )
    }

    /// Calls for an all-reduce operation with the given parameters and returns the result.
    /// The `params` must be the same as the parameters passed by the other nodes.
    ///
    /// # Arguments
    /// * `id` - The peer id of the caller
    /// * `tensor` - The input tensor to reduce with the peers' tensors
    /// * `config` - Config of the collective operation. Must be coherent with the other calls.
    ///
    /// # Result
    /// - `Ok(tensor)` if the operation was successful
    /// - `Err(CollectiveError)` on error.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, tensor))
    )]
    pub fn all_reduce(
        &self,
        id: PeerId,
        tensor: B::FloatTensorPrimitive,
        op: ReduceOperation,
    ) -> AllReduceResult<B::FloatTensorPrimitive> {
        self.all_reduce_start(id, tensor, op).wait()
    }

    /// Starts an all-reduce operation with the given parameters.
    ///
    /// The `params` must be the same as the parameters passed by the other nodes.
    ///
    /// This receiver can be waited on using [`LocalCollectiveClient::operation_wait`].
    ///
    /// # Arguments
    /// * `id` - The peer id of the caller
    /// * `tensor` - The input tensor to reduce with the peers' tensors
    /// * `config` - Config of the collective operation. Must be coherent with the other calls.
    ///
    /// # Result
    ///
    /// A `Receiver<>` that will yield:
    /// - `Ok(AllReduceResult<B::FloatTensorPrimitive>)` if the operation was successful
    /// - `Err(SendError)` if the channel was dropped.
    pub(crate) fn all_reduce_start(
        &self,
        id: PeerId,
        tensor: B::FloatTensorPrimitive,
        op: ReduceOperation,
    ) -> PendingCollectiveOperation<B::FloatTensorPrimitive> {
        self.start_operation(|callback| Message::AllReduce {
            device_id: id,
            tensor,
            op,
            callback,
        })
    }

    /// Reduces a tensor onto one device.
    ///
    /// # Arguments
    /// - `id` - The peer id of the caller.
    /// - `tensor` - The tensor to send as input.
    /// - `op` - The reduce operation to apply.
    /// - `root` - The ID of the peer that will receive the result.
    ///
    /// Returns Ok(None) if the root tensor is not the caller. Otherwise, returns the reduced tensor.
    pub fn reduce(
        &self,
        id: PeerId,
        tensor: B::FloatTensorPrimitive,
        op: ReduceOperation,
        root: PeerId,
    ) -> ReduceResult<B::FloatTensorPrimitive> {
        self.reduce_start(id, tensor, op, root).wait()
    }

    /// Starts a reduce operation on a tensor onto one device.
    ///
    /// This receiver can be waited on using [`LocalCollectiveClient::operation_wait`].
    ///
    /// # Arguments
    /// - `id` - The peer id of the caller.
    /// - `tensor` - The tensor to send as input.
    /// - `op` - The reduce operation to apply.
    /// - `root` - The ID of the peer that will receive the result.
    ///
    /// # Result
    ///
    /// A `Receiver<>` that will yield:
    /// - `Ok(ReduceResult<B::FloatTensorPrimitive>)` if the operation was successful
    /// - `Err(SendError)` if the channel was dropped.
    pub(crate) fn reduce_start(
        &self,
        id: PeerId,
        tensor: B::FloatTensorPrimitive,
        op: ReduceOperation,
        root: PeerId,
    ) -> PendingCollectiveOperation<Option<B::FloatTensorPrimitive>> {
        self.start_operation(|callback| Message::Reduce {
            device_id: id,
            tensor,
            op,
            root,
            callback,
        })
    }

    /// Broadcasts, or receives a broadcasted tensor.
    ///
    /// # Arguments
    /// - `id` - The peer id of the caller
    /// - `tensor` - If defined, this tensor will be broadcasted.
    ///   Otherwise, this call will receive the broadcasted tensor.
    ///
    /// # Result
    /// Synchronously waits on the broadcasted tensor.
    pub fn broadcast(
        &self,
        id: PeerId,
        tensor: Option<B::FloatTensorPrimitive>,
    ) -> BroadcastResult<B::FloatTensorPrimitive> {
        self.broadcast_start(id, tensor).wait()
    }

    /// Starts a Broadcast, or receives a broadcasted tensor.
    ///
    /// This receiver can be waited on using [`LocalCollectiveClient::operation_wait`].
    ///
    /// # Arguments
    /// - `id` - The peer id of the caller
    /// - `tensor` - If defined, this tensor will be broadcasted. Otherwise, this call will receive
    ///   the broadcasted tensor.
    ///
    /// # Result
    ///
    /// A `Receiver<>` that will yield:
    /// - `Ok(BroadcastResult<B::FloatTensorPrimitive>)` if the operation was successful
    /// - `Err(SendError)` if the channel was dropped.
    pub(crate) fn broadcast_start(
        &self,
        id: PeerId,
        tensor: Option<B::FloatTensorPrimitive>,
    ) -> PendingCollectiveOperation<B::FloatTensorPrimitive> {
        self.start_operation(|callback| Message::Broadcast {
            device_id: id,
            tensor,
            callback,
        })
    }

    pub(crate) fn finish(&self, id: PeerId) -> FinishResult {
        self.finish_start(id).wait()
    }

    pub(crate) fn finish_start(&self, id: PeerId) -> PendingCollectiveOperation<()> {
        self.start_operation(|callback| Message::Finish { id, callback })
    }
}
