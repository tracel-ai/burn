use crate::{
    CollectiveConfig, CollectiveError, PeerId, ReduceOperation,
    local::server::{
        AllReduceResult, BroadcastResult, FinishResult, Message, ReduceResult, RegisterResult,
    },
};
use burn_tensor::{Tensor, backend::Backend};
use std::sync::mpsc::SyncSender;

/// Local client to comminicate with the local server. Each thread has a client.
#[derive(Clone)]
pub(crate) struct LocalCollectiveClient<B: Backend> {
    pub channel: SyncSender<Message<B>>,
}

impl<B: Backend> LocalCollectiveClient<B> {
    pub(crate) fn reset(&self) {
        self.channel.send(Message::Reset).unwrap();
    }

    pub(crate) fn register(
        &mut self,
        id: PeerId,
        device: B::Device,
        config: CollectiveConfig,
    ) -> RegisterResult {
        if !config.is_valid() {
            return Err(CollectiveError::InvalidConfig);
        }

        let (callback, rec) = std::sync::mpsc::sync_channel::<RegisterResult>(1);

        self.channel
            .send(Message::Register {
                device_id: id,
                device,
                config,
                callback,
            })
            .unwrap();

        rec.recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))
    }

    /// Calls for an all-reduce operation with the given parameters, and returns the result.
    /// The `params` must be the same as the parameters passed by the other nodes.
    ///
    /// * `id` - The peer id of the caller
    /// * `tensor` - The input tensor to reduce with the peers' tensors
    /// * `config` - Config of the collective operation, must be coherent with the other calls
    pub fn all_reduce<const D: usize>(
        &self,
        id: PeerId,
        tensor: Tensor<B, D>,
        op: ReduceOperation,
    ) -> AllReduceResult<Tensor<B, D>> {
        let (callback, rec) =
            std::sync::mpsc::sync_channel::<AllReduceResult<B::FloatTensorPrimitive>>(1);

        let msg = Message::AllReduce {
            device_id: id,
            tensor: tensor.into_primitive().tensor(),
            op,
            callback,
        };

        self.channel.send(msg).unwrap();

        let primitive = rec
            .recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))?;

        // The resulting tensor should be on the correct device
        let tensor = Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive));

        Ok(tensor)
    }

    /// Reduces a tensor onto one device.
    ///
    /// * `id` - The peer id of the caller
    /// * `tensor` - The tensor to send as input
    /// * `op` - The operation to do for reduce
    /// * `root` - The ID of the peer that will receive the result.
    ///
    /// Returns Ok(None) if the root tensor is not the caller. Otherwise, returns the reduced tensor.
    pub fn reduce<const D: usize>(
        &self,
        id: PeerId,
        tensor: Tensor<B, D>,
        op: ReduceOperation,
        root: PeerId,
    ) -> ReduceResult<Tensor<B, D>> {
        let (callback, rec) =
            std::sync::mpsc::sync_channel::<ReduceResult<B::FloatTensorPrimitive>>(1);
        let msg = Message::Reduce {
            device_id: id,
            tensor: tensor.into_primitive().tensor(),
            op,
            root,
            callback,
        };

        self.channel.send(msg).unwrap();

        // returns a tensor or none depending on if this device is the root
        let primitive = rec
            .recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))?;

        let tensor = primitive.map(|primitive| {
            Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive))
        });

        Ok(tensor)
    }

    /// Broadcasts, or receives a broadcasted tensor.
    ///
    /// * `id` - The peer id of the caller
    /// * `tensor` - If defined, this tensor will be broadcasted. Otherwise, this call will receive
    ///   the broadcasted tensor.
    ///
    /// Returns the broadcasted tensor.
    pub fn broadcast<const D: usize>(
        &self,
        id: PeerId,
        tensor: Option<Tensor<B, D>>,
    ) -> BroadcastResult<Tensor<B, D>> {
        let tensor = tensor.map(|tensor| tensor.into_primitive().tensor());

        let (callback, rec) =
            std::sync::mpsc::sync_channel::<BroadcastResult<B::FloatTensorPrimitive>>(1);
        let msg = Message::Broadcast {
            device_id: id,
            tensor,
            callback,
        };

        self.channel.send(msg).unwrap();

        // returns a tensor or none depending on if this device is the root
        let primitive = rec
            .recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))?;

        let tensor = Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive));

        Ok(tensor)
    }

    pub(crate) fn finish(&self, id: PeerId) -> FinishResult {
        let (callback, rec) = std::sync::mpsc::sync_channel::<FinishResult>(1);
        self.channel.send(Message::Finish { id, callback }).unwrap();

        rec.recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))
    }
}
