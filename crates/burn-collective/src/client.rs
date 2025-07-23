use std::sync::mpsc::SyncSender;

use burn_tensor::backend::Backend;

use crate::{
    CollectiveConfig, CollectiveError, DeviceId, ReduceOperation,
    local_server::{
        AllReduceResult, BroadcastResult, FinishResult, Message, ReduceResult, RegisterResult,
    },
};

#[derive(Clone)]
pub(crate) struct LocalCollectiveClient<B: Backend> {
    pub channel: SyncSender<Message<B>>,
}

impl<B: Backend> LocalCollectiveClient<B> {
    pub(crate) fn reset(&self) {
        self.channel.send(Message::Reset).unwrap();
    }

    pub(crate) fn register(&mut self, id: DeviceId, config: CollectiveConfig) -> RegisterResult {
        if config.is_valid() {
            return Err(CollectiveError::InvalidConfig);
        }

        let (callback, rec) = std::sync::mpsc::sync_channel::<RegisterResult>(1);

        self.channel
            .send(Message::Register {
                device_id: id,
                config,
                callback,
            })
            .unwrap();

        rec.recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))
    }

    pub(crate) fn all_reduce(
        &self,
        id: DeviceId,
        tensor: B::FloatTensorPrimitive,
        op: ReduceOperation,
    ) -> AllReduceResult<B::FloatTensorPrimitive> {
        let (callback, rec) =
            std::sync::mpsc::sync_channel::<AllReduceResult<B::FloatTensorPrimitive>>(1);
        let msg = Message::AllReduce {
            device_id: id,
            tensor,
            op,
            callback,
        };

        self.channel.send(msg).unwrap();

        // returns a tensor primitive that may or may not be on the correct device,
        // depending on the strategy used.
        rec.recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))
    }

    pub(crate) fn reduce(
        &self,
        id: DeviceId,
        tensor: B::FloatTensorPrimitive,
        op: ReduceOperation,
        root: DeviceId,
    ) -> ReduceResult<B::FloatTensorPrimitive> {
        let (callback, rec) =
            std::sync::mpsc::sync_channel::<ReduceResult<B::FloatTensorPrimitive>>(1);
        let msg = Message::Reduce {
            device_id: id,
            tensor,
            op,
            root,
            callback,
        };

        self.channel.send(msg).unwrap();

        // returns a tensor or none depending on if this device is the root
        rec.recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))
    }

    pub(crate) fn broadcast(
        &self,
        id: DeviceId,
        tensor: Option<B::FloatTensorPrimitive>,
        root: DeviceId,
    ) -> BroadcastResult<B::FloatTensorPrimitive> {
        let (callback, rec) =
            std::sync::mpsc::sync_channel::<BroadcastResult<B::FloatTensorPrimitive>>(1);
        let msg = Message::Broadcast {
            device_id: id,
            tensor,
            root,
            callback,
        };

        self.channel.send(msg).unwrap();

        // returns a tensor or none depending on if this device is the root
        rec.recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))
    }

    pub(crate) fn finish(&self, id: DeviceId) -> FinishResult {
        let (callback, rec) = std::sync::mpsc::sync_channel::<FinishResult>(1);
        self.channel.send(Message::Finish { id, callback }).unwrap();

        rec.recv()
            .unwrap_or(Err(CollectiveError::LocalServerMissing))
    }
}
