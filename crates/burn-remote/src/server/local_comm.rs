//! In-process communication between two sessions hosted by the same server.
//!
//! Whenever sessions on the **same** server (same address, different device index) need to
//! hand device-resident data to each other, there is no reason to round-trip through the host:
//! both sessions' [`TensorInterpreter`](burn_router::TensorInterpreter)s live in the same
//! process, so the source can hand its primitive straight to the target, which moves it onto
//! its own device with the inner backend's `to_device`. This is the same-host counterpart of the
//! cross-server tensor transfer, which moves data *between* servers over the network.
//!
//! It backs cross-device `to_device` and the all-reduce over devices of one server, whose
//! members each hand over their primitive and wait for the last to run it over them all.
//!
//! This registry is the rendezvous point. The source session exposes a primitive under a
//! [`LocalTransferId`]; the target session takes it. Either may arrive first — the source op
//! and the target op travel on separate connections with no cross-connection ordering — so the
//! taker waits on a [`Notify`] until the primitive shows up.

use crate::shared::{LocalAllReducePart, LocalCollectiveId, LocalTransferId, SessionId};
use burn_backend::{DeviceOps, TensorMetadata, tensor::FloatTensor};
use burn_ir::{BackendIr, HandleKind};
use burn_std::id::StreamId;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, Notify, oneshot};

/// A primitive parked in the registry, tagged with the session that exposed it so the entry can
/// be reclaimed if that session goes away before its target takes it.
struct Pending<B: BackendIr> {
    source: SessionId,
    tensor: HandleKind<B>,
}

/// A member that has handed its tensor to an all-reduce, waiting for its result.
struct ArrivedMember<B: BackendIr> {
    tensor: FloatTensor<B>,
    result: oneshot::Sender<Result<FloatTensor<B>, String>>,
}

/// Rendezvous registry for same-host communication, keyed by [`LocalTransferId`].
pub(crate) struct LocalCommService<B: BackendIr> {
    /// Primitives exposed by source sessions, waiting to be taken by their target.
    pending: Mutex<HashMap<LocalTransferId, Pending<B>>>,
    /// The members of each all-reduce that have arrived, by member, until the last does.
    all_reduces: Mutex<HashMap<LocalCollectiveId, Vec<Option<ArrivedMember<B>>>>>,
    /// Wakes takers whenever a new primitive is exposed.
    notify: Arc<Notify>,
}

impl<B: BackendIr> LocalCommService<B> {
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            all_reduces: Mutex::new(HashMap::new()),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Expose `tensor` under `transfer_id`, waking any session already waiting for it. `source` is
    /// the session exposing it, so a stranded entry can be purged if that session closes before its
    /// target takes the tensor (see [`purge_session`](Self::purge_session)).
    pub async fn expose(
        &self,
        source: SessionId,
        transfer_id: LocalTransferId,
        tensor: HandleKind<B>,
    ) {
        self.pending
            .lock()
            .await
            .insert(transfer_id, Pending { source, tensor });
        self.notify.notify_waiters();
    }

    /// Take the primitive exposed under `transfer_id`, waiting until it is exposed.
    pub async fn take(&self, transfer_id: LocalTransferId) -> HandleKind<B> {
        loop {
            // Register interest *before* checking the map so an `expose` that lands between
            // the check and the await can't slip through the gap and leave us parked forever.
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

            if let Some(pending) = self.pending.lock().await.remove(&transfer_id) {
                return pending.tensor;
            }

            notified.await;
        }
    }

    /// This member's result of its all-reduce, once every member has handed over its tensor. The
    /// last to arrive runs it on `stream_id`; until then the others wait, as NCCL's members do.
    ///
    /// # Errors
    ///
    /// When the part does not fit its all-reduce's members, which fails the members already
    /// waiting too, and for every other member when the all-reduce panics in the last one.
    pub async fn all_reduce(
        &self,
        part: &LocalAllReducePart,
        tensor: FloatTensor<B>,
        stream_id: StreamId,
    ) -> Result<FloatTensor<B>, String> {
        let collective = part.collective;
        let (member, members) = (part.member as usize, part.members as usize);
        let (sender, result) = oneshot::channel();
        let complete = {
            let mut all_reduces = self.all_reduces.lock().await;
            let arrived = all_reduces
                .entry(collective)
                .or_insert_with(|| (0..members).map(|_| None).collect());
            if arrived.len() != members || arrived.get(member).is_none_or(Option::is_some) {
                let error = format!(
                    "member {member} of {members} does not fit all-reduce {collective:?} of {}",
                    arrived.len()
                );
                let waiting = all_reduces.remove(&collective).into_iter().flatten();
                for waiting in waiting.flatten() {
                    let _ = waiting.result.send(Err(error.clone()));
                }
                return Err(error);
            }
            arrived[member] = Some(ArrivedMember {
                tensor,
                result: sender,
            });
            match arrived.iter().all(Option::is_some) {
                true => all_reduces.remove(&collective),
                false => None,
            }
        };
        if let Some(arrived) = complete {
            let mut arrived: Vec<_> = arrived.into_iter().flatten().collect();
            // A native group takes its members in ascending device order, and every member
            // receives the same result in any.
            arrived.sort_by_key(|member| member.tensor.device().id());
            let (tensors, results): (Vec<_>, Vec<_>) = arrived
                .into_iter()
                .map(|member| (member.tensor, member.result))
                .unzip();
            let reduced = stream_id.executes(|| B::float_all_reduce(tensors, part.op));
            for (result, tensor) in results.into_iter().zip(reduced) {
                let _ = result.send(Ok(tensor));
            }
        }
        match result.await {
            Ok(reduced) => reduced,
            Err(_) => Err(format!(
                "all-reduce {collective:?} failed in its last member"
            )),
        }
    }

    /// Drop every primitive `session` exposed that no target ever took.
    ///
    /// Called when a session closes. In normal operation every exposed primitive is taken (and so
    /// removed) by its target, but if the source session tears down first — a crash, a dropped
    /// connection, a transfer whose target never runs — its entry would otherwise sit in the map
    /// for the server's lifetime, pinning device memory. Purging by source session reclaims it.
    pub async fn purge_session(&self, session: SessionId) {
        self.pending
            .lock()
            .await
            .retain(|_, pending| pending.source != session);
    }

    #[cfg(all(test, not(target_family = "wasm")))]
    pub(crate) async fn is_empty(&self) -> bool {
        self.pending.lock().await.is_empty()
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use burn_backend::{
        DType, Shape, TensorData, distributed::ReduceOperation, ops::FloatTensorOps,
    };
    use burn_flex::Flex;
    use burn_ir::{TensorId, TensorIr};
    use std::{task::Poll, time::Duration};

    fn part(collective: LocalCollectiveId, member: u32, members: u32) -> LocalAllReducePart {
        LocalAllReducePart {
            tensor: TensorIr::uninit(TensorId::new(member as u64), Shape::new([2]), DType::F32),
            out: TensorId::new(10 + member as u64),
            collective,
            member,
            members,
            op: ReduceOperation::Sum,
        }
    }

    fn tensor(values: [f32; 2]) -> FloatTensor<Flex> {
        Flex::float_from_data(TensorData::from(values), &Default::default())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn every_member_receives_the_reduction_once_the_last_arrives() {
        let comm = Arc::new(LocalCommService::<Flex>::new());
        let collective = LocalCollectiveId::random();
        let first = tokio::spawn({
            let comm = comm.clone();
            async move {
                let part = part(collective, 0, 2);
                comm.all_reduce(&part, tensor([1.0, -2.5]), StreamId::current())
                    .await
            }
        });
        let second = comm
            .all_reduce(
                &part(collective, 1, 2),
                tensor([3.5, 4.0]),
                StreamId::current(),
            )
            .await
            .unwrap();
        let first = first.await.unwrap().unwrap();

        for reduced in [first, second] {
            Flex::float_into_data(reduced)
                .await
                .unwrap()
                .assert_eq(&TensorData::from([4.5f32, 1.5]), false);
        }
    }

    #[tokio::test]
    async fn a_member_outside_its_all_reduce_is_refused() {
        let comm = LocalCommService::<Flex>::new();

        let refused = comm
            .all_reduce(
                &part(LocalCollectiveId::random(), 2, 2),
                tensor([1.0, 2.0]),
                StreamId::current(),
            )
            .await;

        assert!(refused.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_part_that_does_not_fit_fails_the_members_waiting() {
        let comm = LocalCommService::<Flex>::new();
        let collective = LocalCollectiveId::random();
        let first_part = part(collective, 0, 2);
        let first = comm.all_reduce(&first_part, tensor([1.0, 2.0]), StreamId::current());
        tokio::pin!(first);
        let waiting =
            std::future::poll_fn(|cx| Poll::Ready(first.as_mut().poll(cx).is_pending())).await;
        assert!(waiting);

        let duplicate = comm
            .all_reduce(
                &part(collective, 0, 2),
                tensor([3.0, 4.0]),
                StreamId::current(),
            )
            .await;
        let first = tokio::time::timeout(Duration::from_secs(10), first)
            .await
            .expect("the waiting member was left waiting");

        assert!(duplicate.is_err());
        assert!(first.is_err());
    }
}
