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
use burn_backend::tensor::FloatTensor;
use burn_ir::{BackendIr, HandleKind};
use burn_std::id::StreamId;
use std::collections::{HashMap, HashSet};
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

/// What fails an all-reduce, which decides whether one of the members still to come already knows.
#[derive(Clone, Copy)]
enum FailedBy {
    /// A member still to come that cannot take part, told by its own failure.
    MemberToCome,
    /// A part that does not fit, holding no member's place.
    Misfit,
}

/// Where one all-reduce stands, with its members' sessions, by member.
enum Rendezvous<B: BackendIr> {
    /// Its members so far, by member.
    Gathering {
        sessions: Vec<SessionId>,
        arrived: Vec<Option<ArrivedMember<B>>>,
    },
    /// Failed, kept until the members still to come are told so.
    Failed {
        sessions: Vec<SessionId>,
        error: String,
        remaining: usize,
    },
}

impl<B: BackendIr> Rendezvous<B> {
    fn sessions(&self) -> &[SessionId] {
        match self {
            Rendezvous::Gathering { sessions, .. } | Rendezvous::Failed { sessions, .. } => {
                sessions
            }
        }
    }
}

/// The all-reduces of this server's sessions that are gathering their members or failed.
struct AllReduces<B: BackendIr> {
    rendezvous: HashMap<LocalCollectiveId, Rendezvous<B>>,
    /// Kept for the server's lifetime: a part naming a closed session can arrive at any time.
    closed_sessions: HashSet<SessionId>,
}

impl<B: BackendIr> AllReduces<B> {
    /// Hand `member` over for `part`: every member once the last arrives, nothing while some are
    /// still to come.
    fn arrive(
        &mut self,
        part: &LocalAllReducePart,
        member: ArrivedMember<B>,
    ) -> Result<Option<Vec<ArrivedMember<B>>>, String> {
        let collective = part.collective;
        if let Some(closed) = part
            .sessions
            .iter()
            .find(|session| self.closed_sessions.contains(session))
        {
            return Err(format!(
                "all-reduce {collective:?} lost a member: {closed} closed"
            ));
        }
        let index = part.member as usize;
        let rendezvous =
            self.rendezvous
                .entry(collective)
                .or_insert_with(|| Rendezvous::Gathering {
                    sessions: part.sessions.clone(),
                    arrived: (0..part.members()).map(|_| None).collect(),
                });
        let arrived = match rendezvous {
            Rendezvous::Gathering { sessions, arrived } if *sessions == part.sessions => arrived,
            Rendezvous::Gathering { .. } => {
                let error =
                    format!("member {index} of all-reduce {collective:?} names other members");
                self.fail(collective, &part.sessions, FailedBy::Misfit, &error);
                return Err(error);
            }
            Rendezvous::Failed {
                error, remaining, ..
            } => {
                let error = error.clone();
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    self.rendezvous.remove(&collective);
                }
                return Err(error);
            }
        };
        if arrived[index].is_some() {
            let error = format!("member {index} of all-reduce {collective:?} arrived twice");
            self.fail(collective, &part.sessions, FailedBy::Misfit, &error);
            return Err(error);
        }
        arrived[index] = Some(member);
        if arrived.iter().any(Option::is_none) {
            return Ok(None);
        }
        match self.rendezvous.remove(&collective) {
            Some(Rendezvous::Gathering { arrived, .. }) => {
                Ok(Some(arrived.into_iter().flatten().collect()))
            }
            _ => unreachable!("a complete all-reduce is gathering"),
        }
    }

    /// Fail `collective` with `error`: the members waiting are told now, and the rest as they
    /// arrive, less the one failing it when that one is still to come. `sessions` are its members'
    /// when nothing has arrived for it yet.
    fn fail(
        &mut self,
        collective: LocalCollectiveId,
        sessions: &[SessionId],
        by: FailedBy,
        error: &str,
    ) {
        let settled = match by {
            FailedBy::MemberToCome => 1,
            FailedBy::Misfit => 0,
        };
        let (sessions, to_come) = match self.rendezvous.remove(&collective) {
            Some(Rendezvous::Gathering { sessions, arrived }) => {
                let members = arrived.len();
                let waiting: Vec<_> = arrived.into_iter().flatten().collect();
                let to_come = members - waiting.len();
                for member in waiting {
                    let _ = member.result.send(Err(error.to_string()));
                }
                (sessions, to_come)
            }
            Some(Rendezvous::Failed {
                sessions,
                remaining,
                ..
            }) => (sessions, remaining),
            None => (sessions.to_vec(), sessions.len()),
        };
        let remaining = to_come.saturating_sub(settled);
        if remaining > 0 {
            let error = error.to_string();
            self.rendezvous.insert(
                collective,
                Rendezvous::Failed {
                    sessions,
                    error,
                    remaining,
                },
            );
        }
    }

    /// Fail every all-reduce with a member in `session`, waiting or still to come, and refuse
    /// every later part that names it.
    fn abandon(&mut self, session: SessionId) {
        self.closed_sessions.insert(session);
        self.rendezvous.retain(|collective, rendezvous| {
            if !rendezvous.sessions().contains(&session) {
                return true;
            }
            if let Rendezvous::Gathering { arrived, .. } = rendezvous {
                for member in arrived.drain(..).flatten() {
                    let error =
                        format!("all-reduce {collective:?} lost a member: {session} closed");
                    let _ = member.result.send(Err(error));
                }
            }
            false
        });
    }
}

/// Rendezvous registry for same-host communication, keyed by [`LocalTransferId`].
pub(crate) struct LocalCommService<B: BackendIr> {
    /// Primitives exposed by source sessions, waiting to be taken by their target.
    pending: Mutex<HashMap<LocalTransferId, Pending<B>>>,
    all_reduces: Mutex<AllReduces<B>>,
    /// The devices this server hosts, which bound an all-reduce's members: each is a distinct one.
    hosted_devices: usize,
    /// Wakes takers whenever a new primitive is exposed.
    notify: Arc<Notify>,
}

impl<B: BackendIr> LocalCommService<B> {
    pub fn new(hosted_devices: usize) -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
            all_reduces: Mutex::new(AllReduces {
                rendezvous: HashMap::new(),
                closed_sessions: HashSet::new(),
            }),
            hosted_devices,
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
    /// When the part names more members than this server hosts devices, or a member past them;
    /// when it does not fit its all-reduce's members, which fails it for every member; when it
    /// already failed; for every other member when it panics in the last one; and when the
    /// session of any of its members closes.
    pub async fn all_reduce(
        &self,
        part: &LocalAllReducePart,
        tensor: FloatTensor<B>,
        stream_id: StreamId,
    ) -> Result<FloatTensor<B>, String> {
        let collective = part.collective;
        self.fits(part)?;
        let (sender, result) = oneshot::channel();
        let member = ArrivedMember {
            tensor,
            result: sender,
        };
        let complete = self.all_reduces.lock().await.arrive(part, member)?;
        if let Some(arrived) = complete {
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

    /// Fail this part's all-reduce for every member, when this one cannot take part.
    pub async fn fail(&self, part: &LocalAllReducePart, error: &str) {
        if self.fits(part).is_err() {
            return;
        }
        self.all_reduces.lock().await.fail(
            part.collective,
            &part.sessions,
            FailedBy::MemberToCome,
            error,
        );
    }

    /// Fail every all-reduce with a member in `session`, closed or stopped by a panic, whose other
    /// members would otherwise wait for it for good.
    pub async fn abandon_session(&self, session: SessionId) {
        self.all_reduces.lock().await.abandon(session);
    }

    /// Refuses a part that names more members than this server hosts devices, each member being a
    /// distinct one, or a member past them.
    fn fits(&self, part: &LocalAllReducePart) -> Result<(), String> {
        let members = part.members();
        if (part.member as usize) < members && members <= self.hosted_devices {
            return Ok(());
        }
        Err(format!(
            "member {} of {members} does not fit an all-reduce over the {} devices of this server",
            part.member, self.hosted_devices
        ))
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
        self.pending.lock().await.is_empty() && self.all_reduces.lock().await.rendezvous.is_empty()
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

    fn part(
        collective: LocalCollectiveId,
        member: u32,
        sessions: &[SessionId],
    ) -> LocalAllReducePart {
        LocalAllReducePart {
            tensor: TensorIr::uninit(TensorId::new(member as u64), Shape::new([2]), DType::F32),
            out: TensorId::new(10 + member as u64),
            collective,
            member,
            sessions: sessions.to_vec(),
            op: ReduceOperation::Sum,
        }
    }

    fn sessions(members: usize) -> Vec<SessionId> {
        (0..members).map(|_| SessionId::new()).collect()
    }

    fn tensor(values: [f32; 2]) -> FloatTensor<Flex> {
        Flex::float_from_data(TensorData::from(values), &Default::default())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn every_member_receives_the_reduction_once_the_last_arrives() {
        let comm = Arc::new(LocalCommService::<Flex>::new(2));
        let collective = LocalCollectiveId::random();
        let members = sessions(2);
        let first = tokio::spawn({
            let comm = comm.clone();
            let part = part(collective, 0, &members);
            async move {
                comm.all_reduce(&part, tensor([1.0, -2.5]), StreamId::current())
                    .await
            }
        });
        let second = comm
            .all_reduce(
                &part(collective, 1, &members),
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
        let comm = LocalCommService::<Flex>::new(2);

        for (member, members) in [(2, sessions(2)), (0, sessions(3))] {
            let refused = comm
                .all_reduce(
                    &part(LocalCollectiveId::random(), member, &members),
                    tensor([1.0, 2.0]),
                    StreamId::current(),
                )
                .await;

            assert!(refused.is_err());
        }
        assert!(comm.is_empty().await);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_part_that_does_not_fit_fails_the_members_waiting() {
        let comm = LocalCommService::<Flex>::new(2);
        let collective = LocalCollectiveId::random();
        let members = sessions(2);
        let first_part = part(collective, 0, &members);
        let first = comm.all_reduce(&first_part, tensor([1.0, 2.0]), StreamId::current());
        tokio::pin!(first);
        let waiting =
            std::future::poll_fn(|cx| Poll::Ready(first.as_mut().poll(cx).is_pending())).await;
        assert!(waiting);

        let duplicate = comm
            .all_reduce(
                &part(collective, 0, &members),
                tensor([3.0, 4.0]),
                StreamId::current(),
            )
            .await;
        let first = tokio::time::timeout(Duration::from_secs(10), first)
            .await
            .expect("the waiting member was left waiting");

        assert!(duplicate.is_err());
        assert!(first.is_err());

        let late = tokio::time::timeout(
            Duration::from_secs(10),
            comm.all_reduce(
                &part(collective, 1, &members),
                tensor([5.0, 6.0]),
                StreamId::current(),
            ),
        )
        .await
        .expect("a member arriving after the failure was left waiting");
        assert!(late.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_member_that_cannot_take_part_fails_the_one_waiting() {
        let comm = LocalCommService::<Flex>::new(2);
        let collective = LocalCollectiveId::random();
        let members = sessions(2);
        let first_part = part(collective, 0, &members);
        let first = comm.all_reduce(&first_part, tensor([1.0, 2.0]), StreamId::current());
        tokio::pin!(first);
        let waiting =
            std::future::poll_fn(|cx| Poll::Ready(first.as_mut().poll(cx).is_pending())).await;
        assert!(waiting);

        comm.fail(
            &part(collective, 1, &members),
            "its input was never produced",
        )
        .await;
        let first = tokio::time::timeout(Duration::from_secs(10), first)
            .await
            .expect("the waiting member was left waiting");

        assert!(first.is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_member_session_that_closes_before_arriving_fails_the_all_reduce() {
        let comm = LocalCommService::<Flex>::new(2);
        let collective = LocalCollectiveId::random();
        let members = sessions(2);
        let first_part = part(collective, 0, &members);
        let first = comm.all_reduce(&first_part, tensor([1.0, 2.0]), StreamId::current());
        tokio::pin!(first);
        let waiting =
            std::future::poll_fn(|cx| Poll::Ready(first.as_mut().poll(cx).is_pending())).await;
        assert!(waiting);

        comm.abandon_session(members[1]).await;
        let first = tokio::time::timeout(Duration::from_secs(10), first)
            .await
            .expect("the waiting member was left waiting");
        let late = tokio::time::timeout(
            Duration::from_secs(10),
            comm.all_reduce(
                &part(collective, 1, &members),
                tensor([3.0, 4.0]),
                StreamId::current(),
            ),
        )
        .await
        .expect("the closed session's own part was left waiting");

        assert!(first.is_err());
        assert!(late.is_err());
        assert!(comm.is_empty().await);
    }
}
