//! Per-device service: one [`DeviceService`] per physical device, owning every session's
//! interpreter and running its tasks on the device's cubecl runner thread.
//!
//! This replaces the old one-OS-thread-per-session worker. Instead of spawning a thread per
//! session, every session targeting a device is a [`SessionState`] entry in a single
//! [`ServerDeviceService`], and tasks are dispatched onto the device's runner thread through the
//! optimized [`DeviceHandle`] channel (the same fire-and-forget channel the client already rides).
//! The service owns the session map by `&mut self` on the runner thread, so the map needs no
//! locking and the service is the natural home for any future cross-session scheduling/priority.
//!
//! ## Why [`DeviceServiceStage::Upstream`]
//!
//! The server *interprets* tasks (decode op IR, look up handles, hand them to the inner backend)
//! ahead of the inner backend's own kernel dispatch. Running on the Upstream stage gives the
//! server its own runner thread per device, distinct from the backend's Downstream thread, so
//! interpretation overlaps with kernel dispatch.
//!
//! ## Blocking tasks
//!
//! Some tasks block the runner thread synchronously: a collective op (all-reduce) parks inside
//! `register_op` until every device reaches the barrier, a same-host transfer's
//! [`RegisterTensorLocal`](Task::RegisterTensorLocal) blocks on the rendezvous, and a cross-server
//! [`RegisterTensorRemote`](Task::RegisterTensorRemote) blocks on the download. With one runner
//! thread per device this is correct for the common one-session-per-device configuration and for
//! all same-host transfers (source and target always live on *different* devices, hence different
//! runner threads). Multiple sessions on the *same* device sharing a runner is best-effort: a
//! blocking task on one stalls the others until it unblocks. Detached readbacks (`ReadTensor`,
//! `ExposeTensorRemote`) and fetch-channel sends are spawned onto the runtime so they never block
//! the runner.

use std::collections::HashMap;
use std::sync::Arc;

use burn_backend::tensor::Device;
use burn_backend::{DeviceId, DeviceService, DeviceServiceStage, ServerUtilitiesHandle, TensorData};
use burn_communication::{Protocol, external_comm::ExternalCommService};
use burn_ir::{BackendIr, HandleKind, TensorId, TensorIr};
use burn_router::{RouterClient, TensorInterpreter};
use burn_std::id::StreamId;
use tokio::runtime::Handle;

use crate::server::service::FetchSender;
use crate::shared::{SessionId, Task, TaskResponseContent};

/// Everything constant for the lifetime of one session on a device.
struct SessionState<B: BackendIr> {
    interpreter: TensorInterpreter<B>,
    fetch_sender: FetchSender,
}

/// One service per physical device. Owns the interpreters of every session pinned to that device
/// and runs their tasks on the device's runner thread.
///
/// Always constructed on a runtime thread and inserted into a [`DeviceHandle`] via
/// [`DeviceHandle::insert`](burn_backend::DeviceHandle::insert) — never via [`DeviceService::init`]
/// — because it needs the tokio [`Handle`] and the comm services, which `init` can't provide.
pub(crate) struct ServerDeviceService<B: BackendIr, P: Protocol> {
    device: Device<B>,
    /// Runtime used to `spawn` the async parts of a task (detached readbacks, cross-server
    /// hand-offs, fetch-channel sends). The runner itself never blocks on it — the blocking transfer
    /// halves are resolved off the runner, in the submit forwarder.
    runtime: Handle,
    external_comm: Arc<ExternalCommService<B, P>>,
    sessions: HashMap<SessionId, SessionState<B>>,
}

impl<B: BackendIr, P: Protocol> ServerDeviceService<B, P> {
    pub(crate) fn new(
        device: Device<B>,
        runtime: Handle,
        external_comm: Arc<ExternalCommService<B, P>>,
    ) -> Self {
        Self {
            device,
            runtime,
            external_comm,
            sessions: HashMap::new(),
        }
    }

    /// Create a session's interpreter, registering its fetch channel. Runs on the runner thread,
    /// ordered before any task forwarded for the session.
    pub(crate) fn create_session(&mut self, session_id: SessionId, fetch_sender: FetchSender) {
        let runner = TensorInterpreter::new(self.device.clone());
        self.sessions.insert(
            session_id,
            SessionState {
                interpreter: runner,
                fetch_sender,
            },
        );
    }

    /// Tear a session down: flush and drop its interpreter (freeing its tensors) and hand freed
    /// memory back to the allocator.
    ///
    /// Mirrors the old per-session worker's close path: dropping `SessionState` drops the session's
    /// [`FetchSender`], which closes the fetch writer's channel and ends it. Reclaiming any same-host
    /// transfers the session left exposed is done by the caller (`close`) off the runner, so this
    /// stays a non-blocking runner task.
    pub(crate) fn remove_session(&mut self, session_id: SessionId) {
        if let Some(SessionState {
            interpreter: runner,
            ..
        }) = self.sessions.remove(&session_id)
        {
            let device = runner.device();

            // Flush outstanding backend work before dropping the runner so the session's tensors
            // aren't freed with work still queued against them.
            if let Err(err) = runner.sync() {
                log::warn!("runner.sync() at session {session_id} close failed: {err:?}");
            }
            if let Err(err) = B::sync(&device) {
                log::warn!("B::sync(device) at session {session_id} close failed: {err:?}");
            }

            // Drop the runner: frees every tensor handle back to the backend allocator. Must happen
            // before `memory_cleanup`, otherwise the memory is still live and nothing is reclaimable.
            drop(runner);
            B::memory_cleanup(&device);
        }
    }

    /// Finish a same-host transfer: move the source's already-resolved primitive `kind` onto this
    /// session's device and register it under `new_id`.
    ///
    /// The blocking half — waiting on the [`local_comm`](LocalCommService) rendezvous — happens off
    /// the runner, in the submit forwarder; by the time this runs the primitive is in hand, so the
    /// runner only does the (non-blocking) device move. Ordered, like every task, after the ops the
    /// forwarder submitted before it.
    pub(crate) fn register_local(
        &mut self,
        session_id: SessionId,
        stream_id: StreamId,
        new_id: TensorId,
        kind: HandleKind<B>,
    ) {
        let Some(state) = self.sessions.get_mut(&session_id) else {
            log::error!("RegisterTensorLocal for unknown session {session_id} dropped");
            return;
        };
        stream_id.executes(|| state.interpreter.register_tensor_to_device(new_id, kind));
    }

    /// Source side of a same-host transfer: grab the device-resident primitive for `tensor` (no
    /// host readback). Returns `None` if the session is gone.
    ///
    /// Only this device-touching half runs on the runner; the forwarder then parks the primitive in
    /// the [`local_comm`](LocalCommService) rendezvous *off* the runner, so the runner doesn't block
    /// on the (async) registry insert. Ordered, like every task, after the op that produced
    /// `tensor`, so the handle is present.
    pub(crate) fn get_tensor(
        &self,
        session_id: SessionId,
        stream_id: StreamId,
        tensor: TensorIr,
    ) -> Option<HandleKind<B>> {
        let state = self.sessions.get(&session_id)?;
        Some(stream_id.executes(|| state.interpreter.get_tensor(&tensor)))
    }

    /// Finish a cross-server transfer: register the already-downloaded `data` under `new_id`.
    ///
    /// As with [`register_local`](Self::register_local), the blocking half — the cross-server
    /// download — happens off the runner in the submit forwarder, so the runner only does the
    /// (non-blocking) registration on the client stream that will consume `new_id`.
    pub(crate) fn register_remote(
        &mut self,
        session_id: SessionId,
        stream_id: StreamId,
        new_id: TensorId,
        data: TensorData,
    ) {
        let Some(state) = self.sessions.get_mut(&session_id) else {
            log::error!("RegisterTensorRemote for unknown session {session_id} dropped");
            return;
        };
        stream_id.executes(|| state.interpreter.register_tensor_data_id(new_id, data));
    }

    /// Execute a single [`Task`] against `session_id`'s state.
    ///
    /// Runs on the device's runner thread, and **never blocks it**: sync work is wrapped in
    /// [`StreamId::executes`](burn_std::id::StreamId::executes) so the runner's thread-local stream
    /// id matches the one the client assigned to the op, and the only async work — detached
    /// readbacks, cross-server hand-offs and fetch-channel sends — is `spawn`ed onto the runtime.
    /// The blocking parts of transfers (the same-host rendezvous, the cross-server download, and the
    /// source-side `expose`) are all resolved off the runner in the submit forwarder; see
    /// [`register_local`](Self::register_local) / [`register_remote`](Self::register_remote) /
    /// [`get_tensor`](Self::get_tensor). Keeping the runner non-blocking is what stops a transfer
    /// from wedging it (and, via the device handle's backpressure, the whole runtime).
    pub(crate) fn process_task(&mut self, session_id: SessionId, task: Task) {
        // Clone the shared (cheap `Arc` / `Handle`) handles up front so the per-session `&mut`
        // borrow below doesn't conflict with the service-wide fields.
        let external_comm = self.external_comm.clone();
        let runtime = self.runtime.clone();

        let Some(state) = self.sessions.get_mut(&session_id) else {
            log::error!("Task for unknown session {session_id} dropped");
            return;
        };
        let runner = &state.interpreter;
        let fetch_sender = &state.fetch_sender;

        match task {
            Task::RegisterOperation(stream_id, op) => {
                stream_id.executes(|| runner.register_op(op));
            }
            Task::RegisterTensor(stream_id, id, data) => {
                stream_id.executes(|| runner.register_tensor_data_id(id, data));
            }
            // The transfer tasks that would block the runner — the same-host rendezvous
            // (`RegisterTensorLocal`), the cross-server download (`RegisterTensorRemote`) and the
            // source-side `expose` (`ExposeTensorLocal`, whose rendezvous insert is awaited off the
            // runner) — are all resolved in the submit forwarder and reach the runner only as the
            // non-blocking [`register_local`](Self::register_local) /
            // [`register_remote`](Self::register_remote) / [`get_tensor`](Self::get_tensor) calls.
            Task::RegisterTensorLocal { .. }
            | Task::RegisterTensorRemote(..)
            | Task::ExposeTensorLocal { .. } => {
                unreachable!("blocking transfer halves are resolved in the submit forwarder, not on the runner")
            }
            Task::ExposeTensorRemote {
                stream_id,
                tensor,
                count,
                transfer_id,
            } => {
                log::trace!("Exposing tensor (transfer {transfer_id:?})");
                // Like `ReadTensor`: the sync part of `read_tensor_async` runs in order to preserve
                // stream ordering, but the readback + expose are detached so a cross-server hand-off
                // doesn't stall the runner on a GPU→host copy.
                let fut = stream_id.executes(|| runner.read_tensor_async(tensor));
                runtime.spawn(async move {
                    match fut.await {
                        Ok(data) => external_comm.expose_data(data, count, transfer_id).await,
                        Err(e) => log::error!(
                            "read_tensor_async for transfer {transfer_id:?} failed: {e:?}"
                        ),
                    }
                });
            }
            Task::Seed(seed) => runner.seed(seed),
            Task::ReadTensor(request_id, stream_id, tensor) => {
                // `read_tensor_async` is sync at construction (locks the context, captures the
                // tensor's position in the command stream) and returns a future for the host
                // readback. Run the sync part in order, then detach the readback so awaiting it
                // doesn't stall the runner on the GPU→host copy. The client demuxes responses by
                // request id, so out-of-order completion is fine.
                let fut = stream_id.executes(|| runner.read_tensor_async(tensor));
                let sender = fetch_sender.clone();
                runtime.spawn(async move {
                    let data = fut.await;
                    sender
                        .send(request_id, TaskResponseContent::ReadTensor(data))
                        .await;
                });
            }
            Task::SyncBackend(request_id, stream_id) => {
                let res = stream_id.executes(|| runner.sync());
                let sender = fetch_sender.clone();
                runtime.spawn(async move {
                    sender
                        .send(request_id, TaskResponseContent::SyncBackend(res))
                        .await;
                });
            }
            Task::DTypeUsage(request_id, dtype) => {
                let res = runner.dtype_usage(dtype);
                let sender = fetch_sender.clone();
                runtime.spawn(async move {
                    sender
                        .send(request_id, TaskResponseContent::DTypeUsage(res))
                        .await;
                });
            }
        }
    }
}

impl<B: BackendIr, P: Protocol> DeviceService for ServerDeviceService<B, P> {
    fn init(_device_id: DeviceId) -> Self {
        // We always build the service with `DeviceHandle::insert` so it can carry the tokio handle
        // and comm services; `init` (which only gets a `DeviceId`) is never used.
        unreachable!("ServerDeviceService is constructed via DeviceHandle::insert, not init");
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        Arc::new(())
    }

    fn stage() -> DeviceServiceStage {
        DeviceServiceStage::Upstream
    }
}
